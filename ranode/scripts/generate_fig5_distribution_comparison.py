"""
Generate Fig. 5-style distribution comparison plots using the artifacts
produced by the full LAW pipeline run.

The script
  * selects a trained signal model `model_S.pt` (best validation loss by default),
  * samples from the learned density in feature space,
  * converts both data and samples back to the original (H, L, H+L, H-L)
    coordinates using the recorded preprocessing parameters, and
  * compares the learned signal to the true signal/background that actually
    entered training (after all masks).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure we can import project modules regardless of invocation location.
REPO_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ranode_path = REPO_ROOT / "ranode"
if str(ranode_path) not in sys.path:
    sys.path.insert(0, str(ranode_path))

from src.data_prep.utils import (
    inverse_logit_transform,
    inverse_standardize,
)
from src.models.model_S import flows_model_RQS


def _parse_ratio_component(component: str) -> Tuple[int, float, str]:
    """
    Convert directory token like `s_index_11_ratio_0p028547`
    into (s_index, ratio_float).
    """
    match = re.match(r"s_index_(\d+)_ratio_(.+)", component)
    if not match:
        raise ValueError(f"Unrecognised s_index token: {component}")
    s_index = int(match.group(1))
    ratio_str = match.group(2).replace("p", ".")
    return s_index, float(ratio_str), match.group(2)


def _extract_settings(model_path: Path) -> Dict[str, str]:
    """
    Extract configuration tokens from a RNodeTemplate directory structure.
    """
    settings: Dict[str, str] = {}
    for part in model_path.parts:
        if part.startswith("use_full_stats_"):
            settings["use_full_stats"] = part.split("_")[-1]
        elif part.startswith("mx_"):
            settings["mx"] = part.split("_")[1]
        elif part.startswith("my_"):
            settings["my"] = part.split("_")[1]
        elif part.startswith("ensemble_"):
            settings["ensemble"] = part.split("_")[1]
        elif part.startswith("fold_split_num_"):
            settings["fold_split_num"] = part.split("_")[3]
        elif part.startswith("fold_split_seed_"):
            settings["fold_split_seed"] = part.split("_")[3]
        elif part.startswith("train_seed_"):
            settings["train_seed"] = part.split("_")[2]
        elif part.startswith("use_perfect_bkg_model_"):
            settings["use_perfect_bkg_model"] = part.split("_")[-1]
        elif part.startswith("use_bkg_model_gen_data_"):
            settings["use_bkg_model_gen_data"] = part.split("_")[-1]
        elif part.startswith("s_index_"):
            s_index, ratio_value, ratio_token = _parse_ratio_component(part)
            settings["s_index"] = str(s_index)
            settings["s_ratio_value"] = ratio_value
            settings["s_ratio_token"] = ratio_token
        elif part.startswith("w_"):
            settings["w_token"] = part  # keep original token for logging
    required = [
        "use_full_stats",
        "mx",
        "my",
        "ensemble",
        "s_index",
        "s_ratio_token",
        "fold_split_num",
        "fold_split_seed",
    ]
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(
            f"Could not extract required settings {missing} from path {model_path}"
        )
    # Default flags if absent (older outputs may omit)
    settings.setdefault("use_perfect_bkg_model", "False")
    settings.setdefault("use_bkg_model_gen_data", "False")
    return settings


def _locate_best_model(model_root: Path) -> Path:
    """
    Among all metadata.json files underneath model_root,
    choose the one with the smallest validation loss.
    """
    best_path: Path | None = None
    best_loss = math.inf
    for metadata_path in model_root.glob("**/metadata.json"):
        with metadata_path.open() as f:
            metadata = json.load(f)
        val_losses = metadata.get("min_val_loss_list") or metadata.get("val_loss", [])
        if not val_losses:
            continue
        candidate = min(val_losses)
        if candidate < best_loss:
            model_candidate = metadata_path.parent / "model_S.pt"
            if model_candidate.exists():
                best_loss = candidate
                best_path = model_candidate
    if best_path is None:
        raise FileNotFoundError(
            f"No model found under {model_root}. "
            "Ensure the LAW pipeline has produced RNodeTemplate outputs."
        )
    return best_path


def _recover_original_space(data: np.ndarray, pre_params: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Reverse the preprocessing applied in `PreprocessingFold`.

    Parameters
    ----------
    data : np.ndarray
        Array with shape (N, 6) where column ordering is
        [time_shifted, features (standardised/logit), label].
    pre_params : dict
        Dictionary with 'mean', 'std', 'min', 'max' vectors for the four features.
    """
    restored = data.copy()
    restored[:, 0] += 3.5  # revert the -3.5 shift applied before training
    restored[:, 1:-1] = inverse_standardize(
        restored[:, 1:-1],
        pre_params["mean"],
        pre_params["std"],
    )
    restored[:, 1:-1] = inverse_logit_transform(
        restored[:, 1:-1],
        pre_params["min"],
        pre_params["max"],
    )
    return restored


def _load_preprocessed_datasets(
    base_output: Path,
    settings: Dict[str, str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load SR data (train+test) prepared for the signal model and return
    the concatenated array along with preprocessing parameters.
    """
    use_full_stats = settings["use_full_stats"]
    mx = settings["mx"]
    my = settings["my"]
    ensemble = settings["ensemble"]
    s_index = settings["s_index"]
    ratio_token = settings["s_ratio_token"]
    fold_split_num = settings["fold_split_num"]
    fold_split_seed = settings["fold_split_seed"]

    preproc_dir = (
        base_output
        / "PreprocessingFold"
        / f"use_full_stats_{use_full_stats}"
        / f"mx_{mx}"
        / f"my_{my}"
        / f"ensemble_{ensemble}"
        / f"s_index_{s_index}_ratio_{ratio_token}"
        / f"fold_split_num_{fold_split_num}"
        / f"fold_split_seed_{fold_split_seed}"
    )
    train_path = preproc_dir / "data_SR_data_trainval_model_S.npy"
    test_path = preproc_dir / "data_SR_data_test_model_S.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Could not find preprocessed data for model at {preproc_dir}. "
            "Have you run the full PreprocessingFold stage?"
        )

    train = np.load(train_path)
    test = np.load(test_path)
    combined = np.concatenate([train, test], axis=0)

    # Load preprocessing parameters from ProcessBkg output
    process_bkg_root = base_output / "ProcessBkg"
    pre_params_candidates = list(
        (process_bkg_root / f"use_full_stats_{use_full_stats}").glob(
            "**/pre_parameters.json"
        )
    )
    if not pre_params_candidates:
        raise FileNotFoundError(
            f"Missing preprocessing parameters under {process_bkg_root}. "
            "Run ProcessBkg before generating Fig. 5."
        )
    pre_params_path = pre_params_candidates[0]
    with pre_params_path.open() as f:
        pre_params = json.load(f)
    # Ensure arrays are numpy arrays
    pre_params = {k: np.asarray(v) for k, v in pre_params.items()}
    return combined, pre_params


def _prepare_true_distributions(
    sr_data_original_space: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the restored SR data into (signal, background) components.
    """
    labels = sr_data_original_space[:, -1]
    signal = sr_data_original_space[labels == 1]
    background = sr_data_original_space[labels == 0]
    if signal.size == 0:
        raise RuntimeError("No signal events found after preprocessing mask.")
    if background.size == 0:
        raise RuntimeError("No background events found after preprocessing mask.")
    return signal, background


def _sample_learned_signal(
    model_path: Path,
    device: torch.device,
    pre_params: Dict[str, np.ndarray],
    num_samples: int,
) -> np.ndarray:
    """
    Sample events from the trained normalising flow and map them
    back to the original feature space.
    """
    model = flows_model_RQS(device=device, num_features=5, context_features=None)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    torch.manual_seed(0)
    with torch.no_grad():
        samples = model.sample(num_samples)
    samples = samples.cpu().numpy()

    # Insert dummy label column (set to 1) so that inverse transformation works
    augmented = np.concatenate(
        [samples[:, :5], np.ones((samples.shape[0], 1), dtype=samples.dtype)], axis=1
    )
    restored = _recover_original_space(augmented, pre_params)
    return restored[:, :5]  # drop label


def _summarise_distribution(name: str, data: np.ndarray) -> None:
    feature_labels = ["time", "H", "L", "H+L", "H-L"]
    print(f"\n{name}")
    for idx, label in enumerate(feature_labels[1:], start=1):
        column = data[:, idx]
        print(
            f"  {label:<8} mean={column.mean(): .6f}  std={column.std(): .6f}"
        )


def _plot_comparison(
    true_signal: np.ndarray,
    true_background: np.ndarray,
    learned_signal: np.ndarray,
    output_path: Path,
) -> None:
    feature_names = ["H detector", "L detector", "H+L", "H-L"]
    feature_indices = [1, 2, 3, 4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, idx, title in zip(axes, feature_indices, feature_names):
        stacks = [
            true_signal[:, idx],
            true_background[:, idx],
            learned_signal[:, idx],
        ]
        combined = np.concatenate(stacks)
        vmin, vmax = np.percentile(combined, [0.5, 99.5])
        bins = np.linspace(vmin, vmax, 60)

        ax.hist(
            true_background[:, idx],
            bins=bins,
            alpha=0.5,
            color="#ffb463",
            label="Background (SR)",
            density=True,
        )
        ax.hist(
            true_signal[:, idx],
            bins=bins,
            alpha=0.5,
            color="#5dade2",
            label="True signal (SR)",
            density=True,
        )
        ax.hist(
            learned_signal[:, idx],
            bins=bins,
            histtype="step",
            color="black",
            linewidth=1.8,
            label="Learned signal",
            density=True,
        )
        ax.set_xlabel(title)
        ax.set_ylabel("Normalised counts")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    fig.suptitle("Signal distribution comparison in SR", fontsize=16, y=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Fig. 5 style plots from LAW pipeline outputs."
    )
    parser.add_argument(
        "--version",
        default="test_full",
        help="Name of the LAW version directory (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional path to a specific model_S.pt. "
        "If omitted, the model with the best validation loss is chosen automatically.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (e.g., 'cpu', 'mps', 'cuda'). "
        "Default picks 'mps' if available else CPU.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10_000,
        help="Number of samples to draw from the learned density.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for output PDF path.",
    )
    args = parser.parse_args()

    version_dir = Path(REPO_ROOT) / "output" / f"version_{args.version}"
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Expected pipeline outputs under {version_dir}, but nothing was found."
        )

    model_root = version_dir / "RNodeTemplate"
    if args.model:
        model_path = Path(args.model).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Specified model path does not exist: {model_path}")
    else:
        model_path = _locate_best_model(model_root)

    settings = _extract_settings(model_path)
    print(f"Selected model: {model_path}")
    print(f"Settings extracted from path: {settings}")

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    sr_data_preprocessed, pre_params = _load_preprocessed_datasets(version_dir, settings)
    sr_data_original = _recover_original_space(sr_data_preprocessed, pre_params)
    true_signal, true_background = _prepare_true_distributions(sr_data_original)

    learned_signal = _sample_learned_signal(
        model_path=model_path,
        device=device,
        pre_params=pre_params,
        num_samples=args.num_samples,
    )

    _summarise_distribution("True signal (SR)", true_signal)
    _summarise_distribution("Background (SR)", true_background)
    _summarise_distribution("Learned signal samples", learned_signal)

    output_pdf = (
        Path(args.output).resolve()
        if args.output
        else version_dir / "fig5_distribution_comparison.pdf"
    )
    _plot_comparison(true_signal, true_background, learned_signal, output_pdf)
    print(f"\nSaved comparison plot to {output_pdf}")


if __name__ == "__main__":
    main()
