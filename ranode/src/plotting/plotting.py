import numpy as np
import pandas as pd
import os
from quickstats.plots import General1DPlot, TwoPanel1DPlot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial
from quickstats.plots import DataModelingPlot
from quickstats.plots.colors import get_cmap, get_rgba
from quickstats.concepts import Histogram1D
from quickstats.maths.histograms import bin_center_to_bin_edge
from quickstats.plots.variable_distribution_plot import VariableDistributionPlot


def mu2sig(mu, B: int):
    # change this to the actual number of background events used in training + validatio
    S = B * mu
    return S / B**0.5


def sig2mu(sig, B):
    # change this to the actual number of background events used in training + validation
    S = sig * B**0.5
    return S / B


def plot_mu_scan_results(
    dfs,
    metadata,
    output_path,
):

    # -------------------- plotting settings --------------------
    colors = get_cmap("simple_contrast").colors
    styles = {
        "plot": {"marker": "o"},
        "legend": {"fontsize": 15},
        "ratio_frame": {"height_ratios": (2, 1), "hspace": 0.05},
    }
    styles_map = {
        "true": {
            "plot": {"color": "hdbs:spacecadet"},
            "fill_between": {"color": "none"},
        },
        "predicted": {
            "plot": {"color": "hdbs:pictorialcarmine"},
            "fill_between": {
                "facecolor": get_rgba(colors[0], 0.2),
                "alpha": None,
                "edgecolor": get_rgba(colors[0], 0.9),
            },
        },
    }
    config = {
        "error_on_top": False,
        "inherit_color": False,
        # 'draw_legend': False,
    }
    label_map = {
        "true": "Truth",
        "predicted": "Predicted",
    }

    # -------------------- making plots --------------------
    mx = metadata["mx"]
    my = metadata["my"]
    num_B = metadata["num_B"]
    use_full_stats = metadata["use_full_stats"]
    use_perfect_bkg_model = metadata["use_perfect_modelB"]
    use_bkg_model_gen_data = metadata["use_modelB_genData"]

    if use_full_stats:
        text = f"Full stats"
    else:
        text = f"Lumi matched"
    if use_perfect_bkg_model:
        text += ", model B trained in SR"
    if use_bkg_model_gen_data:
        text += ", use model B to generate bkgs in data"

    text += f"//Signal at $(m_X, m_Y) = ({mx}, {my})$ GeV"

    xticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
    xticklabels = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "5"]
    xticks2 = mu2sig(np.array(xticks), B=num_B)
    xticklabels2 = [0.03, 0.07, 0.17, 0.35, 0.7, 1.74, 3.49, 17.43]
    # xticklabels2 = [str(round(v, 2)) for v in xticks2]
    xlabel = r"$\mu_{inj}\,(\%)$"
    plotter = General1DPlot(
        dfs, styles=styles, styles_map=styles_map, label_map=label_map, config=config
    )
    plotter.add_text(text, 0.05, 0.95, fontsize=18)

    ax = plotter.draw(
        "x",
        "y",
        targets=["true", "predicted"],
        yerrloattrib="yerrlo",
        yerrhiattrib="yerrhi",
        xlabel=xlabel,
        ylabel=r"$\hat{\mu}\,(\%)$",
        ymin=2e-5,
        ymax=0.1,
        logx=True,
        logy=True,
        offset_error=False,
        legend_order=["true", "predicted"],
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    mu2sig_f = partial(mu2sig, B=num_B)
    sig2mu_f = partial(sig2mu, B=num_B)
    ax2 = ax.secondary_xaxis("top", functions=(mu2sig_f, sig2mu_f))
    ax2.tick_params(
        axis="x",
        which="major",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.tick_params(
        axis="x",
        which="minor",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.set_xticks(xticks2)
    ax2.set_xticklabels(xticklabels2)
    ax2.set_xlabel(r"$S/\sqrt{B}$", labelpad=10, fontsize=18)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_mu_scan_results_multimodels(
    dfs,
    metadata,
    output_path,
):

    # -------------------- plotting settings --------------------
    colors = get_cmap("simple_contrast").colors
    styles = {
        "plot": {"marker": "o"},
        "legend": {"fontsize": 18},
        "ratio_frame": {"height_ratios": (2, 1), "hspace": 0.05},
    }
    styles_map = {
        "true": {
            "plot": {"color": "hdbs:spacecadet"},
            "fill_between": {"color": "none"},
        },
    }

    key_list = [key for key in dfs.keys() if key != "true"]

    for index, key in enumerate(key_list):
        styles_map[key] = {
            "plot": {"color": colors[index]},
            "fill_between": {
                "facecolor": get_rgba(colors[index], 0.2),
                "alpha": None,
                "edgecolor": get_rgba(colors[index], 0.9),
            },
        }

    config = {
        "error_on_top": False,
        "inherit_color": False,
        # 'draw_legend': False,
    }

    # -------------------- making plots --------------------
    mx = metadata["mx"]
    my = metadata["my"]
    num_B = metadata["num_B"]
    use_full_stats = metadata["use_full_stats"]
    num_ensemble = metadata["num_ensemble"]
    label_map = metadata["label_map"]

    # if use_full_stats:
    #     text = f"Full stats, {num_ensemble} ensembles"
    # else:
    #     text = f"Lumi matched, {num_ensemble} ensembles"

    # text += f"//Signal at $(m_X, m_Y) = ({mx}, {my})$ GeV"

    xticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
    xticklabels = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "5"]
    xticks2 = mu2sig(np.array(xticks), B=num_B)
    xticklabels2 = [0.03, 0.07, 0.17, 0.35, 0.7, 1.74, 3.49, 17.43]
    # xticklabels2 = [str(round(v, 2)) for v in xticks2]
    xlabel = r"$\mu_{inj}\,(\%)$"
    plotter = General1DPlot(
        dfs, styles=styles, styles_map=styles_map, label_map=label_map, config=config
    )
    # plotter.add_text(text, 0.05, 0.95, fontsize=18)

    ax = plotter.draw(
        "x",
        "y",
        targets=dfs.keys(),
        yerrloattrib="yerrlo",
        yerrhiattrib="yerrhi",
        xlabel=xlabel,
        ylabel=r"$\hat{\mu}\,(\%)$",
        ymin=2e-5,
        ymax=0.1,
        logx=True,
        logy=True,
        offset_error=False,
        # legend_order=["true", "predicted"],
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    mu2sig_f = partial(mu2sig, B=num_B)
    sig2mu_f = partial(sig2mu, B=num_B)
    ax2 = ax.secondary_xaxis("top", functions=(mu2sig_f, sig2mu_f))
    ax2.tick_params(
        axis="x",
        which="major",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.tick_params(
        axis="x",
        which="minor",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.set_xticks(xticks2)
    ax2.set_xticklabels(xticklabels2)
    ax2.set_xlabel(r"$S/\sqrt{B}$", labelpad=10, fontsize=18)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_event_feature_distribution(dfs, misc, plot_options, output_path):

    mx = misc["mx"]
    my = misc["my"]
    numB = misc["numB"]
    use_full_stats = misc["use_full_stats"]
    use_perfect_modelB = misc["use_perfect_modelB"]
    use_modelB_genData = misc["use_modelB_genData"]
    columns = misc["columns"]

    # text = f"Signal at ({mx}, {my}) GeV \nmu_true = {mu_true}%, significance = {sig_significance}, mu_test = {mu_test}%"
    # if use_full_stats:
    #     text += "\n full stats"
    # else:
    #     text += "\n lumi matched"

    # if use_perfect_modelB:
    #     text += ", model B trained in SR"
    # if use_modelB_genData:
    #     text += ", use model B to generate bkgs in data"
    # if not use_modelB_genData and not use_perfect_modelB:
    #     text += ", model B trained in CR"

    plotter = VariableDistributionPlot(dfs, plot_options=plot_options)
    # plotter.add_text(text, 0.05, 0.95, fontsize=18)

    with PdfPages(output_path) as pdf:
        for feature in columns:
            axis = plotter.draw(
                feature,
                logy=False,
                bins=np.linspace(
                    dfs["background"][feature].min(),
                    dfs["background"][feature].max(),
                    101,
                ),
                show_error=True,
                comparison_options=None,
            )
            axis.set_xlabel(feature, fontsize=18)
            axis.set_title(f"{feature} distribution", fontsize=18)
            axis.set_ylim(0, 0.15)
            pdf.savefig(bbox_inches="tight")
            plt.close()


def plot_mu_scan_results_multimass(
    dfs,
    metadata,
    output_path,
):

    # -------------------- plotting settings --------------------
    colors = get_cmap("simple_contrast").colors
    styles = {
        "plot": {"marker": "o"},
        "legend": {"fontsize": 18},
        "ratio_frame": {"height_ratios": (2, 1), "hspace": 0.05},
    }
    styles_map = {
        "true": {
            "plot": {"color": "hdbs:spacecadet"},
            "fill_between": {"color": "none"},
        },
    }

    key_list = [key for key in dfs.keys() if key != "true"]

    for index, key in enumerate(key_list):
        styles_map[key] = {
            "plot": {"color": colors[index]},
            "fill_between": {
                "facecolor": get_rgba(colors[index], 0.2),
                "alpha": None,
                "edgecolor": get_rgba(colors[index], 0.9),
            },
        }

    config = {
        "error_on_top": False,
        "inherit_color": False,
        # 'draw_legend': False,
    }
    label_map = metadata["label_map"]

    # -------------------- making plots --------------------
    num_B = metadata["num_B"]
    use_full_stats = metadata["use_full_stats"]
    num_ensemble = metadata["num_ensemble"]

    xticks = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2]
    xticklabels = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "5"]
    xticks2 = mu2sig(np.array(xticks), B=num_B)
    xticklabels2 = [0.03, 0.07, 0.17, 0.35, 0.7, 1.74, 3.49, 17.43]
    xlabel = r"$\mu_{inj}\,(\%)$"
    plotter = General1DPlot(
        dfs, styles=styles, styles_map=styles_map, label_map=label_map, config=config
    )

    ax = plotter.draw(
        "x",
        "y",
        targets=dfs.keys(),
        yerrloattrib="yerrlo",
        yerrhiattrib="yerrhi",
        xlabel=xlabel,
        ylabel=r"$\hat{\mu}\,(\%)$",
        ymin=2e-5,
        ymax=0.1,
        logx=True,
        logy=True,
        offset_error=False,
        # legend_order=["true", "predicted"],
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    mu2sig_f = partial(mu2sig, B=num_B)
    sig2mu_f = partial(sig2mu, B=num_B)
    ax2 = ax.secondary_xaxis("top", functions=(mu2sig_f, sig2mu_f))
    ax2.tick_params(
        axis="x",
        which="major",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.tick_params(
        axis="x",
        which="minor",
        length=0,
        width=0,
        labeltop=True,
        labelbottom=False,
        top=True,
        bottom=False,
        direction="in",
        labelsize=18,
    )
    ax2.set_xticks(xticks2)
    ax2.set_xticklabels(xticklabels2)
    ax2.set_xlabel(r"$S/\sqrt{B}$", labelpad=10, fontsize=18)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_event_feature_distribution(dfs, misc, plot_options, output_path):

    mx = misc["mx"]
    my = misc["my"]
    numB = misc["numB"]
    use_full_stats = misc["use_full_stats"]
    use_perfect_modelB = misc["use_perfect_modelB"]
    use_modelB_genData = misc["use_modelB_genData"]
    columns = misc["columns"]

    # text = f"Signal at ({mx}, {my}) GeV \nmu_true = {mu_true}%, significance = {sig_significance}, mu_test = {mu_test}%"
    # if use_full_stats:
    #     text += "\n full stats"
    # else:
    #     text += "\n lumi matched"

    # if use_perfect_modelB:
    #     text += ", model B trained in SR"
    # if use_modelB_genData:
    #     text += ", use model B to generate bkgs in data"
    # if not use_modelB_genData and not use_perfect_modelB:
    #     text += ", model B trained in CR"

    plotter = VariableDistributionPlot(dfs, plot_options=plot_options)
    # plotter.add_text(text, 0.05, 0.95, fontsize=18)

    with PdfPages(output_path) as pdf:
        for feature in columns:
            plotter.draw(
                feature,
                logy=False,
                bins=np.linspace(
                    dfs["Background"][feature].min(),
                    dfs["Background"][feature].max(),
                    101,
                ),
                unit="GeV",
                show_error=False,
                comparison_options=None,
                xlabel=feature,
            )
            # axis.set_title(f"{feature} distribution", fontsize=18)
            # axis.set_ylim(0, 0.25)
            pdf.savefig(bbox_inches="tight")
            plt.close()
