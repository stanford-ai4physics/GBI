import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
from src.utils.utils import NumpyEncoder

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    ProcessMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
)


class ProcessSignal(SignalStrengthMixin, ProcessMixin, BaseTask):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=1)
    """

    def output(self):
        return {
            "signals": self.local_target("reprocessed_signals.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        # GRAVITATIONAL WAVE MODE: Generate synthetic GW signals instead of loading particle physics data
        # This adapts R-ANODE methodology to gravitational wave anomaly detection
        from src.data_prep.gw_processing import process_gw_signals

        print(f"[GW MODE] Generating gravitational wave signals with amplitude={self.s_ratio*100}")

        # Generate GW signal data dynamically (no external files needed!)
        # The amplitude is scaled by s_ratio to simulate different signal strengths
        sig_data = process_gw_signals(amplitude=self.s_ratio * 100.0)

        self.output()["signals"].parent.touch()
        np.save(self.output()["signals"].path, sig_data)

        print(f"[GW MODE] Generated {len(sig_data)} signal events")


class ProcessBkg(BaseTask):
    """
    Will reprocess the background such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=0)

    Bkg events will first be splitted into SR and CR
    The overall CR will be used to calculate the normalizing parameters, then it will be applied on CR events
    Then CR events will be splitted into train and val set
    """

    def output(self):
        return {
            "SR_bkg": self.local_target("reprocessed_bkgs_sr.npy"),
            "CR_train": self.local_target("reprocessed_bkgs_cr_train.npy"),
            "CR_val": self.local_target("reprocessed_bkgs_cr_val.npy"),
            "pre_parameters": self.local_target("pre_parameters.json"),
        }

    @law.decorator.safe_output
    def run(self):
        # GRAVITATIONAL WAVE MODE: Generate synthetic GW backgrounds instead of loading QCD data
        from src.data_prep.gw_processing import process_gw_backgrounds, gw_background_split

        print("[GW MODE] Generating gravitational wave background events")

        # Generate GW background data dynamically (no external files needed!)
        gw_bkg_data = process_gw_backgrounds()

        print(f"[GW MODE] Generated {len(gw_bkg_data)} background events")

        # Split into Signal Region (SR) and Control Region (CR) using GW-adapted splitting
        SR_bkg, CR_bkg = gw_background_split(
            gw_bkg_data,
            resample_seed=42,
        )

        # save SR data
        self.output()["SR_bkg"].parent.touch()
        np.save(self.output()["SR_bkg"].path, SR_bkg)

        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # ----------------------- calculate normalizing parameters -----------------------
        pre_parameters = preprocess_params_fit(CR_bkg)
        # save pre_parameters
        self.output()["pre_parameters"].parent.touch()
        with open(self.output()["pre_parameters"].path, "w") as f:
            json.dump(pre_parameters, f, cls=NumpyEncoder)

        # ----------------------- process data in CR -----------------------
        CR_bkg = preprocess_params_transform(CR_bkg, pre_parameters)
        CR_bkg_train, CR_bkg_val = train_test_split(
            CR_bkg, test_size=0.25, random_state=42
        )

        # save training and validation data in CR
        np.save(self.output()["CR_train"].path, CR_bkg_train)
        np.save(self.output()["CR_val"].path, CR_bkg_val)


class PreprocessingFold(
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):
    """
    This task will take signal train val test set with a given signal strength and fold split index
    It also takes SR bkg trainval set, using the same train val split index as seed to split the SR bkg
    into train and val set

    Then it will mix the signal and bkg into SR data, and normalize the data using the normalizing parameters
    calculated from CR data
    """

    def requires(self):

        if self.s_ratio != 0:
            return {
                "signal": ProcessSignal.req(self),
                "bkg": ProcessBkg.req(self),
            }
        else:
            return {
                "bkg": ProcessBkg.req(self),
            }

    def output(self):
        return {
            "SR_data_trainval_model_S": self.local_target(
                "data_SR_data_trainval_model_S.npy"
            ),
            "SR_data_test_model_S": self.local_target("data_SR_data_test_model_S.npy"),
            "SR_data_trainval_model_B": self.local_target(
                "data_SR_data_trainval_model_B.npy"
            ),
            "SR_data_test_model_B": self.local_target("data_SR_data_test_model_B.npy"),
            "SR_mass_hist": self.local_target("SR_mass_hist.json"),
        }

    @law.decorator.safe_output
    def run(self):

        from src.data_prep.utils import (
            logit_transform,
            preprocess_params_transform,
            preprocess_params_fit,
        )

        # load data
        if self.s_ratio != 0:
            SR_signal = np.load(self.input()["signal"]["signals"].path)
        SR_bkg = np.load(self.input()["bkg"]["SR_bkg"].path)

        pre_parameters = json.load(
            open(self.input()["bkg"]["pre_parameters"].path, "r")
        )
        for key in pre_parameters.keys():
            pre_parameters[key] = np.array(pre_parameters[key])

        # ----------------------- mass hist in SR -----------------------
        from config.configs import SR_MIN, SR_MAX

        mass = SR_bkg[SR_bkg[:, -1] == 0, 0]
        bins = np.linspace(SR_MIN, SR_MAX, 50)
        hist_back = np.histogram(mass, bins=bins, density=True)
        # save mass histogram and bins
        self.output()["SR_mass_hist"].parent.touch()
        with open(self.output()["SR_mass_hist"].path, "w") as f:
            json.dump({"hist": hist_back[0], "bins": hist_back[1]}, f, cls=NumpyEncoder)

        # ----------------------- make SR data -----------------------

        # concatenate signal and bkg
        if self.s_ratio != 0:
            SR_data = np.concatenate([SR_signal, SR_bkg], axis=0)
        else:
            SR_data = SR_bkg

        # process data
        _, mask = logit_transform(
            SR_data[:, 1:-1], pre_parameters["min"], pre_parameters["max"]
        )
        SR_data = SR_data[mask]
        SR_data = preprocess_params_transform(SR_data, pre_parameters)

        # split into trainval and test set
        from src.data_prep.utils import fold_splitting

        SR_data_trainval, SR_data_test = fold_splitting(
            SR_data,
            n_folds=self.fold_split_num,
            random_seed=self.ensemble,
            test_fold=self.fold_split_seed,
        )

        # For signal model, we shift the mass by -3.5 following RANODE workflow
        # copy one set for signal model
        SR_data_trainval_model_S = SR_data_trainval.copy()
        SR_data_test_model_S = SR_data_test.copy()
        # shift mass by -3.5 for signals
        SR_data_trainval_model_S[:, 0] -= 3.5
        SR_data_test_model_S[:, 0] -= 3.5

        np.save(
            self.output()["SR_data_trainval_model_S"].path, SR_data_trainval_model_S
        )
        np.save(self.output()["SR_data_test_model_S"].path, SR_data_test_model_S)

        # copy another set for background model
        SR_data_trainval_model_B = SR_data_trainval.copy()
        SR_data_test_model_B = SR_data_test.copy()

        np.save(
            self.output()["SR_data_trainval_model_B"].path, SR_data_trainval_model_B
        )
        np.save(self.output()["SR_data_test_model_B"].path, SR_data_test_model_B)

        # print out some info
        trainval_sig_num = SR_data_trainval_model_B[:, -1].sum()
        trainval_bkg_num = (SR_data_trainval_model_B[:, -1] == 0).sum()
        train_mu = trainval_sig_num / (trainval_sig_num + trainval_bkg_num)
        test_sig_num = SR_data_test_model_B[:, -1].sum()
        test_bkg_num = (SR_data_test_model_B[:, -1] == 0).sum()
        test_mu = test_sig_num / (test_sig_num + test_bkg_num)
        print("Fold splitting index is: ", self.fold_split_seed)
        print("true mu: ", self.s_ratio)
        print(
            "trainval mu: ",
            train_mu,
            "trainval sig num: ",
            trainval_sig_num,
            "trainval bkg num: ",
            trainval_bkg_num,
        )
        print(
            "test mu: ",
            test_mu,
            "test sig num: ",
            test_sig_num,
            "test bkg num: ",
            test_bkg_num,
        )


class PlotMjjDistribution(
    ProcessMixin,
    BaseTask,
):

    def requires(self):
        return {
            "bkg": ProcessBkg.req(self),
        }

    def output(self):
        return self.local_target("mjj_distribution.pdf")

    @law.decorator.safe_output
    def run(self):

        # load bkg
        SR_bkg = np.load(self.input()["bkg"]["SR_bkg"].path)
        SR_bkg_mjj = SR_bkg[:, 0]
        SB_bkg_train = np.load(self.input()["bkg"]["CR_train"].path)
        SB_bkg_mjj_train = SB_bkg_train[:, 0]
        SB_bkg_val = np.load(self.input()["bkg"]["CR_val"].path)
        SB_bkg_mjj_val = SB_bkg_val[:, 0]
        SB_bkg_mjj = np.concatenate([SB_bkg_mjj_train, SB_bkg_mjj_val], axis=0)
        bkg_mjj = np.concatenate([SR_bkg_mjj, SB_bkg_mjj], axis=0)

        # process signal
        data_dir = os.environ.get("DATA_DIR")

        data_path = f"{data_dir}/extra_raw_lhco_samples/events_anomalydetection_Z_XY_qq_parametric.h5"

        from src.data_prep.signal_processing import process_raw_signals

        signal = process_raw_signals(
            data_path, output_path=None, mx=self.mx, my=self.my
        )
        signal_mjj = signal[:, 0]

        # make plot
        from quickstats.plots import VariableDistributionPlot
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        dfs = {
            "background": pd.DataFrame({"mjj": bkg_mjj}),
            "signal": pd.DataFrame({"mjj": signal_mjj}),
        }

        plot_options = {
            "background": {
                "styles": {
                    "color": "black",
                    "histtype": "step",
                    "lw": 2,
                }
            },
            "signal": {
                "styles": {
                    "color": "red",
                    "histtype": "stepfilled",
                    "lw": 2,
                }
            },
        }

        plotter = VariableDistributionPlot(dfs, plot_options=plot_options)

        self.output().parent.touch()
        output_path = self.output().path
        with PdfPages(output_path) as pdf:

            axis = plotter.draw(
                "mjj",
                logy=True,
                normalize=False,
                bins=np.linspace(
                    2,
                    8,
                    151,
                ),
                unit="TeV",
                show_error=False,
                comparison_options=None,
                xlabel="mjj",
            )

            plt.axvline(
                x=3.3,
                color="black",
                linestyle="--",
                label="Signal Region Boundary (3.3 - 3.7 TeV)",
            )

            plt.axvline(x=3.7, color="black", linestyle="--")

            plt.xlim(2, 8)
            pdf.savefig(bbox_inches="tight")
            plt.close()
