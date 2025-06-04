import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    BkgModelMixin,
    WScanMixin,
)
from src.tasks.preprocessing import PreprocessingFold
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.rnodetemplate import (
    ScanRANODE,
    RNodeTemplate,
)


class FittingScanResults(
    ScanRANODE,
):

    def requires(self):
        return ScanRANODE.req(self)

    def output(self):
        return {
            "scan_plot": self.local_target(
                f"scan_plot_{str_encode_value(self.s_ratio)}.pdf"
            ),
            "peak_info": self.local_target("peak_info.json"),
        }

    @law.decorator.safe_output
    def run(self):

        # load scan results
        prob_S_scan = np.load(self.input()["prob_S_scan"].path)
        prob_B_scan = np.load(self.input()["prob_B_scan"].path)
        w_scan_range = self.w_range
        w_true = self.s_ratio

        from src.fitting.fitting import bootstrap_and_fit

        self.output()["scan_plot"].parent.touch()
        output_dir = self.output()
        bootstrap_and_fit(prob_S_scan, prob_B_scan, w_scan_range, w_true, output_dir)


class ScanOverTrueMu(
    BkgModelMixin,
    ProcessMixin,
    BaseTask,
):

    scan_index = luigi.ListParameter(
        default=[
            1,  # 0.01%
            5,  # 0.10%
            6,  # 0.17%
            7,  # 0.30%
            8,  # 0.53%
            9,  # 0.93%
            10,  # 1.63%
            11,  # 2.85%
            12,  # 5.01%
        ]
    )

    def requires(self):
        return [
            FittingScanResults.req(self, s_ratio_index=index)
            for index in self.scan_index
        ]

    def output(self):
        return {
            "plot": self.local_target("full_scan.pdf"),
            "plot_info": self.local_target("plot_info.json"),
        }

    @law.decorator.safe_output
    def run(self):

        if self.use_full_stats:
            num_B = 738020
        else:
            num_B = 121980

        mu_true_list = []
        mu_pred_list = []
        mu_lowerbound_list = []
        mu_upperbound_list = []

        for index in range(len(self.scan_index)):
            with open(self.input()[index]["peak_info"].path, "r") as f:
                peak_info = json.load(f)

            mu_true = peak_info["mu_true"]
            mu_pred = peak_info["mu_pred"]
            mu_lowerbound = peak_info["left_CI"]
            mu_upperbound = peak_info["right_CI"]

            mu_true_list.append(mu_true)
            mu_pred_list.append(mu_pred)
            mu_lowerbound_list.append(mu_lowerbound)
            mu_upperbound_list.append(mu_upperbound)

        dfs = {
            "true": pd.DataFrame(
                {
                    "x": np.array(mu_true_list),
                    "y": np.array(mu_true_list),
                }
            ),
            "predicted": pd.DataFrame(
                {
                    "x": np.array(mu_true_list),
                    "y": np.array(mu_pred_list),
                    "yerrlo": np.array(mu_lowerbound_list),
                    "yerrhi": np.array(mu_upperbound_list),
                }
            ),
        }

        misc = {
            "mx": self.mx,
            "my": self.my,
            "use_full_stats": self.use_full_stats,
            "use_perfect_modelB": self.use_perfect_bkg_model,
            "use_modelB_genData": self.use_bkg_model_gen_data,
            "num_B": num_B,
        }

        self.output()["plot"].parent.touch()
        output_path = self.output()["plot"].path

        from src.plotting.plotting import plot_mu_scan_results

        plot_mu_scan_results(
            dfs,
            misc,
            output_path,
        )

        # save plot info
        plot_info = {
            "true": {
                "x": mu_true_list,
                "y": mu_true_list,
            },
            "predicted": {
                "x": mu_true_list,
                "y": mu_pred_list,
                "yerrlo": mu_lowerbound_list,
                "yerrhi": mu_upperbound_list,
            },
            "misc": misc,
        }
        with open(self.output()["plot_info"].path, "w") as f:
            json.dump(plot_info, f, cls=NumpyEncoder)


class ScanOverTrueMuEnsembleAvg(
    BkgModelMixin,
    BaseTask,
):

    mx = luigi.IntParameter(default=100)
    my = luigi.IntParameter(default=500)

    num_ensemble = luigi.IntParameter(default=10)

    def store_parts(self):
        return super().store_parts() + (
            f"mx_{self.mx}",
            f"my_{self.my}",
            f"num_ensemble_{self.num_ensemble}",
        )

    def requires(self):
        return [
            ScanOverTrueMu.req(self, ensemble=index)
            for index in range(1, self.num_ensemble + 1)
        ]

    def output(self):
        return {
            "plot": self.local_target("ensemble_avg_scan_plot.pdf"),
            "plot_info": self.local_target("ensemble_avg_plot_info.json"),
        }

    @law.decorator.safe_output
    def run(self):

        pred_y = []
        pred_yerrlo = []
        pred_yerrhi = []

        for ensemble_index in range(self.num_ensemble):
            with open(self.input()[ensemble_index]["plot_info"].path, "r") as f:
                plot_info = json.load(f)

            if ensemble_index == 0:
                true_x = plot_info["true"]["x"]
                true_y = plot_info["true"]["y"]
                misc = plot_info["misc"]
                pred_x = plot_info["predicted"]["x"]

            pred_y.append(plot_info["predicted"]["y"])
            pred_yerrlo.append(plot_info["predicted"]["yerrlo"])
            pred_yerrhi.append(plot_info["predicted"]["yerrhi"])

        # average the predicted values in log 10 base which matches the fitting
        pred_y = np.log10(np.array(pred_y))
        pred_yerrlo = np.log10(np.array(pred_yerrlo))
        mask_nan_inlo = np.any(np.isneginf(pred_yerrlo), axis=0)
        pred_yerrhi = np.log10(np.array(pred_yerrhi))

        ensemble_averaged_pred_y_log = np.mean(pred_y, axis=0)
        ensemble_averaged_pred_yerrlo_log = np.mean(pred_yerrlo, axis=0)
        ensemble_averaged_pred_yerrhi_log = np.mean(pred_yerrhi, axis=0)

        ensemble_averaged_pred_y = np.power(10, ensemble_averaged_pred_y_log)
        ensemble_averaged_pred_yerrlo = np.power(10, ensemble_averaged_pred_yerrlo_log)
        ensemble_averaged_pred_yerrlo[mask_nan_inlo] = 0
        ensemble_averaged_pred_yerrhi = np.power(10, ensemble_averaged_pred_yerrhi_log)

        dfs = {
            "true": pd.DataFrame(
                {
                    "x": np.array(true_x),
                    "y": np.array(true_y),
                }
            ),
            "predicted": pd.DataFrame(
                {
                    "x": np.array(pred_x),
                    "y": np.array(ensemble_averaged_pred_y),
                    "yerrlo": np.array(ensemble_averaged_pred_yerrlo),
                    "yerrhi": np.array(ensemble_averaged_pred_yerrhi),
                }
            ),
        }

        self.output()["plot"].parent.touch()
        output_path = self.output()["plot"].path

        from src.plotting.plotting import plot_mu_scan_results

        plot_mu_scan_results(
            dfs,
            misc,
            output_path,
        )

        # save plot info
        plot_info = {
            "true": {
                "x": true_x,
                "y": true_y,
            },
            "predicted": {
                "x": pred_x,
                "y": ensemble_averaged_pred_y,
                "yerrlo": ensemble_averaged_pred_yerrlo,
                "yerrhi": ensemble_averaged_pred_yerrhi,
            },
            "misc": misc,
        }
        with open(self.output()["plot_info"].path, "w") as f:
            json.dump(plot_info, f, cls=NumpyEncoder)


class ScanMultiModelsOverTrueMuEnsembleAvg(ScanOverTrueMuEnsembleAvg):

    def requires(self):
        return {
            "modelB_inSR": ScanOverTrueMuEnsembleAvg.req(
                self, use_perfect_bkg_model=True, use_bkg_model_gen_data=False
            ),
            "modelB_genData": ScanOverTrueMuEnsembleAvg.req(
                self, use_perfect_bkg_model=False, use_bkg_model_gen_data=True
            ),
            "modelB_inSB": ScanOverTrueMuEnsembleAvg.req(
                self, use_perfect_bkg_model=False, use_bkg_model_gen_data=False
            ),
        }

    def output(self):
        if self.use_full_stats:
            return self.local_target(f"fullstats_{self.mx}_{self.my}_scan.pdf")
        else:
            return self.local_target(f"lumi_matched_{self.mx}_{self.my}_scan.pdf")

    @law.decorator.safe_output
    def run(self):

        dfs = {}

        for model in ["modelB_inSB", "modelB_inSR", "modelB_genData"]:

            with open(self.input()[model]["plot_info"].path, "r") as f:
                avg_scan_info = json.load(f)

            if "true" not in dfs:
                dfs["true"] = pd.DataFrame(
                    {
                        "x": np.array(avg_scan_info["true"]["x"]),
                        "y": np.array(avg_scan_info["true"]["y"]),
                    }
                )

            dfs[model] = pd.DataFrame(
                {
                    "x": np.array(avg_scan_info["predicted"]["x"]),
                    "y": np.array(avg_scan_info["predicted"]["y"]),
                    "yerrlo": np.array(avg_scan_info["predicted"]["yerrlo"]),
                    "yerrhi": np.array(avg_scan_info["predicted"]["yerrhi"]),
                }
            )

            misc = avg_scan_info["misc"]

        misc["num_ensemble"] = self.num_ensemble

        label_map = {
            "true": "Truth",
            "modelB_inSB": "model B trained in SB",
            "modelB_inSR": "model B trained in SR",
            "modelB_genData": "model B generates data",
        }
        misc["label_map"] = label_map

        self.output().parent.touch()
        output_path = self.output().path

        from src.plotting.plotting import plot_mu_scan_results_multimodels

        plot_mu_scan_results_multimodels(
            dfs,
            misc,
            output_path,
        )


class ScanMultiMassOverTrueMuEnsembleAvg(
    BkgModelMixin,
    BaseTask,
):
    num_ensemble = luigi.IntParameter(default=10)

    def requires(self):
        return {
            "100, 500": ScanOverTrueMuEnsembleAvg.req(
                self,
                use_perfect_bkg_model=False,
                use_bkg_model_gen_data=False,
                mx=100,
                my=500,
            ),
            "300, 300": ScanOverTrueMuEnsembleAvg.req(
                self,
                use_perfect_bkg_model=False,
                use_bkg_model_gen_data=False,
                mx=300,
                my=300,
            ),
        }

    def output(self):
        if self.use_full_stats:
            return self.local_target(f"fullstats_scan_multimass.pdf")
        else:
            return self.local_target(f"lumi_matched_scan_multimass.pdf")

    @law.decorator.safe_output
    def run(self):

        dfs = {}

        for model in ["100, 500", "300, 300"]:

            with open(self.input()[model]["plot_info"].path, "r") as f:
                avg_scan_info = json.load(f)

            if "true" not in dfs:
                dfs["true"] = pd.DataFrame(
                    {
                        "x": np.array(avg_scan_info["true"]["x"]),
                        "y": np.array(avg_scan_info["true"]["y"]),
                    }
                )

            dfs[model] = pd.DataFrame(
                {
                    "x": np.array(avg_scan_info["predicted"]["x"]),
                    "y": np.array(avg_scan_info["predicted"]["y"]),
                    "yerrlo": np.array(avg_scan_info["predicted"]["yerrlo"]),
                    "yerrhi": np.array(avg_scan_info["predicted"]["yerrhi"]),
                }
            )

            misc = avg_scan_info["misc"]

        misc["num_ensemble"] = self.num_ensemble

        label_map = {
            "true": "Truth",
            "100, 500": "mx=100 GeV, my=500 GeV",
            "300, 300": "mx=300 GeV, my=300 GeV",
        }
        misc["label_map"] = label_map

        self.output().parent.touch()
        output_path = self.output().path

        from src.plotting.plotting import plot_mu_scan_results_multimass

        plot_mu_scan_results_multimass(
            dfs,
            misc,
            output_path,
        )
