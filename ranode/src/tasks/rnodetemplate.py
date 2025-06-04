import os, sys
import importlib
import luigi
import law
import numpy as np
import pandas as pd
import json

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    ProcessMixin,
    WScanMixin,
    BkgModelMixin,
)
from src.tasks.preprocessing import PreprocessingFold
from src.tasks.bkgtemplate import PredictBkgProb
from src.utils.utils import NumpyEncoder, str_encode_value
from src.tasks.bkgsampling import PredictBkgProbGen, PreprocessingFoldwModelBGen


class RNodeTemplate(
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    TemplateRandomMixin,
    BkgModelMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    device = luigi.Parameter(default="cuda:0")
    batchsize = luigi.IntParameter(default=2048)
    epoches = luigi.IntParameter(default=200)
    w_value = luigi.FloatParameter(default=0.05)
    early_stopping_patience = luigi.IntParameter(default=10)

    def store_parts(self):
        w_value = str_encode_value(self.w_value)
        return super().store_parts() + (f"w_{w_value}",)

    def requires(self):

        if self.use_bkg_model_gen_data:
            return {
                "preprocessed_data": PreprocessingFoldwModelBGen.req(self),
                "bkgprob": PredictBkgProbGen.req(self),
            }
        else:
            return {
                "preprocessed_data": PreprocessingFold.req(self),
                "bkgprob": PredictBkgProb.req(self),
            }

    def output(self):
        return {
            "sig_model": self.local_target(f"model_S.pt"),
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
            "metadata": self.local_target("metadata.json"),
        }

    @law.decorator.safe_output
    def run(self):

        input_dict = {
            "preprocessing": {
                "data_trainval_SR_model_S": self.input()["preprocessed_data"][
                    "SR_data_trainval_model_S"
                ],
                "data_trainval_SR_model_B": self.input()["preprocessed_data"][
                    "SR_data_trainval_model_B"
                ],
                "SR_mass_hist": self.input()["preprocessed_data"]["SR_mass_hist"],
            },
            "bkgprob": {
                "log_B_trainval": self.input()["bkgprob"]["log_B_trainval"],
            },
        }

        print(
            f"train model S with train random seed {self.train_random_seed}, sample fold {self.fold_split_seed}, s_ratio {self.s_ratio}"
        )
        from src.models.train_model_S import train_model_S

        train_model_S(
            input_dict,
            self.output(),
            self.s_ratio,
            self.w_value,
            self.batchsize,
            self.epoches,
            self.early_stopping_patience,
            self.train_random_seed,
            self.device,
        )


class ScanRANODEFixedSeed(
    TemplateRandomMixin,
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    WScanMixin,
    BkgModelMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    def requires(self):

        model_list = {}

        for index in range(self.scan_number):
            model_list[f"model_{index}"] = RNodeTemplate.req(
                self,
                w_value=self.w_range[index],
                train_random_seed=self.train_random_seed,
            )

        return model_list

    def output(self):
        return {
            "model_list": self.local_target("model_list.json"),
        }

    @law.decorator.safe_output
    def run(self):

        val_loss_scan = []
        model_path_list_scan = {}

        for index_w in range(self.scan_number):
            print(self.input()[f"model_{index_w}"]["metadata"].path)
            # save min val loss
            metadata_w = self.input()[f"model_{index_w}"]["metadata"].load()
            min_val_loss = metadata_w["min_val_loss_list"]
            val_events_num = metadata_w["num_val_events"]

            # save model paths
            model_path = [self.input()[f"model_{index_w}"]["sig_model"].path]

            val_loss_scan.append(min_val_loss)
            model_path_list_scan[f"scan_index_{index_w}"] = model_path

        val_loss_scan = np.array(val_loss_scan)
        val_loss_scan = -1 * val_loss_scan.flatten()

        print(self.w_range)
        print(val_loss_scan)

        self.output()["model_list"].parent.touch()
        with open(self.output()["model_list"].path, "w") as f:
            json.dump(model_path_list_scan, f, cls=NumpyEncoder)


class ScanRANODE(
    SigTemplateTrainingUncertaintyMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    BkgModelMixin,
    WScanMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    def requires(self):

        model_results = {}

        for index in range(self.train_num_sig_templates):
            model_results[f"model_seed_{index}"] = ScanRANODEFixedSeed.req(
                self, train_random_seed=index
            )

        if self.use_bkg_model_gen_data:
            return {
                "model_S_scan_result": model_results,
                "data": PreprocessingFoldwModelBGen.req(self),
                "bkgprob": PredictBkgProbGen.req(self),
            }
        else:
            return {
                "model_S_scan_result": model_results,
                "data": PreprocessingFold.req(self),
                "bkgprob": PredictBkgProb.req(self),
            }

    def output(self):
        return {
            "prob_S_scan": self.local_target("prob_S_scan.npy"),
            "prob_B_scan": self.local_target("prob_B_scan.npy"),
        }

    @law.decorator.safe_output
    def run(self):

        from src.models.ranode_pred import ranode_pred

        prob_S_list = []

        # for each w test value
        for w_index in range(self.scan_number):

            prob_S_list_w = []

            w_value = self.w_range[w_index]
            print(f" - evaluating scan index {w_index}, w value {w_value}")

            # for each random seed, load the model, evaluate the model on test data, and save the prob_S
            for index in range(self.train_num_sig_templates):

                # data path
                test_data_path = self.input()["data"]

                # prob B path
                bkg_prob_test_path = self.input()["bkgprob"]["log_B_test"]

                # model list
                model_list_path = self.input()["model_S_scan_result"][
                    f"model_seed_{index}"
                ]["model_list"].path

                print(model_list_path)
                with open(model_list_path, "r") as f:
                    model_scan_dict = json.load(f)

                model_list = model_scan_dict[f"scan_index_{w_index}"]

                prob_S, prob_B = ranode_pred(
                    model_list, test_data_path, bkg_prob_test_path
                )

                # prob_S shape is (num_models, num_samples), prob_B shape is (num_samples,)
                if len(prob_S_list_w) == 0:
                    prob_S_list_w = prob_S
                else:
                    prob_S_list_w = np.concatenate([prob_S_list_w, prob_S], axis=0)

            prob_S_list.append(prob_S_list_w)

        prob_S_list = np.array(prob_S_list)
        prob_B_list = prob_B

        self.output()["prob_S_scan"].parent.touch()
        np.save(self.output()["prob_S_scan"].path, prob_S_list)
        np.save(self.output()["prob_B_scan"].path, prob_B_list)
