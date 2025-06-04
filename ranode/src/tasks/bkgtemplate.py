import os, sys
import importlib
import luigi
import law
import json
import copy
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
from sklearn.model_selection import train_test_split
import pickle
import torch

from src.utils.law import (
    BaseTask,
    SignalStrengthMixin,
    TemplateRandomMixin,
    BkgTemplateUncertaintyMixin,
    BkgModelMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    ProcessMixin,
)
from src.tasks.preprocessing import PreprocessingFold, ProcessBkg


class BkgTemplateTraining(TemplateRandomMixin, BaseTask):
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=2048)
    epochs = luigi.IntParameter(default=200)
    early_stopping_patience = luigi.IntParameter(default=20)

    def requires(self):
        return ProcessBkg.req(self)

    def output(self):
        return {
            "bkg_model": self.local_target("model_B_CR.pt"),
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
        }

    @law.decorator.safe_output
    def run(self):

        # freeze the random seed of torch
        torch.manual_seed(self.train_random_seed)

        # need:
        # "data_train_CR": self.local_target("data_train_cr.npy"),
        # "data_val_CR": self.local_target("data_val_cr.npy"),

        # load data
        data_train_CR = np.load(self.input()["CR_train"].path)
        data_val_CR = np.load(self.input()["CR_val"].path)

        traintensor = torch.from_numpy(data_train_CR.astype("float32")).to(self.device)
        valtensor = torch.from_numpy(data_val_CR.astype("float32")).to(self.device)

        train_tensor = torch.utils.data.TensorDataset(traintensor)
        val_tensor = torch.utils.data.TensorDataset(valtensor)

        trainloader = torch.utils.data.DataLoader(
            train_tensor, batch_size=self.batchsize, shuffle=True
        )
        valloader = torch.utils.data.DataLoader(
            val_tensor, batch_size=self.batchsize * 5, shuffle=False
        )

        # define model
        from src.models.model_B import DensityEstimator, anode

        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        model_B = DensityEstimator(config_file, eval_mode=False, device=self.device)

        trainloss_list = []
        valloss_list = []
        min_valloss = np.inf
        patience = 0

        for epoch in range(self.epochs):

            trainloss = anode(
                model_B.model,
                trainloader,
                model_B.optimizer,
                params=None,
                device=self.device,
                mode="train",
            )
            valloss = anode(
                model_B.model,
                valloader,
                model_B.optimizer,
                params=None,
                device=self.device,
                mode="val",
            )

            # torch.save(model_B.model.state_dict(), scrath_path+'/model_B/model_CR_'+str(epoch)+'.pt')
            state_dict = copy.deepcopy(
                {k: v.cpu() for k, v in model_B.model.state_dict().items()}
            )

            valloss_list.append(valloss)
            trainloss_list.append(trainloss)

            # early stopping
            if valloss < min_valloss:
                min_valloss = valloss
                patience = 0
                best_model = state_dict
            else:
                patience += 1
                if patience > self.early_stopping_patience:
                    print("early stopping at epoch: ", epoch)
                    break

            print("epoch: ", epoch, "trainloss: ", trainloss, "valloss: ", valloss)

        # save trainings and validation losses
        trainloss_list = np.array(trainloss_list)
        valloss_list = np.array(valloss_list)
        self.output()["trainloss_list"].parent.touch()
        np.save(self.output()["trainloss_list"].path, trainloss_list)
        np.save(self.output()["valloss_list"].path, valloss_list)

        # save best models
        torch.save(best_model, self.output()["bkg_model"].path)


class BkgTemplateChecking(
    BkgTemplateUncertaintyMixin,
    BaseTask,
):

    device = luigi.Parameter(default="cuda")
    num_CR_samples = luigi.IntParameter(default=100000)

    def requires(self):
        return {
            "bkg_models": [
                BkgTemplateTraining.req(self, train_random_seed=i)
                for i in range(self.num_bkg_templates)
            ],
            "preprocessed_data": ProcessBkg.req(self),
        }

    def output(self):
        return {
            "CR_comparison_plot": self.local_target("CR_comparison_plots.pdf"),
        }

    @law.decorator.safe_output
    def run(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        self.output()["CR_comparison_plot"].parent.touch()
        # -------------------------------- CR comparison plots --------------------------------
        # load the sample to compare with
        data_val_CR = np.load(self.input()["preprocessed_data"]["CR_val"].path)
        # generate CR events using the model with condition from data_train_CR
        mass_cond_CR = (
            torch.from_numpy(data_val_CR[:, 0])
            .reshape((-1, 1))
            .type(torch.FloatTensor)
            .to(self.device)
        )
        mass_cond_CR = mass_cond_CR[: self.num_CR_samples]

        sampled_CR_events = []

        # ----------------------------------- load all models and make prediction --------------------------------
        from src.models.model_B import DensityEstimator

        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        for seed_i in range(self.num_bkg_templates):
            # load the models
            model_B_seed_i = DensityEstimator(
                config_file, eval_mode=True, device=self.device
            )
            best_model_dir_seed_i = self.input()["bkg_models"][seed_i]["bkg_model"].path
            model_B_seed_i.model.load_state_dict(torch.load(best_model_dir_seed_i))
            model_B_seed_i.model.to(self.device)
            model_B_seed_i.model.eval()

            with torch.no_grad():
                sampled_CR_events_seed_i = model_B_seed_i.model.sample(
                    num_samples=len(mass_cond_CR), cond_inputs=mass_cond_CR
                )

            sampled_CR_events.extend(
                sampled_CR_events_seed_i.cpu().numpy().astype("float32")
            )

        sampled_CR_events = np.array(sampled_CR_events)
        sampled_CR_events_weight = (
            np.ones(len(sampled_CR_events))
            / len(sampled_CR_events)
            * self.num_CR_samples
        )

        # plot the comparison
        with PdfPages(self.output()["CR_comparison_plot"].path) as pdf:
            # first plot the mass distribution
            mass_cond_CR = mass_cond_CR.cpu().numpy().reshape(-1).astype("float32")
            f = plt.figure()
            plt.hist(mass_cond_CR, bins=100, histtype="step", label="mass condition CR")
            plt.xlabel("mass")
            plt.ylabel("counts")
            plt.legend()
            pdf.savefig(f)
            plt.close(f)

            # then plot the rest of the variables
            for i in range(len(sampled_CR_events[0])):
                bins = np.linspace(
                    data_val_CR[:, i + 1].min(), data_val_CR[:, i + 1].max(), 100
                )
                f = plt.figure()
                plt.hist(
                    data_val_CR[: self.num_CR_samples, i + 1],
                    bins=bins,
                    histtype="step",
                    label="data_train_CR",
                )
                plt.hist(
                    sampled_CR_events[:, i],
                    weights=sampled_CR_events_weight,
                    bins=bins,
                    histtype="step",
                    label="sampled CR events",
                )
                plt.xlabel(f"var {i}")
                plt.ylabel("counts")
                plt.legend()
                pdf.savefig(f)
                plt.close(f)


# --------------------------------- Ideal Bkg Model ---------------------------------
class PerfectBkgTemplateTraining(TemplateRandomMixin, BaseTask):
    device = luigi.Parameter(default="cuda")
    batchsize = luigi.IntParameter(default=2048)
    epochs = luigi.IntParameter(default=200)
    early_stopping_patience = luigi.IntParameter(default=20)

    def requires(self):
        return PreprocessingFold.req(self, s_ratio_index=0, use_full_stats=True)

    def output(self):
        return {
            "bkg_model": self.local_target("model_B_SR_perfect.pt"),
            "trainloss_list": self.local_target("trainloss_list.npy"),
            "valloss_list": self.local_target("valloss_list.npy"),
        }

    @law.decorator.safe_output
    def run(self):

        # freeze the random seed of torch
        torch.manual_seed(self.train_random_seed)

        # need:
        # "data_train_CR": self.local_target("data_train_cr.npy"),
        # "data_val_CR": self.local_target("data_val_cr.npy"),

        # load bkg in SR but no label
        data_trainval_SR = np.load(self.input()["SR_data_trainval_model_B"].path)
        data_test_SR = np.load(self.input()["SR_data_test_model_B"].path)
        assert (
            data_trainval_SR[:, -1].sum() == 0
        ), "data_trainval_SR should have no signal"
        assert data_test_SR[:, -1].sum() == 0, "data_test_SR should have no signal"

        # combine all data and resplit into train and val
        data_all = np.concatenate([data_trainval_SR, data_test_SR], axis=0)
        data_train_SR, data_val_SR = train_test_split(
            data_all, test_size=0.2, random_state=self.train_random_seed
        )

        print("train with ", len(data_train_SR), " samples")
        print("val with ", len(data_val_SR), " samples")

        traintensor = torch.from_numpy(data_train_SR.astype("float32")).to(self.device)
        valtensor = torch.from_numpy(data_val_SR.astype("float32")).to(self.device)

        train_tensor = torch.utils.data.TensorDataset(traintensor)
        val_tensor = torch.utils.data.TensorDataset(valtensor)

        trainloader = torch.utils.data.DataLoader(
            train_tensor, batch_size=self.batchsize, shuffle=True
        )
        valloader = torch.utils.data.DataLoader(
            val_tensor, batch_size=self.batchsize * 5, shuffle=False
        )

        # define model
        from src.models.model_B import DensityEstimator, anode

        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        model_B = DensityEstimator(config_file, eval_mode=False, device=self.device)

        trainloss_list = []
        valloss_list = []
        min_valloss = np.inf
        patience = 0

        for epoch in range(self.epochs):

            trainloss = anode(
                model_B.model,
                trainloader,
                model_B.optimizer,
                params=None,
                device=self.device,
                mode="train",
            )
            valloss = anode(
                model_B.model,
                valloader,
                model_B.optimizer,
                params=None,
                device=self.device,
                mode="val",
            )

            # torch.save(model_B.model.state_dict(), scrath_path+'/model_B/model_CR_'+str(epoch)+'.pt')
            state_dict = copy.deepcopy(
                {k: v.cpu() for k, v in model_B.model.state_dict().items()}
            )

            valloss_list.append(valloss)
            trainloss_list.append(trainloss)

            # early stopping
            if valloss < min_valloss:
                min_valloss = valloss
                patience = 0
                best_model = state_dict
            else:
                patience += 1
                if patience > self.early_stopping_patience:
                    print("early stopping at epoch: ", epoch)
                    break

            print("epoch: ", epoch, "trainloss: ", trainloss, "valloss: ", valloss)

        # save trainings and validation losses
        trainloss_list = np.array(trainloss_list)
        valloss_list = np.array(valloss_list)
        self.output()["trainloss_list"].parent.touch()
        np.save(self.output()["trainloss_list"].path, trainloss_list)
        np.save(self.output()["valloss_list"].path, valloss_list)

        # save best models
        torch.save(best_model, self.output()["bkg_model"].path)


class PredictBkgProb(
    BkgTemplateUncertaintyMixin,
    BkgModelMixin,
    FoldSplitRandomMixin,
    FoldSplitUncertaintyMixin,
    SignalStrengthMixin,
    ProcessMixin,
    BaseTask,
):

    device = luigi.Parameter(default="cuda")

    def requires(self):

        if self.use_perfect_bkg_model:
            return {
                "bkg_models": [
                    PerfectBkgTemplateTraining.req(self, train_random_seed=i)
                    for i in range(self.num_bkg_templates)
                ],
                "preprocessed_data": PreprocessingFold.req(self),
            }
        else:
            return {
                "bkg_models": [
                    BkgTemplateTraining.req(self, train_random_seed=i)
                    for i in range(self.num_bkg_templates)
                ],
                "preprocessed_data": PreprocessingFold.req(self),
            }

    def output(self):
        return {
            "log_B_trainval": self.local_target("log_B_trainval.npy"),
            "log_B_test": self.local_target("log_B_test.npy"),
        }

    @law.decorator.safe_output
    def run(self):
        # load the models
        from src.models.model_B import DensityEstimator, anode

        config_file = os.path.join("src", "models", "DE_MAF_model.yml")

        model_Bs = []

        for i in range(self.num_bkg_templates):
            model_B = DensityEstimator(config_file, eval_mode=True, device="cuda")
            best_model_dir = self.input()["bkg_models"][i]["bkg_model"].path
            model_B.model.load_state_dict(torch.load(best_model_dir))
            model_B.model.to("cuda")
            model_B.model.eval()
            model_Bs.append(model_B)

        # load the sample to compare with
        data_trainval_SR_B = np.load(
            self.input()["preprocessed_data"]["SR_data_trainval_model_B"].path
        )
        trainvaltensor_SR_B = torch.from_numpy(data_trainval_SR_B.astype("float32")).to(
            self.device
        )

        data_test_SR_B = np.load(
            self.input()["preprocessed_data"]["SR_data_test_model_B"].path
        )
        testtensor_SR_B = torch.from_numpy(data_test_SR_B.astype("float32")).to(
            self.device
        )

        # get avg probility of 10 models
        log_B_trainval_list = []
        log_B_test_list = []
        for model_B in model_Bs:
            with torch.no_grad():
                log_B_trainval = model_B.model.log_probs(
                    inputs=trainvaltensor_SR_B[:, 1:-1],
                    cond_inputs=trainvaltensor_SR_B[:, 0].reshape(-1, 1),
                )
                # set all nans to 0
                log_B_trainval[torch.isnan(log_B_trainval)] = 0
                log_B_trainval_list.append(log_B_trainval.cpu().numpy())

                log_B_test = model_B.model.log_probs(
                    inputs=testtensor_SR_B[:, 1:-1],
                    cond_inputs=testtensor_SR_B[:, 0].reshape(-1, 1),
                )
                # set all nans to 0
                log_B_test[torch.isnan(log_B_test)] = 0
                log_B_test_list.append
                log_B_test_list.append(log_B_test.cpu().numpy())

        log_B_trainval = np.array(log_B_trainval_list)
        B_trainval = np.exp(log_B_trainval).mean(axis=0)
        log_B_trainval = np.log(B_trainval + 1e-32)

        log_B_test = np.array(log_B_test_list)
        B_test = np.exp(log_B_test).mean(axis=0)
        log_B_test = np.log(B_test + 1e-32)

        self.output()["log_B_trainval"].parent.touch()
        np.save(self.output()["log_B_trainval"].path, log_B_trainval)
        np.save(self.output()["log_B_test"].path, log_B_test)
