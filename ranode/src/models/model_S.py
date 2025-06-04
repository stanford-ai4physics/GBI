import numpy as np
from nflows import flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from torch import nn
import nflows
import src.models.flow_models as fnn
from torch.utils.data import IterableDataset

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.permutations import RandomPermutation

# def flows_model_RQS(num_layers = 6, num_features=4, num_blocks = 2,
#                 hidden_features = 64, device = 'cpu',
#                 context_features = 1, random_mask = True,
#                 use_batch_norm = True, dropout_probability = 0.2):


def flows_model_RQS(
    num_layers=2,
    num_features=4,
    num_blocks=2,
    hidden_features=32,
    device="cpu",
    context_features=1,
    random_mask=True,
    use_batch_norm=True,
    dropout_probability=0.2,
):

    flow_params_rec_energy = {
        "num_blocks": num_blocks,  # num of layers per block
        "features": num_features,
        "context_features": context_features,
        "hidden_features": hidden_features,
        "use_residual_blocks": False,
        "use_batch_norm": use_batch_norm,
        "dropout_probability": dropout_probability,
        "activation": getattr(F, "leaky_relu"),
        "random_mask": random_mask,
        "num_bins": 8,
        "tails": "linear",
        "tail_bound": 8,
        "min_bin_width": 1e-6,
        "min_bin_height": 1e-6,
        "min_derivative": 1e-6,
    }
    rec_flow_blocks = []
    for _ in range(num_layers):
        rec_flow_blocks.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_rec_energy
            )
        )
        #   rec_flow_blocks.append(BatchNorm(num_features))
        rec_flow_blocks.append(RandomPermutation(num_features))
    rec_flow_transform = CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    # rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = StandardNormal(shape=[num_features])

    model_S = flows.Flow(
        transform=rec_flow_transform, distribution=rec_flow_base_distribution
    ).to(device)

    return model_S


def r_anode_mass_joint_untransformed(
    model_S, w, optimizer, data_loader, device="cpu", mode="train", scheduler=None
):

    n_nans = 0
    if mode == "train":
        model_S.train()

    else:
        model_S.eval()

    total_loss = 0

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0]
        model_B_log_prob = data[1].to(device).flatten()
        mass_density_bkg = data[2].to(device).flatten()

        if mode == "train":
            optimizer.zero_grad()

        model_S_log_prob = model_S.log_prob(data_SR[:, :-1])

        if batch_idx == 0:
            assert model_S_log_prob.shape == model_B_log_prob.shape
            print(f"value of w: {w}")

        if mode == "train":
            data_p = (
                w * torch.exp(model_S_log_prob)
                + (1 - w) * torch.exp(model_B_log_prob) * mass_density_bkg
            )
        else:
            with torch.no_grad():
                data_p = (
                    w * torch.exp(model_S_log_prob)
                    + (1 - w) * torch.exp(model_B_log_prob) * mass_density_bkg
                )

        data_loss = torch.log(data_p + 1e-32)

        #############################################
        ##############################################

        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]

        loss = -data_loss.mean()
        total_loss += loss.item()

        if mode == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_S.parameters(), 1)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    total_loss /= len(data_loader)

    if n_nans > 0:
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print(f"WARNING: {n_nans} nans in data_loss in mode {mode}")
        print("---------------------------------------------------")
        print("---------------------------------------------------")
        print(f"max model_S_log_prob: {torch.max(model_S_log_prob)}")
        print(f"min model_S_log_prob: {torch.min(model_S_log_prob)}")
        print(f"max model_B_log_prob: {torch.max(model_B_log_prob)}")
        print(f"min model_B_log_prob: {torch.min(model_B_log_prob)}")

    return total_loss


class modelSDataLoader(IterableDataset):
    def __init__(self, data, device, batch_size):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        traintensor_S, log_B_train_tensor, train_mass_prob_B = data
        self.traintensor_S = traintensor_S.to(device)
        self.log_B_train_tensor = log_B_train_tensor.to(device)
        self.train_mass_prob_B = train_mass_prob_B.to(device)

    def __iter__(self):
        num_samples = self.traintensor_S.size(0)
        indices = torch.randperm(num_samples, device=self.device)
        for i in range(0, num_samples - num_samples % self.batch_size, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            yield (
                self.traintensor_S[batch_idx],
                self.log_B_train_tensor[batch_idx],
                self.train_mass_prob_B[batch_idx],
            )

    def __len__(self):
        return self.traintensor_S.size(0) // self.batch_size
