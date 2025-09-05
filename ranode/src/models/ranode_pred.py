import os, sys
import importlib
import luigi
import copy
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
import torch
from tqdm import tqdm
import json
from src.utils.utils import NumpyEncoder, str_encode_value
from src.models.train_model_S import pred_model_S


def ranode_pred(model_S_list, test_data_dict, bkg_prob_dir, device="cuda"):

    # load data
    print("loading data")
    # load data
    data_test_SR_S = np.load(test_data_dict["SR_data_test_model_S"].path)
    data_test_SR_B = np.load(test_data_dict["SR_data_test_model_B"].path)
    data_tesr_SR_B_logprob = np.load(bkg_prob_dir.path).flatten()

    print("num sig in file: ", (data_test_SR_B[:, -1] == 1).sum())
    print("truth mu: ", (data_test_SR_B[:, -1] == 1).sum() / len(data_test_SR_B))

    # p(m) for bkg model p(x|m)
    with open(test_data_dict["SR_mass_hist"].path, "r") as f:
        mass_hist = json.load(f)
    SR_mass_hist = np.array(mass_hist["hist"])
    SR_mass_bins = np.array(mass_hist["bins"])
    density_back = rv_histogram((SR_mass_hist, SR_mass_bins))

    # p(m) for bkg model p(x|m)
    test_mass_prob_B = density_back.pdf(data_test_SR_B[:, 0])

    prob_S_list = []
    # make prediction using all models S in the list
    for model_S_i in model_S_list:
        print(model_S_i)
        prob_S_i = pred_model_S(model_S_i, data_test_SR_S, device=device)
        prob_S_list.append(prob_S_i)

    prob_S = np.array(prob_S_list)
    # prob_S = np.mean(prob_S, axis=0)

    prob_B = np.exp(data_tesr_SR_B_logprob) * test_mass_prob_B

    return prob_S, prob_B
