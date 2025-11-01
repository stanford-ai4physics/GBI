import os, sys
import importlib
import luigi
import copy
import law
import numpy as np
import pandas as pd
from scipy.stats import rv_histogram
import torch
import json
from src.utils.utils import NumpyEncoder, str_encode_value
import time, random


def train_model_S(
    input_dir,
    output_dir,
    s_ratio,
    w_value,
    batch_size,
    epoches=200,
    early_stopping_patience=10,
    train_random_seed=42,
    device="cuda",
):
    # fixing random seed
    torch.manual_seed(train_random_seed)

    print("loading data")
    # load data
    data_trainval_SR_B = np.load(
        input_dir["preprocessing"]["data_trainval_SR_model_B"].path
    )
    data_trainval_SR_S = np.load(
        input_dir["preprocessing"]["data_trainval_SR_model_S"].path
    )
    # bkg prob predicted by model_B
    data_trainval_SR_B_prob = np.load(input_dir["bkgprob"]["log_B_trainval"].path)

    # according to train random seed, shuffle index
    np.random.seed(train_random_seed)
    index_shuffle = np.random.permutation(data_trainval_SR_B.shape[0])
    data_trainval_SR_B = data_trainval_SR_B[index_shuffle]
    data_trainval_SR_S = data_trainval_SR_S[index_shuffle]
    data_trainval_SR_B_prob = data_trainval_SR_B_prob[index_shuffle]

    # split train val in 0.8, 0.2
    split_index = int(data_trainval_SR_B.shape[0] * 0.8)
    data_train_SR_B = data_trainval_SR_B[:split_index]
    data_val_SR_B = data_trainval_SR_B[split_index:]

    data_train_SR_S = data_trainval_SR_S[:split_index]
    data_val_SR_S = data_trainval_SR_S[split_index:]

    data_train_SR_B_prob = data_trainval_SR_B_prob[:split_index]
    data_val_SR_B_prob = data_trainval_SR_B_prob[split_index:]

    print("num sig in train: ", (data_train_SR_B[:, -1] == 1).sum())
    print("num sig in val: ", (data_val_SR_B[:, -1] == 1).sum())

    # p(m) for bkg model p(x|m)
    with open(input_dir["preprocessing"]["SR_mass_hist"].path, "r") as f:
        mass_hist = json.load(f)
    SR_mass_hist = np.array(mass_hist["hist"])
    SR_mass_bins = np.array(mass_hist["bins"])
    density_back = rv_histogram((SR_mass_hist, SR_mass_bins))

    # data to train model_S
    traintensor_S = torch.from_numpy(data_train_SR_S.astype("float32")).to(device)
    valtensor_S = torch.from_numpy(data_val_SR_S.astype("float32")).to(device)

    # convert data to torch tensors
    # log B prob in SR
    log_B_train_tensor = torch.from_numpy(data_train_SR_B_prob.astype("float32")).to(
        device
    )
    log_B_val_tensor = torch.from_numpy(data_val_SR_B_prob.astype("float32")).to(device)

    # bkg in SR
    traintensor_B = torch.from_numpy(data_train_SR_B.astype("float32")).to(device)
    valtensor_B = torch.from_numpy(data_val_SR_B.astype("float32")).to(device)
    # p(m) for bkg model p(x|m)
    # Convert to float32 for MPS compatibility (Apple Silicon doesn't support float64)
    train_mass_prob_B = torch.from_numpy(
        density_back.pdf(traintensor_B[:, 0].cpu().detach().numpy()).astype("float32")
    ).to(device)
    val_mass_prob_B = torch.from_numpy(
        density_back.pdf(valtensor_B[:, 0].cpu().detach().numpy()).astype("float32")
    ).to(device)

    print("data loaded")
    print("train val data shape: ", traintensor_S.shape, valtensor_S.shape)
    print("w_true: ", s_ratio)

    # define training input tensors
    # from src.models.model_S import modelSDataLoader

    # traindataset = modelSDataLoader(
    #     (traintensor_S, log_B_train_tensor, train_mass_prob_B),
    #     device=device,
    #     batch_size=batch_size,
    # )
    # valdataset = modelSDataLoader(
    #     (valtensor_S, log_B_val_tensor, val_mass_prob_B),
    #     device=device,
    #     batch_size=batch_size,
    # )
    # trainloader = torch.utils.data.DataLoader(
    #     traindataset, batch_size=None, shuffle=False
    # )
    # valloader = torch.utils.data.DataLoader(valdataset, batch_size=None, shuffle=False)

    train_tensor = torch.utils.data.TensorDataset(
        traintensor_S, log_B_train_tensor, train_mass_prob_B
    )
    val_tensor = torch.utils.data.TensorDataset(
        valtensor_S, log_B_val_tensor, val_mass_prob_B
    )

    trainloader = torch.utils.data.DataLoader(
        train_tensor, batch_size=batch_size, shuffle=True
    )

    test_batch_size = batch_size * 5
    valloader = torch.utils.data.DataLoader(
        val_tensor, batch_size=test_batch_size, shuffle=False
    )

    # define model
    from src.models.model_S import r_anode_mass_joint_untransformed, flows_model_RQS

    model_S = flows_model_RQS(device=device, num_features=5, context_features=None)
    # print num of params
    print("model S num of params: ", sum(p.numel() for p in model_S.parameters()))
    optimizer = torch.optim.AdamW(model_S.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    trainloss_list = []
    valloss_list = []
    min_val_loss = np.inf
    patience = 0

    # define training
    for epoch in range(epoches):

        train_loss = r_anode_mass_joint_untransformed(
            model_S=model_S,
            w=w_value,
            optimizer=optimizer,
            data_loader=trainloader,
            device=device,
            mode="train",
        )
        val_loss = r_anode_mass_joint_untransformed(
            model_S=model_S,
            w=w_value,
            optimizer=optimizer,
            data_loader=valloader,
            device=device,
            mode="val",
        )

        # torch.save(model_S.state_dict(), scrath_path+'/model_S_epoch_'+str(epoch)+'_w_'+str_encode_value(w_value)+'seed_'+str(train_random_seed)+'.pt')
        # save model
        state_dict = copy.deepcopy(
            {k: v.cpu() for k, v in model_S.state_dict().items()}
        )

        # early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience = 0
            best_model = state_dict
        else:
            patience += 1
            if patience > early_stopping_patience:
                print("early stopping at epoch: ", epoch)
                break

        trainloss_list.append(train_loss)
        valloss_list.append(val_loss)
        print("Epoch: ", epoch, "Train loss: ", train_loss, "Val loss: ", val_loss)

        scheduler.step()

    # save train and val loss
    time.sleep(random.uniform(0, 30))
    trainloss_list = np.array(trainloss_list)
    valloss_list = np.array(valloss_list)
    output_dir["trainloss_list"].parent.touch()
    np.save(output_dir["trainloss_list"].path, trainloss_list)
    np.save(output_dir["valloss_list"].path, valloss_list)

    # save best models with lowest val loss
    print("best model val loss: ", min_val_loss)
    time.sleep(random.uniform(0,2))
    torch.save(best_model, output_dir["sig_model"].path)

    # save metadata
    metadata = {
        "w_true": s_ratio,
        "num_train_events": traintensor_S.shape[0],
        "num_val_events": valtensor_S.shape[0],
    }
    metadata["min_val_loss_list"] = [min_val_loss]
    metadata["min_train_loss_list"] = [trainloss_list.min()]

    with open(output_dir["metadata"].path, "w") as f:
        json.dump(metadata, f, cls=NumpyEncoder)


def pred_model_S(model_dir, data_train_SR_S, device="cuda"):

    # data to train model_S
    testtensor_S = torch.from_numpy(data_train_SR_S.astype("float32")).to(device)

    # define model
    from src.models.model_S import flows_model_RQS

    model_S = flows_model_RQS(device=device, num_features=5, context_features=None)

    model_S.load_state_dict(torch.load(model_dir, weights_only=True))

    model_S.eval()

    model_S_log_prob = model_S.log_prob(inputs=testtensor_S[:, :-1])

    model_S_log_prob[torch.isnan(model_S_log_prob)] = 0

    model_S_prob = torch.exp(model_S_log_prob)

    return model_S_prob.cpu().detach().numpy().flatten()
