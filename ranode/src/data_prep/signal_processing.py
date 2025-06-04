import pandas as pd
import h5py
import numpy as np
from src.data_prep.utils import get_dijetmass_ptetaphi
from src.utils.utils import str_encode_value


def process_signals(input_path, mx, my, s_ratio, seed, type):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=1)
    """

    s_ratio_str = str_encode_value(s_ratio)

    data_all_df = h5py.File(input_path, "r")[f"{mx}_{my}"]
    data_all_df = data_all_df[f"ensemble_{seed}"][s_ratio_str][type][:]

    # shape is (N, 14) where N is the number of events, the columns orders are
    # pt_j1, eta_j1, phi_j1, mj1, Nj1, tau12j1, tau23j1, pt_j2, eta_j2, phi_j2, mj2, Nj2, tau12j2, tau23j2
    # all units are in TeV already

    pt_j1 = data_all_df[:, 0]
    eta_j1 = data_all_df[:, 1]
    phi_j1 = data_all_df[:, 2]
    mj1 = data_all_df[:, 3]
    tau21j1 = data_all_df[:, 5]
    jet1_p4 = np.stack([pt_j1, eta_j1, phi_j1, mj1], axis=1)

    pt_j2 = data_all_df[:, 7]
    eta_j2 = data_all_df[:, 8]
    phi_j2 = data_all_df[:, 9]
    mj2 = data_all_df[:, 10]
    tau21j2 = data_all_df[:, 12]
    jet2_p4 = np.stack([pt_j2, eta_j2, phi_j2, mj2], axis=1)

    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_ptetaphi(jets)

    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.stack([mj1, mj2], axis=1)
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack(
        [mjj, mjmin, mjmax - mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1
    )

    return output


def process_signals_test(
    input_path, output_path, mx, my, s_ratio, seed, use_true_mu=True
):
    """
    Will reprocess the signal such that they have shape (N, 6) where N is the number of events.
    The columns are:
    (mjj, mj1, delta_mj=mj2-mj1, tau21j1=tau2j1/tau1j1, tau21j2=tau2j2/tau1j2, label=1)
    """

    s_ratio_str = str_encode_value(s_ratio)

    if use_true_mu:
        data_all_df = h5py.File(input_path, "r")[f"{mx}_{my}"]
        data_all_df = data_all_df[f"ensemble_{seed}"][s_ratio_str]["x_test"][:]
    else:
        raise NotImplementedError("using all events for testing is not implemented yet")

    # shape is (N, 14) where N is the number of events, the columns orders are
    # pt_j1, eta_j1, phi_j1, mj1, Nj1, tau12j1, tau23j1, pt_j2, eta_j2, phi_j2, mj2, Nj2, tau12j2, tau23j2
    # all units are in TeV already

    pt_j1 = data_all_df[:, 0]
    eta_j1 = data_all_df[:, 1]
    phi_j1 = data_all_df[:, 2]
    mj1 = data_all_df[:, 3]
    tau21j1 = data_all_df[:, 5]
    jet1_p4 = np.stack([pt_j1, eta_j1, phi_j1, mj1], axis=1)

    pt_j2 = data_all_df[:, 7]
    eta_j2 = data_all_df[:, 8]
    phi_j2 = data_all_df[:, 9]
    mj2 = data_all_df[:, 10]
    tau21j2 = data_all_df[:, 12]
    jet2_p4 = np.stack([pt_j2, eta_j2, phi_j2, mj2], axis=1)

    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_ptetaphi(jets)

    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.stack([mj1, mj2], axis=1)
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack(
        [mjj, mjmin, mjmax - mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1
    )

    np.save(output_path, output)

    # target_process_df = data_all_df.query(f"mx=={mx} & my=={my}")

    # # get jet p4 info to calculate dijet mjj
    # jet1_p4 = target_process_df[["ptj1", "etaj1", "phij1", "mj1"]].values
    # jet2_p4 = target_process_df[["ptj2", "etaj2", "phij2", "mj2"]].values
    # jets = np.stack([jet1_p4, jet2_p4], axis=1)
    # mjj = get_dijetmass_ptetaphi(jets) / TeV

    # # get other features
    # # get mj1 and mj2, sort them with mj1 being the smaller one
    # mj1mj2 = np.array(target_process_df[['mj1', 'mj2']]) / TeV
    # mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    # mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    # tau21j1 = target_process_df["tau2j1"].values / ( 1e-5 + target_process_df["tau1j1"].values )
    # tau21j2 = target_process_df["tau2j2"].values / ( 1e-5 + target_process_df["tau1j2"].values )
    # tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    # tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    # tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # output = np.stack([mjj, mjmin, mjmax-mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1)

    # np.save(output_path, output)


def process_raw_signals(input_path, output_path, mx, my):
    data_all_df = pd.read_hdf(input_path)
    target_process_df = data_all_df.query(f"mx=={mx} & my=={my}")

    # get jet p4 info to calculate dijet mjj
    jet1_p4 = target_process_df[["ptj1", "etaj1", "phij1", "mj1"]].values
    jet2_p4 = target_process_df[["ptj2", "etaj2", "phij2", "mj2"]].values
    jets = np.stack([jet1_p4, jet2_p4], axis=1)
    mjj = get_dijetmass_ptetaphi(jets) / 1000

    from config.configs import SR_MIN, SR_MAX

    mask_mjj = (mjj > SR_MIN) & (mjj < SR_MAX)
    target_process_df = target_process_df[mask_mjj]
    mjj = mjj[mask_mjj]

    # get other features
    # get mj1 and mj2, sort them with mj1 being the smaller one
    mj1mj2 = np.array(target_process_df[["mj1", "mj2"]]) / 1000
    mjmin = mj1mj2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    mjmax = mj1mj2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    # get tau21j1, tau21j2 and sort by mj1 mj2 in the same way
    tau21j1 = target_process_df["tau2j1"].values / (
        1e-5 + target_process_df["tau1j1"].values
    )
    tau21j2 = target_process_df["tau2j2"].values / (
        1e-5 + target_process_df["tau1j2"].values
    )
    tau21j1j2 = np.stack([tau21j1, tau21j2], axis=1)
    tau21min = tau21j1j2[range(len(mj1mj2)), np.argmin(mj1mj2, axis=1)]
    tau21max = tau21j1j2[range(len(mj1mj2)), np.argmax(mj1mj2, axis=1)]

    output = np.stack(
        [mjj, mjmin, mjmax - mjmin, tau21min, tau21max, np.ones(len(mj1mj2))], axis=1
    )

    print(f"Num signals for mx={mx}, my={my}: {len(output)}")

    if output_path is not None:
        np.save(output_path, output)

    else:
        return output
