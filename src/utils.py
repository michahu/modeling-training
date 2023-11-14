import json
from itertools import chain
from collections import defaultdict
import numpy as np
import os
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from graphviz import Digraph


def get_markov_chain(matrix):
    n = matrix.shape[0]
    dot = Digraph(comment="Markov Chain")
    dot.attr(rankdir="LR", size="8,5")
    dot.attr("node", shape="circle")

    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                dot.edge(str(i), str(j), label=str(matrix[i][j]))

    return dot


def make_hmm_data(data_dir, cols, sort=True, first_n=1000):
    print(data_dir)
    print(glob.glob(data_dir + "*"))
    if sort:
        try:
            dfs = [
                pd.read_csv(file)
                .sort_values("step")
                .reset_index(drop=True)[cols]  # restrict to cols of interest
                .head(first_n)
                for file in glob.glob(data_dir + "*")
            ]
        except KeyError:
            dfs = [
                pd.read_csv(file)
                .sort_values("epoch")
                .reset_index(drop=True)[cols]  # restrict to cols of interest
                .head(first_n)
                for file in glob.glob(data_dir + "*")
            ]
    else:
        dfs = [
            pd.read_csv(file)[cols].head(first_n)  # restrict to cols of interest
            for file in glob.glob(data_dir + "*")
        ]

    dfs = [df for df in dfs if not df.isnull().values.any()]  # remove invalid dfs

    train_dfs, test_dfs = train_test_split(dfs, test_size=0.2, random_state=0)

    train_data = np.vstack(
        [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in train_dfs]
    )
    test_data = np.vstack(
        [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in test_dfs]
    )

    return train_dfs, test_dfs, train_data, test_data


def unpack_vals(
    subsample,
    l1,
    l2,
    trace,
    spectral,
    code_sparsity,
    computational_sparsity,
    mean_lambda,
    var_lambda,
):
    l1.append(subsample["l1"])
    l2.append(subsample["l2"])
    trace.append(subsample["trace"])
    spectral.append(subsample["spectral"])
    code_sparsity.append(subsample["code_sparsity"])
    computational_sparsity.append(subsample["computational_sparsity"])
    mean_lambda.append(subsample["mean_singular_value"])
    var_lambda.append(subsample["var_singular_value"])


# TODO: delete this function, fold into cnn method
def get_stats_for_run(file_pths, is_transformer, has_loss=False):
    buf = defaultdict(list)

    for file_pth in file_pths:
        # logging formats have been inconsistent

        # name format: stats_{seed}_{step}.json
        # step = int(os.path.basename(file_pth).split("_")[-1].split(".")[0])

        # name format: stats_{step}epoch_{lr}lr_{optimizer}_{seed}seed
        # step = int(os.path.basename(file_pth).split("_")[1].split("epoch")[0])

        # name format: stats_{step}_losses.json
        # step = int(os.path.basename(file_pth).split("_")[1])

        # name format: epoch{step}.json
        step = int(os.path.basename(file_pth).split(".")[0].split("epoch")[1])

        # name format: stats_step{step}
        # step = int(os.path.basename(file_pth).split("step")[1])

        # name format: step{step}.json
        # step = int(os.path.basename(file_pth).split(".")[0].split("step")[1])

        l1_buf = []
        l2_buf = []
        trace_buf = []
        spectral_buf = []
        code_sparsity_buf = []
        computational_sparsity_buf = []
        mean_lambda_buf = []
        variance_lambda_buf = []

        with open(file_pth, "r") as f:
            data = json.loads(f.read())

        if is_transformer:
            samples = chain(
                data["k"],
                data["v"],
                data["q"],
                data["in_proj"],
                data["ffn_in"],
                data["ffn_out"],
            )
        else:
            samples = data["w"]
        for sample in samples:
            for subsample in sample.values():
                if isinstance(subsample, dict):
                    unpack_vals(
                        subsample,
                        l1_buf,
                        l2_buf,
                        trace_buf,
                        spectral_buf,
                        code_sparsity_buf,
                        computational_sparsity_buf,
                        mean_lambda_buf,
                        variance_lambda_buf,
                    )
                elif isinstance(subsample, list):
                    for subsubsample in subsample:
                        unpack_vals(
                            subsubsample,
                            l1_buf,
                            l2_buf,
                            trace_buf,
                            spectral_buf,
                            code_sparsity_buf,
                            computational_sparsity_buf,
                            mean_lambda_buf,
                            variance_lambda_buf,
                        )

        buf["l1"].append(np.mean(l1_buf))
        buf["l2"].append(np.mean(l2_buf))
        buf["trace"].append(np.mean(trace_buf))
        buf["spectral"].append(np.mean(spectral_buf))
        buf["code_sparsity"].append(np.mean(code_sparsity_buf))
        buf["computational_sparsity"].append(np.mean(computational_sparsity_buf))
        buf["mean_lambda"].append(np.mean(mean_lambda_buf))
        buf["variance_lambda"].append(np.nanmean(variance_lambda_buf))

        # global stats
        mean_w = data["w_all"]["mean"]
        median_w = data["w_all"]["median"]
        var_w = data["w_all"]["var"]
        mean_b = data["b_all"]["mean"]
        median_b = data["b_all"]["median"]
        var_b = data["b_all"]["var"]

        buf["mean_w"].append(mean_w)
        buf["median_w"].append(median_w)
        buf["var_w"].append(var_w)
        buf["mean_b"].append(mean_b)
        buf["median_b"].append(median_b)
        buf["var_b"].append(var_b)

        buf["step"].append(step)

        if has_loss:
            buf["train_loss"].append(data["train_loss"])
            buf["eval_loss"].append(data["eval_loss"])
            buf["train_accuracy"].append(data["train_accuracy"]["accuracy"])
            buf["eval_accuracy"].append(data["eval_accuracy"]["accuracy"])

    return buf


def get_stats_for_cnn(file_pths, has_loss=False):
    # holds the csv data
    buf = defaultdict(list)

    for file_pth in file_pths:
        # name format: step{step}.json
        step = int(os.path.basename(file_pth).split(".")[0].split("step")[1])

        buffer_dict = defaultdict(list)

        with open(file_pth, "r") as f:
            data = json.loads(f.read())
            samples = data["w"]

        for sample in samples:
            for key, val in sample.items():
                if isinstance(val, list):
                    buffer_dict[key].extend(val)
                else:
                    buffer_dict[key].append(val)

        for key, val in buffer_dict.items():
            buf[key].append(np.mean(val))

        del buf["singular_values"]

        # global stats
        mean_w = data["w_all"]["mean"]
        median_w = data["w_all"]["median"]
        var_w = data["w_all"]["var"]
        mean_b = data["b_all"]["mean"]
        median_b = data["b_all"]["median"]
        var_b = data["b_all"]["var"]

        buf["mean_w"].append(mean_w)
        buf["median_w"].append(median_w)
        buf["var_w"].append(var_w)
        buf["mean_b"].append(mean_b)
        buf["median_b"].append(median_b)
        buf["var_b"].append(var_b)

        buf["step"].append(step)

        if has_loss:
            buf["train_loss"].append(data["train_loss"])
            buf["eval_loss"].append(data["eval_loss"])
            buf["train_accuracy"].append(data["train_accuracy"]["accuracy"])
            buf["eval_accuracy"].append(data["eval_accuracy"]["accuracy"])

    return buf
