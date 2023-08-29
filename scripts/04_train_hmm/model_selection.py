import glob
from hmmlearn import hmm
import numpy as np
from tqdm import trange
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import make_hmm_data



def get_avg_log_likelihoods(
    data_dir,
    dataset_name,
    cols,
    max_components=8,
    bootstrap=5,
    cov_type="diag",
    n_iter=10000,
    first_n=None,
):
    sort = True

    best_scores = []
    mean_scores = []
    scores_stdev = []
    best_models = []
    aics = []
    bics = []

    train_dfs, test_dfs, train_data, test_data = make_hmm_data(
        data_dir, cols, sort=sort, first_n=first_n
    )

    train_lengths = [len(df) for df in train_dfs]
    test_lengths = [len(df) for df in test_dfs]

    for i in trange(1, max_components + 1):
        scores_buf = []
        aics_buf = []
        bics_buf = []
        best_score = -np.inf
        best_model = None
        for seed in range(bootstrap):
            model = hmm.GaussianHMM(
                n_components=i, covariance_type=cov_type, n_iter=n_iter
            )
            model.fit(train_data, lengths=train_lengths)
            score = model.score(test_data, lengths=test_lengths)
            aics_buf.append(model.aic(test_data, lengths=test_lengths))
            bics_buf.append(model.bic(test_data, lengths=test_lengths))
            scores_buf.append(score)
            if score > best_score:
                best_score = score
                best_model = model
        best_scores.append(best_score)
        mean_scores.append(np.mean(scores_buf))
        scores_stdev.append(np.std(scores_buf))
        aics.append(np.mean(aics_buf))
        bics.append(np.mean(bics_buf))
        best_models.append(best_model)

    return {
        "best_scores": best_scores,
        "mean_scores": mean_scores,
        "scores_stdev": scores_stdev,
        "aics": aics,
        "bics": bics,
        "best_models": best_models,
    }


def plot_avg_log_likelihood(data, dataset_name, output_file, max_components=8):
    fig, ax = plt.subplots()
    x = np.arange(1, max_components + 1)
    colors = sns.color_palette("Set2")

    ax.errorbar(
        x,
        data["mean_scores"],
        yerr=data["scores_stdev"],
        label="log density (left axis)",
        color=colors[0],
    )

    ax2 = ax.twinx()
    ax2.plot(x, data["aics"], label="AIC (right axis)", color=colors[1])
    ax2.plot(x, data["bics"], label="BIC (right axis)", color=colors[2])

    handles, _ = ax.get_legend_handles_labels()

    ax.legend(handles=handles + ax2.lines, loc="lower right")
    ax.set_xlabel("Number of HMM components")
    ax.set_ylabel("Average log density (higher is better)")
    ax2.set_ylabel("AIC/BIC (lower is better)")
    fig.suptitle(f"{dataset_name}")
    fig.savefig(f"{output_file}.png", bbox_inches="tight")


def main(
    data_dir,
    output_file,
    dataset_name,
    exp_type,
    cov_type,
    n_iter,
    first_n,
    max_components,
):
    if exp_type == "base":
        cols = [
            "l1",
            "l2",
            "trace",
            "spectral",
            "code_sparsity",
            "computational_sparsity",
            # "mean_singular_value",
            # "var_singular_value",
            "mean_lambda",
            "variance_lambda",
            "mean_w",
            "median_w",
            "var_w",
            "mean_b",
            "median_b",
            "var_b",
        ]
    elif exp_type == "ablation":
        cols = [
            "l1",
            "l2",
            "trace",
            "variance_lambda",
            "mean_w",
            "mean_b",
        ]
    elif exp_type == "ablation2":
        cols = ["l2"]
    elif exp_type == "ablation3":
        cols = ["l1"]
    elif exp_type == "modular_best":
        cols = ['var_w', 'l1', 'l2']
    elif exp_type == "parities_best":
        cols = ['spectral', 'variance_lambda']
    else:
        raise ValueError(f"Unknown exp_type {exp_type}")

    data = get_avg_log_likelihoods(
        data_dir, # +"*scaling1.0*",
        dataset_name,
        cols,
        max_components=max_components,
        bootstrap=5,
        cov_type=cov_type,
        n_iter=n_iter,
        first_n=first_n,
    )

    with open(output_file + ".pkl", "wb") as f:
        pickle.dump(data, f)

    plot_avg_log_likelihood(
        data,
        f"{dataset_name}, {cov_type} covariance",
        output_file,
        max_components=max_components,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMM")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to data, expecting csvs"
    )
    parser.add_argument("--output_file", type=str, default="./output/hmm_data")
    parser.add_argument("--dataset_name", type=str, default="hmm_data")
    parser.add_argument("--cov_type", type=str, default="diag")
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--first_n", type=int, default=1000)
    parser.add_argument("--max_components", type=int, default=8)
    parser.add_argument("--exp_type", type=str, default="base")
    args = parser.parse_args()
    main(
        args.data_dir,
        args.output_file,
        args.dataset_name,
        args.exp_type,
        args.cov_type,
        args.num_iters,
        args.first_n,
        args.max_components,
    )
