import json
import pandas as pd
import re
import os
import glob
import logging

import argparse

from src.utils import get_stats_for_run, get_stats_for_cnn


def parse_logs(logs_pth):
    with open(logs_pth, "r") as f:
        data = json.load(f)

    train_losses = pd.DataFrame(data["log_history"], columns=["loss", "step"]).dropna()
    eval_losses = pd.DataFrame(
        data["log_history"], columns=["eval_loss", "step"]
    ).dropna()

    parsed_data = train_losses.merge(eval_losses, on="step")

    return parsed_data


def main(
    exp_type,
    slurm_file,
    save_dir,
    save_every,
    max_iters,
    is_transformer,
    has_loss,
    input_dir=None,
):
    if exp_type == "glue":
        logging.info(input_dir)
        pths = glob.glob(input_dir + "/stats*")
        vals = get_stats_for_run(pths, is_transformer, has_loss)
        df = pd.DataFrame(vals)
        loss_df = parse_logs(glob.glob(input_dir + "/trainer_state.json")[0])
        df = df.merge(loss_df, on="step")
        seed = re.findall(r"\d+", input_dir.split("/")[-1])

        df.to_csv(f"{save_dir}_{seed}")

    elif exp_type == "modular":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            optimizer = "adamw"
            lr = 0.001
            init_scaling = 1.0
            d = (
                input_dir
                + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
            )
            pths = glob.glob(d)
            logging.info(pths)
            vals = get_stats_for_run(pths, is_transformer, has_loss)
            df = pd.DataFrame(vals)

            file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

            df.to_csv(f"{save_dir}/{file_name}")

    elif exp_type == "parities":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            d = input_dir + f"*seed{seed}*/*.json"
            pths = glob.glob(d)
            vals = get_stats_for_run(pths, is_transformer, has_loss)
            df = pd.DataFrame(vals)

            file_name = f"seed{seed}.csv"

            df.to_csv(f"{save_dir}/{file_name}")
    elif exp_type == "mnist":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            optimizer = "adamw"
            lr = 0.001
            init_scaling = 1.0
            d = (
                input_dir
                + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
            )
            pths = glob.glob(d)
            vals = get_stats_for_run(pths, is_transformer, has_loss)
            df = pd.DataFrame(vals)

            file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

            df.to_csv(f"{save_dir}/{file_name}")
    elif exp_type == "cnn":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            optimizer = "adamw"
            lr = 0.001
            init_scaling = 1.0
            d = (
                input_dir
                + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
            )
            pths = glob.glob(d)
            vals = get_stats_for_cnn(pths, has_loss)
            df = pd.DataFrame(vals)

            file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

            df.to_csv(f"{save_dir}/{file_name}")

    else:
        raise ValueError("Invalid exp_type.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_transformer", action="store_true")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./data/training_runs")

    parser.add_argument("--slurm_file", type=str)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_iters", type=int, default=5000)

    parser.add_argument(
        "--exp_type",
        type=str,
        options=["glue", "modular", "parities", "mnist", "cnn"],
    )
    parser.add_argument("--has_loss", action="store_true")
    args = parser.parse_args()

    main(
        args.exp_type,
        args.slurm_file,
        args.save_dir,
        args.save_every,
        args.max_iters,
        args.is_transformer,
        args.has_loss,
        args.input_dir,
    )
