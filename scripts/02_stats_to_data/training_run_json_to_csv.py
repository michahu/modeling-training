import wandb
import numpy as np
import json
import pandas as pd
import re
import os
import glob

import argparse
from itertools import chain

from src.utils import get_stats_for_run, get_stats_for_cnn


# %%
def find_directories(file_path):
    directories = []
    # Open file
    with open(file_path, "r") as f:
        # Read file line by line
        for line in f:
            # Check if line contains the specified directory format
            match = re.search(
                r"/scratch/myh2014/modeling-training/wandb/run-\d{8}_\d{6}-[a-z0-9]{8}",
                line,
            )
            if match:
                directories.append(match.group(0))
    return directories


def extract_directory_names(directory_paths):
    directory_names = []
    for directory_path in directory_paths:
        directory_names.append(directory_path[-8:])
    return directory_names

def parse_logs(logs_pth):
    with open(logs_pth, "r") as f:
        data = json.load(f)
    
    train_losses = pd.DataFrame(data['log_history'], columns=['loss', 'step']).dropna()
    eval_losses = pd.DataFrame(data['log_history'], columns=['eval_loss', 'step']).dropna()

    parsed_data = train_losses.merge(eval_losses, on='step')

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
    if exp_type == "slurm":
        api = wandb.Api()
        dirs = find_directories(slurm_file)
        run_ids = extract_directory_names(dirs)

        for d, run_id in zip(dirs, run_ids):
            run = api.run(f"myhu/modeling-training/{run_id}")
            data = run.history(samples=max_iters // save_every, pandas=True)
            print(run_id, len(data))

            vals = get_stats_for_run(d, is_transformer, has_loss)

            vals["train_accuracy"] = data["train_accuracy.accuracy"]
            vals["eval_accuracy"] = data["eval_accuracy.accuracy"]
            vals["train_loss"] = data["train_loss"]
            vals["eval_loss"] = data["eval_loss"]

            df = pd.DataFrame(vals)

            directory_path = f"{save_dir}/{save_every}"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            df.to_csv(f"{directory_path}/{run_id}.csv")

    elif exp_type == "glue":
        print(input_dir)
        pths = glob.glob(input_dir + "/stats*")
        vals = get_stats_for_run(
            pths,
            is_transformer,
            has_loss
        )
        df = pd.DataFrame(vals)
        loss_df = parse_logs(glob.glob(input_dir + "/trainer_state.json")[0])
        df = df.merge(loss_df, on='step')
        seed = re.findall(r'\d+', input_dir.split('/')[-1])

        df.to_csv(f"{save_dir}_{seed}")

    elif exp_type == "modular":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            # for optimizer in ['adamw']:
            #     for lr in [0.001, 0.002, 0.0005]:
            #         for init_scaling in [2.0, 1.0, 0.5]:
            optimizer = 'adamw'
            lr = 0.001
            init_scaling = 1.0
            d = input_dir + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
            print(d)
            pths = glob.glob(d)
            print(pths)
            vals = get_stats_for_run(
                pths,
                is_transformer,
                has_loss
            )
            df = pd.DataFrame(vals)

            file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

            df.to_csv(f"{save_dir}/{file_name}")

    elif exp_type == "parities":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            d = input_dir + f"*seed{seed}*/*.json"
            pths = glob.glob(d)
            vals = get_stats_for_run(
                pths,
                is_transformer,
                has_loss
            )
            df = pd.DataFrame(vals)

            file_name = f"seed{seed}.csv"

            df.to_csv(f"{save_dir}/{file_name}")
    elif exp_type == "mnist":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            optimizer = 'adamw'
            lr = 0.001
            init_scaling = 1.0
            d = input_dir + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
            pths = glob.glob(d)
            vals = get_stats_for_run(
                pths,
                is_transformer,
                has_loss
            )
            df = pd.DataFrame(vals)

            file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

            df.to_csv(f"{save_dir}/{file_name}")
    elif exp_type == "cnn":
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for seed in range(40):
            optimizer = 'adamw'
            lr = 0.001
            init_scaling = 1.0
            d = input_dir + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
            pths = glob.glob(d)
            vals = get_stats_for_cnn(
                pths,
                has_loss
            )
            df = pd.DataFrame(vals)

            file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

            df.to_csv(f"{save_dir}/{file_name}")

    else:
        raise ValueError("Invalid exp_type.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_transformer", action="store_true")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./data/training_runs")

    parser.add_argument("--slurm_file", type=str)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_iters", type=int, default=5000)

    parser.add_argument("--exp_type", type=str)
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
