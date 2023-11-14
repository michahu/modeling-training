# %%
from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
import torch.nn as nn

import evaluate
import numpy as np
import wandb
import os
import json
import argparse

from src.data import SparseParity
from src.model import MLP, custom_weight_init
from src.metrics import get_metrics_mlp

# %%
# replicating https://arxiv.org/pdf/2207.08799.pdf

DATASET_NAME = "sparse-parities"


# note: MultiMarginLoss in PyTorch is awkward to use because it is multiclass
class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(torch.squeeze(output), torch.squeeze(target))
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss


def get_dataloaders(
    train_size, test_size, train_bsz, test_bsz, total_bits=40, parity_bits=3
):
    data = SparseParity(
        train_size + test_size, total_bits=total_bits, parity_bits=parity_bits
    )
    train, test = torch.utils.data.random_split(data, [train_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(
        train, shuffle=True, batch_size=train_bsz
    )
    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=test_bsz
    )
    return train_dataloader, test_dataloader


# %%
def train(**config):
    hidden_dim = config["hidden_dim"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    train_bsz = config["train_bsz"]
    train_size = config["train_size"]
    test_bsz = config["test_bsz"]
    test_size = config["test_size"]
    cpu = config["cpu"]
    seed = config["seed"]
    weight_decay = config["weight_decay"]
    optim_name = config["optim_name"]
    output_dir = config["output_dir"]
    eval_every = config["eval_every"]
    wandb = config["wandb"]
    init_scaling = config["init_scaling"]
    use_batch_norm = config["use_batch_norm"]
    clip_grad = config["clip_grad"]
    use_ln = config["use_ln"]

    # make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if wandb:
        wandb.init(config=config, project="modeling-training")

    accelerator = Accelerator(cpu=cpu)
    device = accelerator.device

    set_seed(seed)

    train_dataloader, test_dataloader = get_dataloaders(
        train_size, test_size, train_bsz, test_bsz
    )

    model = MLP(
        input_dim=40,
        hidden_dims=[hidden_dim],
        output_dim=1,
        kaiming_uniform=False,
        use_batch_norm=use_batch_norm,
        use_layer_norm=use_ln,
    )
    model.apply(lambda m: custom_weight_init(m, init_scaling=init_scaling))
    # if optim_name == "adam":
    #     optimizer = torch.optim.Adam(
    #         model.parameters(), lr=lr, weight_decay=weight_decay
    #     )
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer {optim_name}")

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    criterion = HingeLoss()
    train_accuracy = evaluate.load(
        "accuracy", experiment_id=DATASET_NAME, keep_in_memory=True
    )
    test_accuracy = evaluate.load(
        "accuracy", experiment_id=DATASET_NAME, keep_in_memory=True
    )

    run_output_dir = os.path.join(
        output_dir, f"lr{lr}_{optim_name}_seed{seed}_scaling{init_scaling}"
    )
    if not os.path.exists(os.path.join(run_output_dir)):
        os.makedirs(os.path.join(run_output_dir))

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        eval_loss = 0

        for x, y in train_dataloader:
            output = model(x)
            loss = criterion(output, y).mean()

            loss.backward()

            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), lr * 10)
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.detach().cpu().item()

            with torch.no_grad():
                predictions = torch.flatten(torch.sign(output))
                predictions, y = accelerator.gather_for_metrics(
                    (predictions, y.flatten())
                )
                train_accuracy.add_batch(predictions=predictions, references=y)

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                for x, y in test_dataloader:
                    output = model(x)

                    eval_loss += criterion(output, y).mean().item()

                    predictions = torch.flatten(torch.sign(output))
                    predictions, y = accelerator.gather_for_metrics(
                        (predictions, y.flatten())
                    )
                    test_accuracy.add_batch(predictions=predictions, references=y)

            train_loss = train_loss / len(train_dataloader)
            eval_loss = eval_loss / len(test_dataloader)
            train_accuracy_metric = train_accuracy.compute()
            test_accuracy_metric = test_accuracy.compute()

            if wandb:
                wandb.log(
                    data={
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "train_accuracy": train_accuracy_metric,
                        "eval_accuracy": test_accuracy_metric,
                    },
                    step=epoch,
                )
            data = get_metrics_mlp(model)
            data["train_loss"] = train_loss
            data["eval_loss"] = eval_loss
            data["train_accuracy"] = train_accuracy_metric
            data["eval_accuracy"] = test_accuracy_metric

            with open(
                os.path.join(run_output_dir, f"epoch{epoch}.json"),
                "w",
            ) as f:
                json.dump(data, f)

            # accelerator.wait_for_everyone()
            # accelerator.save(model.state_dict(), os.path.join(run_output_dir, f"model_{epoch}.pt"))
            model.train()

    accelerator.wait_for_everyone()
    accelerator.save(model.state_dict(), os.path.join(run_output_dir, f"model.pt"))

    if wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=1000)
    parser.add_argument("--train_bsz", type=int, default=32)
    parser.add_argument("--init_scaling", type=float, default=1.0)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--test_bsz", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--optim_name", type=str, default="sgd", choices=["adamw", "sgd"]
    )
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_batch_norm", action="store_true")
    parser.add_argument("--use_ln", action="store_true")
    parser.add_argument("--clip_grad", action="store_true")
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
