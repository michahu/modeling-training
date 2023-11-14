# %%
from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
import torch.nn as nn

import evaluate
import wandb
import os
import json
import argparse

from src.data import ModularArithmetic
from src.model import Transformer, custom_weight_init
from src.metrics import get_metrics_transformer

# %%
# replicating https://arxiv.org/pdf/2301.05217.pdf

DATASET_NAME = "modular"


# %%
def get_dataloaders(
    train_bsz: int = 2048, test_bsz: int = 2048, data_split: float = 0.3
):
    dataset = ModularArithmetic("add")
    train, test = torch.utils.data.random_split(
        dataset,
        [int(len(dataset) * data_split), len(dataset) - int(len(dataset) * data_split)],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train, shuffle=True, batch_size=train_bsz
    )
    test_dataloader = torch.utils.data.DataLoader(
        test, shuffle=False, batch_size=test_bsz
    )
    return train_dataloader, test_dataloader


# %%
def train(**config):
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    train_bsz = config["train_bsz"]
    cpu = config["cpu"]
    weight_decay = config["weight_decay"]
    eval_every = config["eval_every"]
    nheads = 4
    config["n_heads"] = nheads
    init_scaling = config["init_scaling"]
    optim_name = config["optimizer"]
    output_dir = config["output_dir"]
    seed = config["seed"]
    wandb = config["wandb"]
    use_ln = config["use_ln"]
    test_bsz = 2048
    clip_grad = config["clip_grad"]

    # make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if wandb:
        wandb.init(config=config, project="modeling-training")
    set_seed(seed)

    accelerator = Accelerator(cpu=cpu)  # , mixed_precision="fp16")
    device = accelerator.device

    train_dataloader, test_dataloader = get_dataloaders(
        train_bsz=train_bsz, test_bsz=test_bsz
    )

    model = Transformer(
        d_model=128,
        d_head=32,
        d_vocab=114,
        num_heads=nheads,
        num_layers=1,
        n_ctx=3,
        use_ln=use_ln,
    )

    # model.apply(lambda m: custom_weight_init(m, init_scaling=init_scaling))

    for p in model.parameters():
        p.data *= init_scaling

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optim_name} not supported")

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    criterion = nn.CrossEntropyLoss()
    train_accuracy = evaluate.load(
        "accuracy", experiment_id=DATASET_NAME, keep_in_memory=True
    )
    test_accuracy = evaluate.load(
        "accuracy", experiment_id=DATASET_NAME, keep_in_memory=True
    )

    # create output dir if it doesn't exist
    run_output_dir = os.path.join(
        output_dir, f"lr{lr}_{optim_name}_seed{seed}_scaling{init_scaling}"
    )
    if not os.path.exists(os.path.join(run_output_dir)):
        os.makedirs(os.path.join(run_output_dir))

    model.train()
    for epoch in range(num_epochs):
        train_loss = 0

        for x, y in train_dataloader:
            output = model(x)[:, -1, :]
            loss = criterion(output, y)

            loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), lr * 10)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.detach().cpu().item()

            with torch.no_grad():
                predictions = torch.argmax(output, dim=1)
                predictions, y = accelerator.gather_for_metrics((predictions, y))
                train_accuracy.add_batch(predictions=predictions, references=y)

        if epoch % eval_every == 0:
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for x, y in test_dataloader:
                    output = model(x)[:, -1, :]
                    eval_loss += criterion(output, y).item()

                    predictions = torch.argmax(output, dim=1)
                    predictions, y = accelerator.gather_for_metrics((predictions, y))
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

            data = get_metrics_transformer(model, nheads)
            data["train_loss"] = train_loss
            data["eval_loss"] = eval_loss
            data["train_accuracy"] = train_accuracy_metric
            data["eval_accuracy"] = test_accuracy_metric

            with open(
                os.path.join(run_output_dir, f"epoch{epoch}.json"),
                "w",
            ) as f:
                json.dump(data, f)

            # save model
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--init_scaling", type=float, default=1.0)
    parser.add_argument("--train_bsz", type=int, default=256)
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "sgd"]
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use_ln", action="store_true")
    parser.add_argument("--clip_grad", action="store_true")
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
