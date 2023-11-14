# %%
from accelerate import Accelerator
from accelerate.utils import set_seed

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import MLP, LeNet5

from torchvision import datasets, transforms
import torchvision.models as models

import os
import json
import wandb
import evaluate
from tqdm import trange
import argparse

import numpy as np


from src.metrics import get_metrics_mlp, get_metrics_lenet5, get_metrics_resnet18
from src.model import MyResNet, MyBasicBlock

# %%
# replicating https://arxiv.org/pdf/2210.01117.pdf

MNIST_MEAN=(0.1307,)
MNIST_STD=(0.3081,)

CIFAR100_MEAN=(0.5071, 0.4867, 0.4408)
CIFAR100_STD=(0.2675, 0.2565, 0.2761)
CIFAR_STATS=(CIFAR100_MEAN, CIFAR100_STD)


# %%
def get_dataloaders(dataset_name, model_name, train_bsz, test_bsz, num_workers=4):
    if dataset_name == 'mnist':
        if model_name == 'mlp':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
        else:
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean = MNIST_MEAN, std = MNIST_STD), # known mean and std for mnist
            ])
        train_data = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )

        # you can switch "./data" to your download directory of choice 
        test_data = datasets.MNIST("./data", train=False, transform=transform)

    elif dataset_name == 'cifar100':
        train_data = datasets.CIFAR100(
            "./data", train=True, download=True # , transform=transform
        )

        x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])

        # calculate the mean and std along the (0, 1) axes
        # preprocessing taken from https://www.kaggle.com/code/yiweiwangau/cifar-100-resnet-pytorch-75-17-accuracy
        # https://jovian.com/kumar-shailesh1597/cifar100-resnet18
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(*CIFAR_STATS,inplace=True)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*CIFAR_STATS)])

        # reload train data, processing this time with transform
        train_data = datasets.CIFAR100("./data",
                                         train=True,
                                         download=True,
                                         transform=transform_train)
        test_data = datasets.CIFAR100("./data", train=False, transform=transform_test)

    else:
        raise ValueError(f"unsupported dataset {dataset_name}")

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=train_bsz, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=test_bsz, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_dataloader, test_dataloader


# %%
def train(**config):
    input_dim = config["input_dim"]
    hidden_dims = config["hidden_dims"]
    output_dim = config["output_dim"]
    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    lr = config["lr"]
    init_scaling = config["init_scaling"]
    train_bsz = config["train_bsz"]
    train_subsample = config["train_subsample"]
    cpu = config["cpu"]
    eval_every = config["eval_every"]
    seed = config["seed"]
    use_wandb = config['wandb']
    output_dir = config["output_dir"]
    optim_name = 'adamw'
    dataset_name = config['dataset_name']
    model_name = config['model_name']
    use_batch_norm = config['use_batch_norm']
    use_residual = config['use_residual']
    test_bsz = train_bsz * 2

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "output_dim": output_dim,
        "num_epochs": num_epochs,
        "lr": lr,
        "init_scaling": init_scaling,
        "train_bsz": train_bsz,
        "train_subsample": train_subsample,
        "cpu": cpu,
        "seed": seed,
        "dataset": dataset_name,
        "model": model_name,
    }

    print(config)

    if use_wandb:
        wandb.init(config=config, project="modeling-training")

    set_seed(seed)

    accelerator = Accelerator(cpu=cpu)  # , mixed_precision="fp16")

    train_dataloader, test_dataloader = get_dataloaders(
        dataset_name, model_name, train_bsz=train_bsz, test_bsz=test_bsz, num_workers=num_workers
    )  # eval bsz doesn't matter

    # only resnet18 works for cifar100
    if model_name == "mlp":
        model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            # init_scaling=init_scaling,
            kaiming_uniform=False,
        )
        get_metrics = get_metrics_mlp
        run_output_dir = os.path.join(output_dir, f"lr{lr}_{optim_name}_seed{seed}_scaling{init_scaling}")
    elif model_name == "lenet":
        model = LeNet5()
        get_metrics = get_metrics_lenet5
        run_output_dir = os.path.join(output_dir, f"lr{lr}_{optim_name}_seed{seed}_scaling{init_scaling}")
    elif model_name == "resnet18":
        # model = models.resnet18(weights=None, num_classes=100)
        # need to replace the first layer because the model was built for imagenet
        model = MyResNet(use_batch_norm, use_residual, block=MyBasicBlock, layers=[2,2,2,2], num_classes=100)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        get_metrics = get_metrics_resnet18

        run_output_dir = os.path.join(output_dir, f"{use_batch_norm}_{use_residual}" ,f"lr{lr}_{optim_name}_seed{seed}_scaling{init_scaling}")
    
    else:
        raise ValueError(f"unsupported model {model_name}")

    
    if not os.path.exists(os.path.join(run_output_dir)):
        os.makedirs(os.path.join(run_output_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#, weight_decay=0.5e-4)

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # criterion = nn.MSELoss()
    # NB: recall that nn.CEL() already divides by batch size
    criterion = nn.CrossEntropyLoss()
    train_accuracy = evaluate.load(
        "accuracy", experiment_id=dataset_name+str(seed), keep_in_memory=True
    )
    test_accuracy = evaluate.load(
        "accuracy", experiment_id=dataset_name+str(seed), keep_in_memory=True
    )

    if dataset_name == 'mnist':
        num_classes = 10
    else:
        num_classes = 100

    steps = 0
    model.train()
    for epoch in range(num_epochs):
        
        for batch in train_dataloader:
            x, y = batch
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            output = model(x)
            loss = criterion(output, y_onehot)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_loss = loss.detach().cpu().item()

            with torch.no_grad():
                predictions = torch.argmax(output, dim=1)
                predictions, y = accelerator.gather_for_metrics((predictions, y))
                train_accuracy.add_batch(predictions=predictions, references=y)

            steps += 1

            if steps % eval_every == 0:
                model.eval()
                eval_loss = 0
                with torch.no_grad():
                    for batch in test_dataloader:
                        x, y = batch
                        y_onehot = F.one_hot(y, num_classes=num_classes).float()
                        output = model(x)

                        eval_loss += criterion(output, y_onehot).item()

                        predictions = torch.argmax(output, dim=1)
                        predictions, y = accelerator.gather_for_metrics((predictions, y))
                        test_accuracy.add_batch(predictions=predictions, references=y)

                eval_loss = eval_loss / len(test_dataloader) 
                train_accuracy_metric = train_accuracy.compute()
                test_accuracy_metric = test_accuracy.compute()

                if use_wandb:
                    wandb.log(
                        data={
                            "train_loss": train_loss, # loss from the latest batch
                            "eval_loss": eval_loss,
                            "train_accuracy": train_accuracy_metric,
                            "eval_accuracy": test_accuracy_metric,
                        },
                        step=epoch,
                    )
                # custom eval
                data = get_metrics(model)
                data['train_loss'] = train_loss
                data['eval_loss'] = eval_loss
                data['train_accuracy'] = train_accuracy_metric
                data['eval_accuracy'] = test_accuracy_metric

                with open(os.path.join(run_output_dir, f"step{steps}.json"), "w") as f:
                    json.dump(data, f)

                # accelerator.wait_for_everyone()
                # accelerator.save(model.state_dict(), os.path.join(run_output_dir, f"model_{steps}.pt"))
                model.train()

    accelerator.wait_for_everyone()
    accelerator.save(model.state_dict(), os.path.join(run_output_dir, f"model.pt"))

    if wandb:
        wandb.finish()


# %%


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--input_dim", type=int, default=784)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[800])
    parser.add_argument("--output_dim", type=int, default=10)
    parser.add_argument("--init_scaling", type=float, default=1.0)
    parser.add_argument("--train_bsz", type=int, default=256)
    parser.add_argument("--train_subsample", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="mnist", choices=['mnist', 'cifar100'])
    parser.add_argument("--model_name", type=str, default="resnet18", choices=['mlp', 'lenet', 'resnet18'])
    parser.add_argument("--use_batch_norm", action="store_true")
    parser.add_argument("--use_residual", action="store_true")
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
