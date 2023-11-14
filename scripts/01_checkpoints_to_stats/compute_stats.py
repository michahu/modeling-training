from transformers import (
    AutoModelForMaskedLM,
    BertForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, load_from_disk, Dataset

from torch.utils.data import DataLoader
import os
from src.metrics import get_metrics_hf_transformer, get_lm_loss_hf_transformer
from tqdm import tqdm
import json
import argparse
import glob
import logging


def write_stats_to_file(model, out_dir, epoch, train_dataloader=None, device="cuda"):
    data = get_metrics_hf_transformer(model)

    # make directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logging.info("Writing stats to %s", os.path.join(out_dir, f"stats_{epoch}_losses"))
    with open(os.path.join(out_dir, f"stats_{epoch}_losses"), "w") as f:
        json.dump(data, f)


def main(out_dir, model_dir=None):
    checkpoints = glob.glob(model_dir + "*/")

    for d in tqdm(checkpoints):
        model = BertForSequenceClassification.from_pretrained(d)

        epoch = int(d.split("-")[-1][:-1])

        print(epoch)
        write_stats_to_file(model, out_dir, epoch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, help="Path to huggingface transformer directory"
    )
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir, args.model_dir)
