# Compute CKA on a single pair of models
import argparse
import datasets
import json
import logging
import numpy as np
import os
import pprint
import torch
from torch_cka import CKA
from transformers import (
    AutoModelForMaskedLM,
)


def calculate_cka_pair(model1, model1_name, model2, model2_name, dataloader):
    cka = CKA(
        model1, model2, model1_name=model1_name, model2_name=model2_name, device="cuda"
    )
    cka.compare(dataloader, only_compare_diagonals=True)
    cka_out = cka.export()
    # make values serializable
    cka_out["CKA"] = cka_out["CKA"].cpu().numpy().tolist()
    return cka_out


def parse_args(input_args):
    parser = argparse.ArgumentParser(
        description="Load all checkpoints for an MLM training run and compute CKA between all pairs."
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Filepath to first model to compare.",
    )
    parser.add_argument(
        "--model1-name",
        type=str,
        help="Name of first model. Will be used in output filename.",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Filepath to second model to compare.",
    )
    parser.add_argument(
        "--model2-name",
        type=str,
        help="Name of second model. Will be used in output filename.",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
    )
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--output-filename-prefix",
        type=str,
        default="cka",
        help="Prefix of output filename. Output filepath is {args.output_dir}/{args.output_filename_prefix}_{args.model1_name}_{args.model2_name}.json",
    )
    parser.add_argument(
        "--pretraining-data-dir",
        help="Directory containing Wikipedia pre-training data.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=2048)
    args = parser.parse_args(input_args)
    logging.basicConfig(level=args.loglevel.upper())
    argsdict = vars(args)
    print(pprint.pformat(argsdict))
    return args


def main():
    args = parse_args(None)
    output_fp = os.path.join(
        args.output_dir,
        f"{args.output_filename_prefix}_{args.model1_name}_{args.model2_name}.json",
    )
    if os.path.exists(output_fp):
        logging.error(f"{output_fp} already exists. Returning.")
        return
    dataset = datasets.load_from_disk(args.pretraining_data_dir)
    dataset = dataset.remove_columns(["token_type_ids", "text"])
    subsample = datasets.Dataset.from_dict(dataset["train"][: args.sample_size]).map(
        lambda ex: {
            "input_ids": ex["input_ids"]
            + [0 for _ in range(args.max_sequence_length - len(ex["input_ids"]))]
        },
        desc="Padding input IDs to fixed length",
        keep_in_memory=True,
    )
    subsample.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = torch.utils.data.DataLoader(
        subsample["input_ids"], batch_size=args.batch_size, drop_last=True
    )
    model1 = AutoModelForMaskedLM.from_pretrained(args.model1).to("cuda")
    model2 = AutoModelForMaskedLM.from_pretrained(args.model2).to("cuda")
    cka_dict = calculate_cka_pair(
        model1, args.model1_name, model2, args.model2_name, dataloader
    )
    with open(output_fp, "w") as f:
        f.write(json.dumps(cka_dict))


if __name__ == "__main__":
    main()
