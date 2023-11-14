import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    default_data_collator,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score

import logging
import argparse
import glob
import pandas
from tqdm import tqdm


def evaluate(model, dataloader, device):
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, batch_predicted_labels = torch.max(logits, dim=1)

            predicted_labels.extend(batch_predicted_labels.detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

    # Calculate accuracy
    return accuracy_score(true_labels, predicted_labels)


def hans_eval(model_name, lexical_overlap, subsequence, constituent, cpu=False):
    logging.info(f"Model name: {model_name}")
    device = torch.device("cpu") if cpu else torch.device("cuda")

    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    lexical_overlap_accuracy = evaluate(model, lexical_overlap, device)
    logging.info(f"Lexical overlap accuracy: {lexical_overlap_accuracy}")
    subsequence_accuracy = evaluate(model, subsequence, device)
    logging.info(f"Subsequence accuracy: {subsequence_accuracy}")
    constituent_accuracy = evaluate(model, constituent, device)
    logging.info(f"Constituent accuracy: {constituent_accuracy}")

    return lexical_overlap_accuracy, subsequence_accuracy, constituent_accuracy


def main():
    bsz = 128
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="./data/raw/GLUE-output/MNLI/seed_0"
    )
    parser.add_argument("--save_dir", type=str, default="./data/evals/")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset("hans")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Tokenize and preprocess the dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            add_special_tokens=True,
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

    dataset = dataset.map(preprocess_function, batched=True)["validation"]
    lexical_overlap = dataset.filter(
        lambda example: example["heuristic"] == "lexical_overlap"
    )
    subsequence = dataset.filter(lambda example: example["heuristic"] == "subsequence")
    constituent = dataset.filter(lambda example: example["heuristic"] == "constituent")
    data_collator = default_data_collator

    lexical_dataloader = DataLoader(
        lexical_overlap, batch_size=bsz, collate_fn=data_collator
    )
    subsequence_dataloader = DataLoader(
        subsequence, batch_size=bsz, collate_fn=data_collator
    )
    constituent_dataloader = DataLoader(
        constituent, batch_size=bsz, collate_fn=data_collator
    )

    steps = []
    lexical_overlap_accuracies = []
    subsequence_accuracies = []
    constituent_accuracies = []

    for model_name in tqdm(glob.glob(args.model_dir + "/*")):
        # steps should be the last part of the model name via huggingface
        steps.append(model_name.split("-")[-1])
        lexical, subsequence, constituent = hans_eval(
            model_name,
            lexical_dataloader,
            subsequence_dataloader,
            constituent_dataloader,
            args.cpu,
        )
        lexical_overlap_accuracies.append(lexical)
        subsequence_accuracies.append(subsequence)
        constituent_accuracies.append(constituent)

    df = pandas.DataFrame(
        {
            "seed": steps,
            "lexical_overlap": lexical_overlap_accuracies,
            "subsequence": subsequence_accuracies,
            "constituent": constituent_accuracies,
        }
    )
    df.to_csv(args.save_dir + "hans_eval.csv", index=False)


if __name__ == "__main__":
    main()
