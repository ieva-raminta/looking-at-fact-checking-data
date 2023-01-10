from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import pdb
import json
import random
import pandas as pd
import os
import numpy as np
import evaluate
from datasets import (
    load_dataset,
    load_from_disk,
    concatenate_datasets,
    ClassLabel,
    Value,
)
from sklearn.metrics import f1_score
from collections import Counter

input_file = "rds/hpc-work/parsed_natural_train.jsonl"
output_file = "rds/hpc-work/parsed_natural_train_evaluated_with_bart_trained_on_mnli_no_majority.jsonl"

# input_file = "rds/hpc-work/parsed_chaosnli.json"
# output_file = "rds/hpc-work/parsed_chaosnli_evaluated_with_bart_trained_on_mnli_no_majority.json"

with open(input_file, "r") as read_file:
    parsed_items = json.load(read_file)

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

mnli_labels_to_nli = {0: "c", 1: "n", 2: "e"}

counter = 0
parsed_dataset = []
for result in parsed_items:

    counter += 1
    edited_dataset = {}
    edited_dataset["score"] = {}
    edited_dataset["original_score"] = {}
    edited_item = None
    edited_dataset["edited_item"] = {}
    edited_dataset["original_item"] = {}

    premise = result["example"]["premise"]
    hypothesis = result["example"]["hypothesis"]

    if "c" in result["label_counter"]:
        og_score = result["label_counter"]["c"] / 100
        edited_dataset["original_score"]["c"] = og_score
        edited_dataset["score"]["c"] = og_score
        og_item = (
            premise,
            hypothesis,
            "c",
            None,
            "",
            "original",
        )
        edited_dataset["edited_item"]["c"] = og_item
        edited_dataset["original_item"]["c"] = og_item
    if "e" in result["label_counter"]:
        og_score = result["label_counter"]["e"] / 100
        edited_dataset["original_score"]["e"] = og_score
        edited_dataset["score"]["e"] = og_score
        og_item = (
            premise,
            hypothesis,
            "e",
            None,
            "",
            "original",
        )
        edited_dataset["edited_item"]["e"] = og_item
        edited_dataset["original_item"]["e"] = og_item

    if "n" in result["label_counter"]:
        og_score = result["label_counter"]["n"] / 100
        edited_dataset["original_score"]["n"] = og_score
        edited_dataset["score"]["n"] = og_score
        og_item = (
            premise,
            hypothesis,
            "n",
            None,
            "",
            "original",
        )
        edited_dataset["edited_item"]["n"] = og_item
        edited_dataset["original_item"]["n"] = og_item

    if "subtrees_from_premise" in result.keys():
        subtrees_from_premise = result["subtrees_from_premise"]
        subtrees_from_hypothesis = result["subtrees_from_hypothesis"]

        cropped_premises = result["cropped_premises"]
        cropped_hypotheses = result["cropped_hypotheses"]

        for cropped_id, cropped_premise in enumerate(cropped_premises):
            tokenized_premhyp = tokenizer(
                cropped_premise, hypothesis, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(**tokenized_premhyp).logits
            confidence = logits.softmax(-1).max().item()
            predicted_class_id = logits.argmax().item()
            predicted_label = mnli_labels_to_nli[predicted_class_id]
            if (
                predicted_label in edited_dataset["score"]
                and confidence > edited_dataset["score"][predicted_label]
            ):
                edit = premise[
                    subtrees_from_premise[cropped_id][0] : subtrees_from_premise[
                        cropped_id
                    ][1]
                ]
                edited_item = (
                    cropped_premise,
                    hypothesis,
                    predicted_label,
                    (
                        subtrees_from_premise[cropped_id][0],
                        subtrees_from_premise[cropped_id][1],
                    ),
                    edit,
                    "cropped_premise",
                )
                edited_dataset["score"][predicted_label] = confidence
                edited_dataset["edited_item"][predicted_label] = edited_item

        for cropped_id, cropped_hypothesis in enumerate(cropped_hypotheses):
            tokenized_premhyp = tokenizer(
                premise, cropped_hypothesis, return_tensors="pt"
            )
            with torch.no_grad():
                logits = model(**tokenized_premhyp).logits
            confidence = logits.softmax(-1).max().item()
            predicted_class_id = logits.argmax().item()
            predicted_label = mnli_labels_to_nli[predicted_class_id]
            if (
                predicted_label in edited_dataset["score"]
                and confidence > edited_dataset["score"][predicted_label]
            ):
                edit = hypothesis[
                    subtrees_from_hypothesis[cropped_id][0] : subtrees_from_hypothesis[
                        cropped_id
                    ][1]
                ]
                edited_item = (
                    premise,
                    cropped_hypothesis,
                    predicted_label,
                    (
                        subtrees_from_hypothesis[cropped_id][0],
                        subtrees_from_hypothesis[cropped_id][1],
                    ),
                    edit,
                    "cropped_hypothesis",
                )
                edited_dataset["score"][predicted_label] = confidence
                edited_dataset["edited_item"][predicted_label] = edited_item

    parsed_dataset.append(edited_dataset)
    print(counter, len(parsed_items))


with open(output_file, "w") as outfile:
    json.dump(parsed_dataset, outfile)
