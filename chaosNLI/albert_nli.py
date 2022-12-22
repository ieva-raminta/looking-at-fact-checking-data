from transformers import (
    AlbertTokenizer,
    AlbertModel,
    TrainingArguments,
    Trainer,
    AlbertForSequenceClassification,
)
import torch
import pdb
import json
import random
import pandas as pd
import os
import numpy as np
import evaluate
from datasets import load_dataset, load_from_disk, concatenate_datasets, ClassLabel, Value

metric = evaluate.load("accuracy")
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    num_train_epochs=30,
)

LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2, "hidden": 0}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AlbertTokenizer.from_pretrained(
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
)
model = AlbertForSequenceClassification.from_pretrained(
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

loaded_mnli = load_dataset("multi_nli")
loaded_fever = load_dataset("fever", "v1.0")
loaded_wiki_pages = load_dataset("fever", "wiki_pages")

loaded_fever.save_to_disk("fever_with_wiki")

loaded_fever.rename_column("claim", "hypothesis")


def include_wiki_evidence(example):
    try:
        page_index = loaded_wiki_pages["wikipedia_pages"]["id"].index(
        example["evidence_wiki_url"]
        )
    except: 
        page_index = False
    if page_index: 
        page_lines = loaded_wiki_pages["wikipedia_pages"]["lines"][page_index].splitlines()
        if len(page_lines) == 0:
            example["premise"] = ""
        elif example["evidence_sentence_id"] < len(page_lines):
            example["premise"] = page_lines[example["evidence_sentence_id"]]
        else:
            example["premise"] = page_lines[-1]
    else: 
        example["premise"] = ""

    return example


def map_labels(example):
    the_map = {"NOT ENOUGH INFO": 1, "SUPPORTS": 0, "REFUTES": 2}
    if example["label"] not in the_map.keys():
        example["label"] = 1
    else:
        example["label"] = the_map[example["label"]]
    return example


loaded_fever = loaded_fever.map(map_labels)
loaded_fever_with_wiki = loaded_fever.map(include_wiki_evidence)
loaded_fever_with_wiki = loaded_fever_with_wiki.remove_columns(
    [
        "id",
        "evidence_annotation_id",
        "evidence_id",
        "evidence_wiki_url",
        "evidence_sentence_id",
    ]
)
loaded_mnli = loaded_mnli.remove_columns(
    [
        "promptID",
        "pairID",
        "premise_binary_parse",
        "premise_parse",
        "hypothesis_binary_parse",
        "hypothesis_parse",
        "genre",
    ]
)

assert loaded_mnli.features.type == loaded_fever_with_wiki.features.type
nli_dataset = concatenate_datasets([loaded_mnli, loaded_fever_with_wiki])


def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length")


tokenized_nli = nli_dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_nli["train"].shuffle(seed=42)
eval_dataset = tokenized_nli["validation_matched"].shuffle(seed=42)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("test_trainer")

pdb.set_trace()
