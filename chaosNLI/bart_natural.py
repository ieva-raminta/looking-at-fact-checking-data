from transformers import (
    AlbertTokenizer,
    AlbertModel,
    TrainingArguments,
    Trainer,
    AlbertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    IntervalStrategy,
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
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from collections import Counter
from datasets import Dataset
import random
from tkinter import Tcl
from datetime import date

label_map = {-1: 0, 0: 1, 1: 2}

OUTPUT_DIR = "rds/hpc-work/output_bart_natural"


def load_natural_datasets(filename):
    f = open(filename)
    nat_claims_items = list(f)
    nat_dataset = []
    for item in nat_claims_items:
        result = json.loads(item)
        claim = result["claim"]
        sentence_annotations = [{r[-1]: r[0]} for r in result["annotations"].values()]
        for sentence_annotation in sentence_annotations:
            list_of_labels = [s for s in sentence_annotation.values()][0]
            evidence = [i for i in sentence_annotation.keys()][0]
            if list_of_labels:
                most_common_label_counter = Counter(list_of_labels)
                most_common_label = most_common_label_counter.most_common(1)[0][0]

                premise = evidence
                hypothesis = claim
                label = label_map[most_common_label]

                nat_dataset.append(
                    {"premise": premise, "hypothesis": hypothesis, "label": label}
                )
    random.shuffle(nat_dataset)
    nat_Dataset = Dataset.from_list(nat_dataset)
    return nat_Dataset


nat_dev_dataset = load_natural_datasets("rds/hpc-work/nat_claims_dev.jsonl")
nat_train_dataset = load_natural_datasets("rds/hpc-work/nat_claims_train.jsonl")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy=IntervalStrategy.STEPS,
    eval_steps=500,
    save_steps=500,
    save_total_limit=5,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=20,
    weight_decay=0.01,
    push_to_hub=False,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    gradient_accumulation_steps=1,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base-mnli")

try:  # os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) != 0:
    file_list = os.listdir(OUTPUT_DIR)
    sorted_file_list = Tcl().call("lsort", "-dict", file_list)
    latest_checkpoint = sorted_file_list[-1]
    model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR + "/" + latest_checkpoint
    )
except:
    model = AutoModelForSequenceClassification.from_pretrained(
        "facebook/bart-base-mnli"
    )

for param in model.base_model.parameters():
    param.requires_grad = False


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length")


tokenized_nat_dev = nat_dev_dataset.map(tokenize_function)
tokenized_nat_train = nat_train_dataset.map(tokenize_function)

del nat_dev_dataset
del nat_train_dataset

tokenized_nat_dev = tokenized_nat_dev.remove_columns(["premise", "hypothesis"])
tokenized_nat_train = tokenized_nat_train.remove_columns(["premise", "hypothesis"])


def compute_test_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_nat_train,
    eval_dataset=tokenized_nat_dev,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

test_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=1,
    dataloader_drop_last=False,
    eval_accumulation_steps=1,
)


trainer = Trainer(
    model=model,
    args=test_args,
    eval_dataset=tokenized_nat_dev,
    compute_metrics=compute_test_metrics,
)

evaluation = trainer.predict(tokenized_nat_dev)

print(evaluation)
