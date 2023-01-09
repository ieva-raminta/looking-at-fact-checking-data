from transformers import (
    AlbertTokenizer,
    AlbertModel,
    TrainingArguments,
    Trainer,
    AlbertForSequenceClassification,
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
from datasets import Dataset
import random

label_map = {-1: 0, 0: 1, 1: 2}

OUTPUT_DIR = "output_bart_natural"


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


metric = evaluate.load("accuracy")
training_args = TrainingArguments(
    output_dir="rds/hpc-work/trained_bart_on_nat_claims_dir",
    evaluation_strategy="epoch",
    num_train_epochs=20,
    per_device_train_batch_size=4,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")



def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length")


tokenized_nat_dev = nat_dev_dataset.map(tokenize_function, batched=True)
tokenized_nat_train = nat_train_dataset.map(tokenize_function, batched=True)


def compute_test_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    return metric.compute(predictions=predictions, references=labels)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_nat_train,
    eval_dataset=tokenized_nat_dev,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("trained_bart_on_nat_claims")

test_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 1,
    dataloader_drop_last = False,
    eval_accumulation_steps=1,
)



trainer = Trainer(
    model=model,
    args=test_args,
    eval_dataset=tokenized_nat_dev,
    compute_metrics=compute_test_metrics,
)

evaluation = trainer.predict(tokenized_nat)
print(evaluation)



