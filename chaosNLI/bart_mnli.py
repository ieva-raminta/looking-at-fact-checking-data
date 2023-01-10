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
from datetime import date

#mnli_labels_to_nat = {0: -1, 1: 0, 2: 1}
nat_labels_to_mnli = {-1: 0, 0: 1, 1: 2}

OUTPUT_DIR = "rds/hpc-work/output_bart_mnli"

if os.path.exists(OUTPUT_DIR): 
    OUTPUT_DIR += str(date.today())

f = open("rds/hpc-work/nat_claims_dev.jsonl")
nat_claims_dev_items = list(f)

nat_dataset = []
for item in nat_claims_dev_items:
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
            label = nat_labels_to_mnli[most_common_label]

            nat_dataset.append(
                {"premise": premise, "hypothesis": hypothesis, "label": label}
            )
nat_dataset = Dataset.from_list(nat_dataset)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    with torch.no_grad(): 
        logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
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


tokenized_nat = nat_dataset.map(tokenize_function, batched=True)

test_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 16,
    dataloader_drop_last = False,
    eval_accumulation_steps=1,
    )

trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=compute_metrics,
)

evaluation = trainer.predict(tokenized_nat)
print(evaluation)



