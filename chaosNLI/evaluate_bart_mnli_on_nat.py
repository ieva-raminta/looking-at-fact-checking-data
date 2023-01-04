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


f = open("nat_claims_dev.jsonl")
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
            label = most_common_label

            nat_dataset.append(
                {"premise": premise, "hypothesis": hypothesis, "label": label}
            )


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

tokenized_nat_dev = [
    tokenizer(item["premise"], item["hypothesis"], return_tensors="pt")
    for item in nat_dataset
]

true_labels = [i["label"] for i in nat_dataset]
predicted_labels = []

mnli_labels_to_nat = {0:-1, 1:0, 2:1}

for inputid, inputs in enumerate(tokenized_nat_dev):
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]
    predicted_labels.append(mnli_labels_to_nat[predicted_class_id])

f1 = f1_score(np.array(true_labels), np.array(predicted_labels))
print(f1)
pdb.set_trace()


