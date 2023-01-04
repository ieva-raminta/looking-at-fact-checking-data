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


f = open("parsed_chaosnli.json")
parsed_chaosnli_items = list(f)


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

mnli_labels_to_nli = {0:"c", 1:"n", 2:"e"}

parsed_dataset = []
for item in parsed_chaosnli_items:
    results = json.loads(item)

    for result in results:

        highest_confidence = 0
        edited_item = None

        premise = result["example"]["premise"]
        hypothesis = result["example"]["hypothesis"]
        majority_label = result["majority_label"]
        idx = result["uid"]

        contradiction_percentage = 0
        entailment_percentage = 0
        neutral_percentage = 0
        if "c" in result["label_counter"]: 
            contradiction_percentage = result["label_counter"]["c"]
        if "e" in result["label_counter"]: 
            entailment_percentage = result["label_counter"]["e"]
        if "n" in result["label_counter"]:
            neutral_counter = result["label_counter"]["n"]
    
        if "subtrees_from_premise" in result.keys(): 
            subtrees_from_premise = result["subtrees_from_premise"]
            subtrees_from_hypothesis = result["subtrees_from_hypothesis"]

            cropped_premises = result["cropped_premises"]
            cropped_hypotheses = result["cropped_hypotheses"]
    
            for cropped_id, cropped_premise in enumerate(cropped_premises): 
                tokenized_premhyp = tokenizer(cropped_premise, hypothesis, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**tokenized_premhyp).logits
                confidence = logits.max().item()
                predicted_class_id = logits.argmax().item()
                predicted_label = mnli_labels_to_nli[predicted_class_id]
                if confidence > highest_confidence and predicted_label != majority_label: 
                    edit = premise[subtrees_from_premise[cropped_id][0]:] + premise[:subtrees_from_premise[cropped_id][1]]
                    edited_item = (cropped_premise, hypothesis, predicted_label, edit)
                    pdb.set_trace()

            parsed_dataset.append(
                    {"premise": cropped_premise, "hypothesis": cropped_hypotheses[cropped_id], "label": majority_label}
            )



for inputid, inputs in enumerate(tokenized_nat_dev):
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]


