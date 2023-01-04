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

        highest_confidence = {} 
        highest_confidence["score"] = {}
        edited_item = None
        highest_confidence["edited_item"] = {}

        premise = result["example"]["premise"]
        hypothesis = result["example"]["hypothesis"]
        majority_label = result["majority_label"]
        idx = result["uid"]

        contradiction_percentage = 0
        entailment_percentage = 0
        neutral_percentage = 0
        if "c" in result["label_counter"]: 
            contradiction_percentage = result["label_counter"]["c"]
            highest_confidence["score"]["c"] = 0
        if "e" in result["label_counter"]: 
            entailment_percentage = result["label_counter"]["e"]
            highest_confidence["score"]["e"] = 0
        if "n" in result["label_counter"]:
            neutral_counter = result["label_counter"]["n"]
            highest_confidence["score"]["n"] = 0
    
        majority_label_confidence = result["label_counter"][majority_label]
        highest_confidence["score"][majority_label] = 1
        highest_confidence["edited_item"][majority_label] = (premise, hypothesis, majority_label_confidence, "", "original")

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
                if predicted_label in highest_confidence["score"] and confidence > highest_confidence["score"][predicted_label]: 
                    edit = premise[subtrees_from_premise[cropped_id][0]:subtrees_from_premise[cropped_id][1]]
                    edited_item = (cropped_premise, hypothesis, predicted_label, edit, "cropped premise")
                    highest_confidence["score"][predicted_label] = confidence
                    highest_confidence["edited_item"][predicted_label] = edited_item


            for cropped_id, cropped_hypothesis in enumerate(cropped_hypotheses):
                tokenized_premhyp = tokenizer(premise, cropped_hypothesis, return_tensors="pt")
                with torch.no_grad():
                    logits = model(**tokenized_premhyp).logits
                confidence = logits.softmax.max().item()
                predicted_class_id = logits.argmax().item()
                predicted_label = mnli_labels_to_nli[predicted_class_id]
                if predicted_label in highest_confidence["score"] and confidence > highest_confidence["score"][predicted_label]:
                    edit = hypothesis[subtrees_from_premise[cropped_id][0]:subtrees_from_hypothesis[cropped_id][1]]
                    edited_item = (premise, cropped_hypothesis, predicted_label, edit, "cropped_hypothesis")
                    highest_confidence["score"][predicted_label] = confidence
                    highest_confidence["edited_item"][predicted_label] = edited_item

        pdb.set_trace()
        parsed_dataset.append(highest_confidence)





