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
from datasets import load_dataset

metric = evaluate.load("accuracy")
training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)

LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2, "hidden": 0}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data.
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data


def load_qasc_data(path):
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            data.append(loaded_example)
    return data


def preprocess_qasc(loaded_qasc):
    for item in loaded_qasc:
        item["statement"] = item["question"].replace()


def load_all_wikipedia():
    pages = []
    for file in os.listdir("wiki-pages"):
        filename = os.fsdecode(file)
        with open("wiki-pages/" + filename) as f:
            for lineid, line in enumerate(f):
                if lineid != 0:
                    loaded_page = json.loads(line)
                    pages.append(loaded_page)
    df = pd.DataFrame.from_dict(pages)
    return df


def flatten(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def load_fever_data(path):
    wiki = load_all_wikipedia()
    fever_examples = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            flat_evidence = flatten(loaded_example["evidence"])
            page_ids = [flat_evidence[x][2] for x in range(len(flat_evidence))]
            sentence_ids = [flat_evidence[x][3] for x in range(len(flat_evidence))]
            loaded_example["sentences"] = []
            for i in range(len(page_ids)):
                page_id = page_ids[i]
                sentence_id = sentence_ids[i]
                page_row = wiki.loc[wiki["id"] == page_id]
                if not page_row.empty:

                    page_lines = page_row["lines"]
                    page_line = page_lines[list(page_lines.keys())[0]].split("\t")[
                        sentence_id
                    ]
                    loaded_example["sentences"].append(page_line)
    fever_examples.append(loaded_example)


tokenizer = AlbertTokenizer.from_pretrained(
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
)
model = AlbertForSequenceClassification.from_pretrained(
    "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
)
text = "This is the first sentence"
text_pair = "This is the second sentence"
encoded_input = tokenizer(text=text, text_pair=text_pair, return_tensors="pt")
print(encoded_input)
output = model(**encoded_input)
print(output)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

# loaded_mnli = load_nli_data("multinli_1.0/multinli_1.0_dev_matched.jsonl")
loaded_mnli = load_dataset("multi_nli")
# loaded_qasc = load_qasc_data("qasc/dev.jsonl")
# loaded_fever = load_fever_data("fever_train.jsonl")

# df_mnli = pd.DataFrame(loaded_mnli)
# df_qasc = pd.DataFrame(loaded_qasc)
# df_fever = pd.DataFrame(loaded_fever)

pdb.set_trace()
# mnli_for_train = [{"label": item["label"], "text": item["sentence1"], "text_pair":item["sentence2"]} for item in loaded_mnli]


def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], padding="max_length")


tokenized_mnli = loaded_mnli.map(
    tokenize_function, batched=True
)  # [tokenizer(item["text"], item["text_pair"], padding="max_length") for item in mnli_for_train]

pdb.set_trace()

small_train_dataset = tokenized_mnli["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = (
    tokenized_mnli["validation_matched"].shuffle(seed=42).select(range(1000))
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("test_trainer")

pdb.set_trace()
