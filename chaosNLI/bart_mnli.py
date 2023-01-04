import pdb

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

premise_text = "Children can go on the ride for free."
hypothesis_text = "People can go on the ride for free."

inputs = tokenizer(premise_text, hypothesis_text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits


predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

print(
    premise_text,
    "\n",
    hypothesis_text,
    "\n",
    model.config.id2label[predicted_class_id],
    "\n",
    "Q" * 50,
)

# pdb.set_trace()
