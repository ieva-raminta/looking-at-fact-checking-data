import json
import os

print(os.getcwd())


f = open(
    "/home/irs38/looking-at-fact-checking-data/chaosNLI/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"
)
_items = list(f)

conflicting_evidence_items = []
more_than_two_labels_counter = 0

for item in _items:
    result = json.loads(item)
    two_labels_exist = True if len(result["label_counter"]) == 2 else False
    if two_labels_exist:
        conflicting_evidence = (
            True
            if all(
                [
                    result["label_counter"][key] < 66
                    for key in result["label_counter"].keys()
                ]
            )
            else False
        )
        if conflicting_evidence:
            conflicting_evidence_items.append(result)
    if len(result["label_counter"]) > 2:
        more_than_two_labels_counter += 1

print(
    len(conflicting_evidence_items) / len(_items),
    len(conflicting_evidence_items),
    len(_items),
)
print(more_than_two_labels_counter)
