import json
import pdb
import os

from parsing_for_focal_points import find_subtrees

print(os.getcwd())


f = open("rds/hpc-work/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl")
_items = list(f)

conflicting_evidence_items = []
multiple_evidence = 0

new_results = []

for item in _items:
    result = json.loads(item)
    # pdb.set_trace()
    two_labels_exist = True if len(result["label_counter"]) > 1 else False
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
        multiple_evidence += 1
    premise = result["example"]["premise"]
    hypothesis = result["example"]["hypothesis"]

    result["subtrees_from_premise"] = find_subtrees(premise)
    result["subtrees_from_hypothesis"] = find_subtrees(hypothesis)

    result["cropped_premises"] = []
    result["cropped_hypotheses"] = []

    for subtree in result["subtrees_from_premise"]:
        starts = subtree[0]
        ends = subtree[1]
        premise_cropped = premise[:starts] + premise[ends:]
        result["cropped_premises"].append(premise_cropped)

    for subtree in result["subtrees_from_hypothesis"]:
        starts = subtree[0]
        ends = subtree[1]
        hypothesis_cropped = hypothesis[:starts] + hypothesis[ends:]
        result["cropped_hypotheses"].append(hypothesis_cropped)

    new_results.append(result)

with open("rds/hpc-work/parsed_chaosnli.json", "w") as fout:
    json.dump(new_results, fout)

print(multiple_evidence / len(_items))
print(
    len(conflicting_evidence_items) / len(_items),
    len(conflicting_evidence_items),
    len(_items),
)
