import json
import pdb
import os
from collections import Counter

from parsing_for_focal_points import find_subtrees


conflicting_evidence_items = []
multiple_evidence = 0

new_results = []


f = open("nat_claims_train.jsonl")
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
            label = most_common_label


            two_labels_exist = True if len(set(list_of_labels)) > 1 else False
            if two_labels_exist:
    

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

with open("parsed_natural_train.json", "w") as fout:
    json.dump(new_results, fout)

