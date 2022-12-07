import json
import pdb
from collections import Counter

def flatten(l):
    return [item for sublist in l for item in sublist]

f = open("nat_claims_dev.jsonl")
_items = list(f)

conflicting_evidence_items = []
conflicting_evidence_within_one_sentence_items = []

for item in _items: 
    result = json.loads(item)
    claim = result["claim"]
    sentence_annotations = [{r[-1]:r[0]} for r in result["annotations"].values()]
    block_annotations = [b["label"] for b in result["block_annotations"]["0"]]
    entity = result["entity"]
    section_labels = []
    section_sentences = []
    refuting_sentences = []
    supporting_sentences = []
    supporting_sentences_label_distribution = []
    refuting_sentences_label_distribution = []
    for sentence_annotation in sentence_annotations: 
        if len(set(list(sentence_annotations[0].values())[0])) > 1:
            conflicting_evidence_within_one_sentence_items.append(result)
            pdb.set_trace()
        list_of_labels = [s for s in sentence_annotation.values()][0]
        if list_of_labels: 
            most_common_label_counter = Counter(list_of_labels)

            most_common_label = most_common_label_counter.most_common(1)[0][0]
            section_labels.append(most_common_label)
            sentence = [s for s in sentence_annotation.keys()][0]
            section_sentences.append(sentence)
            if most_common_label == 1: 
                supporting_sentences.append(sentence)
                supporting_sentences_label_distribution.append(list_of_labels)
            elif most_common_label == -1: 
                refuting_sentences.append(sentence)
                refuting_sentences_label_distribution.append(list_of_labels)

    if 1 in section_labels and -1 in section_labels:
        conflicting_evidence_items.append(result)
        #pdb.set_trace()



print(len(conflicting_evidence_items))
print(len(_items))
print(len(conflicting_evidence_items)/len(_items))

