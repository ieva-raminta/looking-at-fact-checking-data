import json
from collections import Counter


def flatten(l):
    return [item for sublist in l for item in sublist]


f = open("nat_claims_train.jsonl")
_items = list(f)

conflicting_evidence_items = []
conflicting_evidence_within_one_sentence_items = []
sentence_annotations_counter = 0

difficulty_level_0 = []
difficulty_level_1 = []
difficulty_level_2 = []
difficulty_level_3 = []


for item in _items:
    disagreement_within_sentences = False
    disagreement_between_sentences = False
    result = json.loads(item)
    claim = result["claim"]
    sentence_annotations = [{r[-1]: r[0]} for r in result["annotations"].values()]
    block_annotations = flatten(
        [
            [b["label"] for b in result["block_annotations"][key]]
            for key in result["block_annotations"].keys()
        ]
    )
    # I don't understand why there are multiple keys here
    # if len(result["block_annotations"].keys()) > 1:
    #    print("many block annotations")

    entity = result["entity"]
    most_common_sentence_labels = []
    section_sentences = []
    refuting_sentences = []
    supporting_sentences = []
    supporting_sentences_label_distribution = []
    refuting_sentences_label_distribution = []
    for sentence_annotation in sentence_annotations:
        list_of_labels = [s for s in sentence_annotation.values()][0]
        if len(set(list_of_labels)) > 1:
            conflicting_evidence_within_one_sentence_items.append(sentence_annotation)
            disagreement_within_sentences = True
            if not (1 in list_of_labels and -1 in list_of_labels): 
                evidence = [i for i in sentence_annotation.keys()][0]
                print(evidence)
        sentence_annotations_counter += 1
        if list_of_labels:
            most_common_label_counter = Counter(list_of_labels)

            most_common_label = most_common_label_counter.most_common(1)[0][0]
            most_common_sentence_labels.append(most_common_label)
            sentence = [s for s in sentence_annotation.keys()][0]
            section_sentences.append(sentence)
            if most_common_label == 1:
                supporting_sentences.append(sentence)
                supporting_sentences_label_distribution.append(list_of_labels)
            elif most_common_label == -1:
                refuting_sentences.append(sentence)
                refuting_sentences_label_distribution.append(list_of_labels)

    if 1 in most_common_sentence_labels and -1 in most_common_sentence_labels:
        conflicting_evidence_items.append(result)
        disagreement_between_sentences = True

    if not disagreement_within_sentences and not disagreement_between_sentences:
        difficulty_level = 0
        difficulty_level_0.append(result)
    elif not disagreement_within_sentences and disagreement_between_sentences:
        difficulty_level = 1
        difficulty_level_1.append(result)
    elif disagreement_within_sentences and not disagreement_between_sentences:
        difficulty_level = 2
        difficulty_level_2.append(result)
    elif disagreement_within_sentences and disagreement_between_sentences:
        difficulty_level = 3
        difficulty_level_3.append(result)


print(
    len(difficulty_level_0),
    len(difficulty_level_1),
    len(difficulty_level_2),
    len(difficulty_level_3),
)

print(len(conflicting_evidence_items))
print(len(_items))
print(len(conflicting_evidence_items) / len(_items))
print(
    len(conflicting_evidence_within_one_sentence_items),
    sentence_annotations_counter,
    len(conflicting_evidence_within_one_sentence_items) / sentence_annotations_counter,
)
