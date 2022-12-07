import json
import pdb

def flatten(l):
    return [item for sublist in l for item in sublist]

f = open("dev_final_v2.jsonl")
_items = list(f)

conflicting_evidence_items = []

for item in _items: 
    result = json.loads(item)
    claim = result["claim"]
    sections_labels = [{section_key:result["labels"][section_key]} for section_key in result["labels"].keys()]
    sections_texts = [{section_key:result["text"][section_key]} for section_key in result["labels"].keys()]
    section_labels_for_claim = [{list(section.keys())[0]:list(section.values())[0]["section_label"]} for section in sections_labels]
    conflicting_evidence = "supported" in [list(section.values())[0] for section in section_labels_for_claim] and "refuted" in [list(section.values())[0] for section in section_labels_for_claim]
    if conflicting_evidence: 
        conflicting_evidence_items.append(item)

        sentence_labels = [{list(section.keys())[0]:list(section.values())[0]["sentence_labels"]} for section in sections_labels]
        refuting_sentence_ids_dict = {}
        supporting_sentence_ids_dict = {}
        refuting_sentence_ids = [{list(section.keys())[0]: [value_id for value_id,value in enumerate(list(section.values())[0]) if value=="refuting"] } for section in sentence_labels]
        supporting_sentence_ids = [{list(section.keys())[0]: [value_id for value_id,value in enumerate(list(section.values())[0]) if value=="supporting"] } for section in sentence_labels]
        for ref_sent_id in refuting_sentence_ids:
            refuting_sentence_ids_dict.update(ref_sent_id)
        for sup_sent_id in supporting_sentence_ids:
            supporting_sentence_ids_dict.update(sup_sent_id)
                
        sections_texts_dict = {}
        for section_ in sections_texts:
            section_key = list(section_.keys())[0]
            sections_texts_dict[section_key] = list(section_.values())[0]["sentences"]

        refuting_sentences = []
        supporting_sentences = []       
        for key in refuting_sentence_ids_dict: 
            refuting_sentences.extend([sections_texts_dict[key][id_] for id_ in refuting_sentence_ids_dict[key]])
        for key in supporting_sentence_ids_dict: 
            supporting_sentences.extend([sections_texts_dict[key][id_] for id_ in supporting_sentence_ids_dict[key]])


        pdb.set_trace()


print(len(conflicting_evidence_items))
print(len(_items))
print(len(conflicting_evidence_items)/len(_items))

