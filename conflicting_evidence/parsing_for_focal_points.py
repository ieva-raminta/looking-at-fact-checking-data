import json
import spacy

nlp = spacy.load("en_core_web_sm")


def flatten(l):
    return [item for sublist in l for item in sublist]


f = open("nat_claims_dev.jsonl")
_items = list(f)


def find_subtrees(sentence): 
    subtrees = []
    tokens = nlp(sentence)
    for token in tokens:
        if token.pos_ in ["VERB", "ADJ", "ADP", "PRON", "NOUN", "CCONJ"]:
            if token.dep_ in [
                "attr",
                "acomp",
                "amod",
                "prep",
                "agent",
                "compound",
            ] and (
                token.head.lemma_ not in ["be", "same"]
            ):  # and token.head.lemma_ == 'be':
                subtrees.append([t.text for t in token.subtree])
            if (
                token.pos_ == "PRON"
                and token.dep_ in ["nsubj", "dobj"]
                and token.head.dep_ in ["relcl"]
            ):
                subtrees.append([t.text for t in token.head.subtree])
            if token.pos_ == "CCONJ":
                next_token = token.i + 1
                subtrees.append(
                    [t.text for t in token.subtree]
                    + [t.text for t in [token for token in tokens][next_token].subtree]
                )
    return subtrees

for item in _items:
    result = json.loads(item)
    claim = result["claim"]
    tokens = nlp(claim)
    claim_subtrees = find_subtrees(claim)

    sentence_annotations = [{r[-1]: r[0]} for r in result["annotations"].values()]
    block_annotations = [b["label"] for b in result["block_annotations"]["0"]]
    entity = result["entity"]
    for sentence_annotation in sentence_annotations:
        sentence = [s for s in sentence_annotation.keys()][0]
        sentence_subtrees = find_subtrees(sentence)
        
