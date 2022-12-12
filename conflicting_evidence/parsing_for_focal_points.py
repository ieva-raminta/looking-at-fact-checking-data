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
        if token.pos_ in [
            "VERB",
            "ADJ",
            "ADP",
            "PRON",
            "NOUN",
            "CCONJ",
            "SCONJ",
            "VERB",
            "AUX",
        ]:
            if token.dep_ in [
                "attr",
                "acomp",
                "amod",
                "prep",
                "agent",
                "compound",
            ] and (token.head.lemma_ not in ["be", "same"]):
                subtrees.append([t for t in token.subtree])
            if token.dep_ in ["relcl", "advcl"]:
                subtrees.append([t for t in token.subtree])
            if token.lemma_ in [
                "think",
                "believe",
                "tell",
                "suspect",
                "guess",
                "hear",
                "hope",
                "assume",
                "bet",
                "fear",
                "expect",
                "pretend",
                "imagine",
                "seem",
                "say",
                "signal",
                "demand",
                "feel",
                "insist",
            ]:  # the list based on commitmentbank
                main_verb = [
                    child for child in token.children if child.pos_ in ["VERB", "AUX"]
                ]
                if main_verb:
                    embedded_clause = [t for t in main_verb[0].subtree]
                    full_clause = [t for t in token.subtree]
                    hedge_clause = [
                        token
                        for token in tokens
                        if token in full_clause and token not in embedded_clause
                    ]

                    subtrees.append(hedge_clause)
            if token.pos_ in ["CCONJ"]:
                conjuncts = [t for t in tokens if t.dep_ == "conj"]
                for conjunct in conjuncts:
                    if token.head == conjunct.head:
                        subtrees.append(
                            [t for t in token.subtree] + [t for t in conjunct.subtree]
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
