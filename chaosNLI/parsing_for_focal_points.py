import json
import spacy

nlp = spacy.load("en_core_web_lg")


def flatten(l):
    return [item for sublist in l for item in sublist]


# f = open("nat_claims_dev.jsonl")
# _items = list(f)


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
            ]:
                token_start_index = [t.idx for t in token.subtree]
                token_end_index = [t.idx + len(t) for t in token.subtree]
                subtrees.append((min(token_start_index), max(token_end_index)))
                print()
            if token.dep_ in ["relcl", "advcl", "ccomp"]:
                token_start_index = [t.idx for t in token.subtree]
                token_end_index = [t.idx + len(t) for t in token.subtree]
                subtrees.append((min(token_start_index), max(token_end_index)))
                print()
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
                "speculate",
            ]:  # the list based on commitmentbank
                embedded_verb = [
                    child for child in token.children if child.pos_ in ["VERB", "AUX"]
                ]
                if embedded_verb:
                    embedded_clause = [t for t in embedded_verb[0].subtree]
                    full_clause = [t for t in token.subtree]
                    token_start_index = [
                        t.idx
                        for t in tokens
                        if t in full_clause and t not in embedded_clause
                    ]
                    token_end_index = [
                        t.idx + len(t)
                        for t in tokens
                        if t in full_clause and t not in embedded_clause
                    ]
                    subtrees.append((min(token_start_index), max(token_end_index)))
                    print()
            if token.pos_ in ["CCONJ"]:
                conjuncts = [t for t in tokens if t.dep_ == "conj"]
                for conjunct in conjuncts:
                    if token.head == conjunct.head:
                        token_start_index = [t.idx for t in token.subtree] + [
                            t.idx for t in conjunct.subtree
                        ]
                        token_end_index = [t.idx + len(t) for t in token.subtree] + [
                            t.idx + len(t) for t in conjunct.subtree
                        ]
                        subtrees.append((min(token_start_index), max(token_end_index)))
                        print()

    return subtrees


"""
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
"""
