import json

f = open("GenericsKB-Waterloo-With-Context.jsonl")
_items = list(f)

for item in _items:
    result = json.loads(item)
    following_sentence = result["context"]["sentences_after"][0].lower()
    following_sentence_starts_with_causal_connective = (
        following_sentence.startswith("therefore")
        or following_sentence.startswith("as a result")
        or following_sentence.startswith("this is why")
        or following_sentence.startswith("because of this")
        or following_sentence.startswith("because of it")
        or following_sentence.startswith("for this reason")
    )
