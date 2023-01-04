import json
import pdb

# f = open("cskb-waterloo-06-21-with-bert-scores.jsonl")
f = open("GenericsKB-SimpleWiki-With-Context.jsonl")
_items = list(f)

causal_counter = 0
total_counter = 0

for item in _items:
    result = json.loads(item)
    total_counter += 1
    if result["knowledge"]["context"]["sentences_after"]:
        following_sentence = result["knowledge"]["context"]["sentences_after"][
            0
        ].lower()
        following_sentence_starts_with_causal_connective = (
            following_sentence.startswith("therefore")
            or following_sentence.startswith("as a result")
            or following_sentence.startswith("this is why")
            or following_sentence.startswith("because of this")
            or following_sentence.startswith("because of it")
            or following_sentence.startswith("for this reason")
            or following_sentence.startswith("on account of this")
            or following_sentence.startswith("consequently")
        )
        if following_sentence_starts_with_causal_connective:
            causal_counter += 1
            if "murder" in result["knowledge"]["sentence"]:
                pdb.set_trace()
            print(total_counter)
            print(causal_counter)
            print(result["knowledge"]["sentence"])
            print(following_sentence)
            print("========================================================")
