import json
import os

print(os.getcwd())


f = open(
    "/home/irs38/looking-at-fact-checking-data/chaosNLI/chaosNLI_v1.0/chaosNLI_mnli_m.jsonl"
)
_items = list(f)

all_item_ids = []
for item in _items:
    result = json.loads(item)

    all_item_ids.append(result["uid"])
    print()


f1 = open(
    "/home/irs38/looking-at-fact-checking-data/chaosNLI/mnli_random_baseline.json"
)

_items = list(f1)
test_item_ids = []

for item in _items:
    result = json.loads(item)
    test_item_ids = [key for key in result["random_baseline"].keys()]

print(len(all_item_ids))
print(len(test_item_ids))
print([i for i in all_item_ids if i not in test_item_ids])
