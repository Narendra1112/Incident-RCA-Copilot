import json
from collections import defaultdict
from datetime import datetime, timedelta

def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

groups = defaultdict(list)

with open("app/data/raw/logs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        key = (row["service"], row["component"])
        groups[key].append(row)

for key, logs in groups.items():
    logs.sort(key=lambda x: parse_ts(x["timestamp"]))
    best = 0
    left = 0
    for right in range(len(logs)):
        while parse_ts(logs[right]["timestamp"]) - parse_ts(logs[left]["timestamp"]) > timedelta(minutes=3):
            left += 1
        best = max(best, right - left + 1)
    print(key, "count=", len(logs), "max_within_3min=", best)