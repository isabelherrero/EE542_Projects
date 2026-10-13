#!/usr/bin/env python3
import sys
from collections import defaultdict

counts = defaultdict(int)

for line in sys.stdin:
    word, val = line.strip().split('\t')
    counts[word] += int(val)

for word, count in counts.items():
    print(f"{word}\t{count}")
