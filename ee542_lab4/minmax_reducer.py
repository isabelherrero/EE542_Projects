#!/usr/bin/env python3
import sys

min_word, min_count = None, float('inf')
max_word, max_count = None, float('-inf')

for line in sys.stdin:
    word, val = line.strip().split('\t')
    count = int(val)

    if count < min_count:
        min_word, min_count = word, count

    if count > max_count:
        max_word, max_count = word, count

print(f"MIN\t{min_word}\t{min_count}")
print(f"MAX\t{max_word}\t{max_count}")
