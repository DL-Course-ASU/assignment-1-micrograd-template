"""Recreate xor.csv using only the standard library."""
import csv
import random
from pathlib import Path

rng = random.Random(23)
with Path(__file__).with_name('xor.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x1', 'x2', 'target', 'split'])
    for split, count in [('train', 12), ('test', 6)]:
        for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            for _ in range(count):
                writer.writerow([round(a + rng.gauss(0, 0.2), 6),
                                 round(b + rng.gauss(0, 0.2), 6), -a * b, split])
