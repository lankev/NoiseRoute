"""Fail when normalized-circuit hashes overlap between train, validation and test."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("manifests", nargs="+")
args = p.parse_args()
seen = {}
for name in args.manifests:
    for record in json.loads(Path(name).read_text())["circuits"]:
        digest, split = record["sha256"], record["split"]
        if digest in seen and seen[digest] != split:
            raise SystemExit(f"Leakage: {record['path']}, {seen[digest]} / {split}")
        seen[digest] = split
print(f"{len(seen)} unique hashes; no exact overlap. Check families/parameters manually too.")
