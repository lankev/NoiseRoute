"""Create a source/results archive, excluding caches and virtual environments."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path(__file__).resolve().parents[1]
excluded = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache", "build", "dist", "target", "data", "runs"}
files = [
    p
    for p in root.rglob("*")
    if p.is_file()
    and not any(
        x in excluded or x.startswith(".venv") or x.endswith(".egg-info") for x in p.relative_to(root).parts
    )
]
manifest = {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in files
    if p.name != "SHA256.json"
}
(root / "SHA256.json").write_text(json.dumps(manifest, indent=2))
with ZipFile(root.parent / "NoiseRoute.zip", "w", ZIP_DEFLATED) as z:
    for p in files:
        if p.name != "SHA256.json":
            z.write(p, Path(root.name) / p.relative_to(root))
    z.write(root / "SHA256.json", Path(root.name) / "SHA256.json")
print(root.parent / "NoiseRoute.zip")
