"""Data preparation with an explicit, reproducible split manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import zipfile

import numpy as np
from PIL import Image
import torch


def patient_key(name):
    # Patient identifiers inferred from the published filenames, not clinical metadata.
    m = re.match(r"(person\d+)_", name)
    if m:
        return m.group(1)
    m = re.match(r"(NORMAL2-IM-\d+|IM-\d+)-", name)
    if m:
        return m.group(1)
    raise ValueError(f"Unrecognized patient filename: {name}")


def prepare(archive, output, seed=42, val_fraction=0.1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # The Kaggle archive repeats the dataset under nested directories.
        candidates = [n for n in z.namelist() if re.search(r"/train/NORMAL/[^/]+\.jpeg$", n)]
        if not candidates:
            raise ValueError("Expected Kaggle chest_xray/train/NORMAL/*.jpeg")
        prefix = min((n.split("train/NORMAL/")[0] for n in candidates), key=lambda s: (len(s), s))
        entries = []
        seen_hashes = {}
        duplicates = []
        for split in ["test", "train", "val"]:
            names = sorted(n for n in z.namelist() if n.startswith(prefix + split + "/")
                           and n.lower().endswith((".jpeg", ".jpg", ".png")) and "/__" not in n)
            for name in names:
                relative = name[len(prefix):]
                if len(Path(relative).parts) != 3:
                    continue
                raw = z.read(name)
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen_hashes:
                    duplicates.append({"removed": relative, "kept": seen_hashes[digest]})
                    continue
                seen_hashes[digest] = relative
                entries.append({"path": relative, "official_split": split,
                                "label": Path(relative).parts[1], "patient": patient_key(Path(name).name),
                                "sha256": digest, "archive_path": name})
        test_groups = {e["patient"] for e in entries if e["official_split"] == "test"}
        train_pool = [e for e in entries if e["official_split"] == "train" and e["patient"] not in test_groups]
        groups = sorted({e["patient"] for e in train_pool})
        random.Random(seed).shuffle(groups)
        val_groups = set(groups[:max(1, round(len(groups)*val_fraction))])
        arrays = {k: [] for k in ["train", "val", "test"]}
        for e in entries:
            source = e["official_split"]
            e["split"] = ("test" if source == "test" else
                          "unused" if source == "val" or e["patient"] in test_groups else
                          "val" if e["patient"] in val_groups else "train")
            if e["split"] != "unused":
                import io
                with Image.open(io.BytesIO(z.read(e["archive_path"]))) as im:
                    im = im.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
                    arrays[e["split"]].append(np.asarray(im).copy())
        for k, images in arrays.items():
            np.save(output / f"{k}.npy", np.stack(images))
    summary = {k: dict(Counter(e["label"] for e in entries if e["split"] == k))
               for k in ["train", "val", "test", "unused"]}
    manifest = {"seed": seed, "val_group_fraction": val_fraction, "archive_prefix": prefix,
                "preprocessing": "PIL grayscale; direct bilinear resize to 64x64; uint8/255; no augmentation",
                "counts": summary, "duplicate_files_removed": duplicates, "entries": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def load_split(data_dir, split):
    x = np.load(Path(data_dir) / f"{split}.npy")
    return torch.from_numpy(x).unsqueeze(1).float().div_(255)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="data/chest-xray-pneumonia.zip")
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare(args.archive, args.output, args.seed)
