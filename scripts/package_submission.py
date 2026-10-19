"""Package code, evidence, selected checkpoint, and report; exclude raw data/caches."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
destination = root / "output/submission.zip"
paths = [root / name for name in ["README.md", "requirements.txt", "requirements-lock.txt"]]
for dirname in ["chest_vae", "tests", "scripts"]:
    paths += [p for p in (root/dirname).rglob("*") if p.suffix in [".py", ".sh"]]
paths += [root / "report" / name for name in ["report.tex", "neurips_2025.sty", "loss_curves.pdf", "generated.png", "reconstructions.png"]]
paths += [root / "runs/vae" / name for name in ["best.pt", "config.json", "history.csv", "metrics.json", "timing.json", "data_manifest.json", "provenance.json", "train.log", "evaluate.log", "discussion.txt", "generated.png", "reconstructions.png"]]
paths += [root / "output/pdf/report.pdf"]
for p in paths:
    if not p.is_file():
        raise FileNotFoundError(p)
destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(paths):
        z.write(p, p.relative_to(root))
with zipfile.ZipFile(destination) as z:
    assert z.testzip() is None
print(f"{destination}: {len(paths)} files, {destination.stat().st_size:,} bytes")
