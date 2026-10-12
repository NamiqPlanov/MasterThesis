"""
Stage S1a: fetch the two benchmark datasets named in the proposal.

  ULB Credit Card Fraud Detection  - 284,807 rows, 0.172% fraud, PCA features
  IEEE-CIS Fraud Detection         - 590,540 rows, ~3.5% fraud, named features

The IEEE-CIS data is competition data: the API refuses it until the rules have
been accepted on the website, so that failure is reported as an action for the
user rather than a crash.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import kaggle
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_dataset(slug: str, folder: str) -> Path | None:
    out = DATA / folder
    if out.exists() and any(out.glob("*.csv")):
        print(f"  [skip] {folder} already present")
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"  downloading dataset {slug} ...")
    kaggle.api.dataset_download_files(slug, path=str(out), unzip=True, quiet=False)
    return out


def fetch_competition(slug: str, folder: str) -> Path | None:
    out = DATA / folder
    if out.exists() and any(out.glob("*.csv")):
        print(f"  [skip] {folder} already present")
        return out
    out.mkdir(parents=True, exist_ok=True)
    print(f"  downloading competition {slug} ...")
    try:
        kaggle.api.competition_download_files(slug, path=str(out), quiet=False)
    except Exception as exc:
        msg = str(exc)
        if "403" in msg or "Forbidden" in msg or "rules" in msg.lower():
            print(f"  [BLOCKED] {slug}: competition rules not accepted.")
            print(f"            Visit https://www.kaggle.com/competitions/{slug}/rules")
            print("            and click 'I Understand and Accept', then re-run.")
            return None
        raise
    for z in out.glob("*.zip"):
        print(f"  unzipping {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        z.unlink()
    return out


def summarise(path: Path, label: str, target: str):
    csvs = sorted(path.glob("*.csv"))
    if not csvs:
        print(f"  {label}: no csv found")
        return
    print(f"\n  {label}")
    for c in csvs:
        print(f"    {c.name}  ({c.stat().st_size / 1024 / 1024:.1f} MB)")
    main = max(csvs, key=lambda p: p.stat().st_size)
    df = pd.read_csv(main, nrows=200_000)
    print(f"    read {main.name}: {len(df):,} rows (capped), {df.shape[1]} columns")
    if target in df.columns:
        rate = df[target].mean()
        print(f"    positive rate in sample: {rate * 100:.3f}%  ({int(df[target].sum())} of {len(df):,})")
    named = [c for c in df.columns[:14]]
    print(f"    first columns: {named}")


def main() -> int:
    kaggle.api.authenticate()
    print("=" * 74)
    print("  S1a - DATA ACQUISITION")
    print("=" * 74)

    ulb = fetch_dataset("mlg-ulb/creditcardfraud", "ulb")
    ieee = fetch_competition("ieee-fraud-detection", "ieee")

    if ulb:
        summarise(ulb, "ULB Credit Card Fraud Detection", "Class")
    if ieee:
        summarise(ieee, "IEEE-CIS Fraud Detection", "isFraud")
    else:
        print("\n  IEEE-CIS not downloaded - see message above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
