from __future__ import annotations

import argparse
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRAIN_DATA_URL = (
    "https://raw.githubusercontent.com/CRLala/NLPLabs-2024/main/"
    "Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv"
)
TRAIN_SPLIT_URL = (
    "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/"
    "semeval-2022/practice%20splits/train_semeval_parids-labels.csv"
)
DEV_SPLIT_URL = (
    "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/"
    "semeval-2022/practice%20splits/dev_semeval_parids-labels.csv"
)
TEST_DATA_URL = (
    "https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/"
    "semeval-2022/TEST/task4_test.tsv"
)


@dataclass
class DataPaths:
    train_main: Path
    train_split: Path
    dev_split: Path
    test_main: Path


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def ensure_data(root_dir: Path) -> DataPaths:
    root_dir.mkdir(parents=True, exist_ok=True)
    train_main = root_dir / "dontpatronizeme_pcl.tsv"
    train_split = root_dir / "train_semeval_parids-labels.csv"
    dev_split = root_dir / "dev_semeval_parids-labels.csv"
    test_main = root_dir / "task4_test.tsv"

    files_to_fetch = [
        (TRAIN_DATA_URL, train_main),
        (TRAIN_SPLIT_URL, train_split),
        (DEV_SPLIT_URL, dev_split),
        (TEST_DATA_URL, test_main),
    ]
    for url, path in files_to_fetch:
        if not path.exists():
            download_file(url, path)

    return DataPaths(
        train_main=train_main,
        train_split=train_split,
        dev_split=dev_split,
        test_main=test_main,
    )


def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_main_training_file(path: Path) -> pd.DataFrame:
    """
    Parses entries like:
    1 @@12345 keyword us Some text ... 0
    2 @@67890 keyword gb Another text ... 1

    The source format is irregular in line breaks, so regex over full text is
    safer than strict TSV parsing.
    """
    content = path.read_text(encoding="utf-8", errors="ignore")

    pattern = re.compile(
        r"(?ms)\s*(\d+)\s+@@(\d+)\s+(\S+)\s+(\S+)\s+(.*?)\s+([0-4])\s*(?=(?:\d+\s+@@\d+\s+\S+\s+\S+)|\Z)"
    )

    rows = []
    for record_id, par_id, keyword, country, text, raw_label in pattern.findall(content):
        rows.append(
            {
                "record_id": int(record_id),
                "par_id": int(par_id),
                "keyword": keyword,
                "country": country,
                "text": _clean_text(text),
                "raw_label": int(raw_label),
                "binary_label": 1 if int(raw_label) >= 2 else 0,
            }
        )

    if not rows:
        raise ValueError(
            f"Could not parse records from {path}. Please check file contents."
        )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["par_id"], keep="first")
    return df


def parse_split_file(path: Path) -> pd.DataFrame:
    """
    Robustly parses files where par_id,label pairs may appear one per line,
    many per line, or with optional header text.
    """
    content = path.read_text(encoding="utf-8", errors="ignore")
    pairs = re.findall(r"(\d+)\s*,\s*([01])", content)
    if not pairs:
        raise ValueError(f"Could not parse par_id,label pairs from {path}")

    df = pd.DataFrame(pairs, columns=["par_id", "label"]).astype(int)
    df = df.drop_duplicates(subset=["par_id"], keep="last")
    return df


def parse_test_file(path: Path) -> pd.DataFrame:
    """
    Parses entries like:
    t_0 @@7258997 vulnerable us Some text ...
    t_1 @@16397324 hopeless ng More text ...

    Records can span lines; parse with regex over the full file.
    """
    content = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"(?ms)\s*(t_\d+)\s+@@(\d+)\s+(\S+)\s+(\S+)\s+(.*?)\s*(?=(?:t_\d+\s+@@\d+\s+\S+\s+\S+)|\Z)"
    )

    rows = []
    for sample_id, par_id, keyword, country, text in pattern.findall(content):
        rows.append(
            {
                "sample_id": sample_id,
                "par_id": int(par_id),
                "keyword": keyword,
                "country": country,
                "text": _clean_text(text),
            }
        )

    if not rows:
        raise ValueError(
            f"Could not parse test records from {path}. Please check file contents."
        )

    df = pd.DataFrame(rows)
    return df


def load_official_splits(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = ensure_data(data_root)

    all_train = parse_main_training_file(paths.train_main)
    train_split = parse_split_file(paths.train_split)
    dev_split = parse_split_file(paths.dev_split)
    test_df = parse_test_file(paths.test_main)

    train_df = train_split.merge(all_train[["par_id", "text"]], on="par_id", how="left")
    dev_df = dev_split.merge(all_train[["par_id", "text"]], on="par_id", how="left")

    train_missing = train_df["text"].isna().sum()
    dev_missing = dev_df["text"].isna().sum()
    if train_missing or dev_missing:
        raise ValueError(
            "Missing text after merge with split files: "
            f"train_missing={train_missing}, dev_missing={dev_missing}"
        )

    train_df = train_df.rename(columns={"label": "binary_label"})
    dev_df = dev_df.rename(columns={"label": "binary_label"})

    return train_df, dev_df, test_df


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and inspect PCL data files.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory where data files are downloaded/read.",
    )
    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    train_df, dev_df, test_df = load_official_splits(args.data_root)

    print(f"Train size: {len(train_df)}")
    print(f"Dev size: {len(dev_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Train positive rate: {train_df['binary_label'].mean():.4f}")


if __name__ == "__main__":
    main()
