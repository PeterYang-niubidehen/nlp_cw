from __future__ import annotations

import argparse
import ast
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


def _clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_main_training_file(path: Path) -> pd.DataFrame:
    """
    Parse the main training file (6 tab-separated fields per data row).

    File contains 3 disclaimer lines at top; we keep only rows matching:
    - numeric record_id
    - article id formatted as @@<digits>
    - raw_label in [0, 4]
    """
    columns = ["record_id", "article_id", "keyword", "country", "text", "raw_label"]
    raw_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        engine="python",
        on_bad_lines="skip",
    )

    raw_label_numeric = pd.to_numeric(raw_df["raw_label"], errors="coerce")
    valid_mask = (
        raw_df["record_id"].astype(str).str.fullmatch(r"\d+")
        & raw_df["article_id"].astype(str).str.fullmatch(r"@@\d+")
        & raw_label_numeric.notna()
        & raw_label_numeric.between(0, 4)
    )
    df = raw_df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError(f"Could not parse records from {path}. Please check file contents.")

    df["record_id"] = df["record_id"].astype(int)
    df["article_id"] = (
        df["article_id"].astype(str).str.replace("@@", "", regex=False).astype(int)
    )
    df["text"] = df["text"].astype(str).map(_clean_text)
    df["raw_label"] = pd.to_numeric(df["raw_label"], errors="raise").astype(int)
    df["binary_label_raw"] = (df["raw_label"] >= 2).astype(int)

    return df


def parse_split_file(path: Path) -> pd.DataFrame:
    """
    Parse split file of shape:
    par_id,label
    4341,"[1, 0, 0, 1, 0, 0, 0]"

    Here `par_id` maps to `record_id` in the main training file.
    """
    df = pd.read_csv(path)
    required_cols = {"par_id", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Expected columns {required_cols} in {path}, found {set(df.columns)}")

    def to_binary(label_value: object) -> int:
        # Supports either scalar 0/1 labels or multi-label vectors like "[1,0,0,1,0,0,0]".
        if pd.isna(label_value):
            return 0
        if isinstance(label_value, (int, float)):
            return int(float(label_value) > 0)

        label_text = str(label_value).strip()
        if label_text in {"0", "1"}:
            return int(label_text)

        try:
            parsed = ast.literal_eval(label_text)
        except (ValueError, SyntaxError):
            bits = [int(x) for x in re.findall(r"[01]", label_text)]
            return int(any(bits))

        if isinstance(parsed, (list, tuple)):
            return int(any(int(x) for x in parsed))
        if isinstance(parsed, (int, float)):
            return int(float(parsed) > 0)
        return 0

    out = pd.DataFrame()
    out["record_id"] = pd.to_numeric(df["par_id"], errors="raise").astype(int)
    out["binary_label"] = df["label"].map(to_binary).astype(int)
    out["label_raw"] = df["label"].astype(str)
    out = out.drop_duplicates(subset=["record_id"], keep="last")
    return out


def parse_test_file(path: Path) -> pd.DataFrame:
    """
    Parse official test TSV:
    t_0  @@7258997  vulnerable  us  <text>
    """
    columns = ["sample_id", "article_id", "keyword", "country", "text"]
    df = pd.read_csv(path, sep="\t", header=None, names=columns, engine="python")

    valid_mask = (
        df["sample_id"].astype(str).str.fullmatch(r"t_\d+")
        & df["article_id"].astype(str).str.fullmatch(r"@@\d+")
    )
    df = df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError(f"Could not parse test records from {path}. Please check file contents.")

    df["article_id"] = (
        df["article_id"].astype(str).str.replace("@@", "", regex=False).astype(int)
    )
    df["text"] = df["text"].astype(str).map(_clean_text)
    return df


def load_official_splits(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = ensure_data(data_root)

    all_train = parse_main_training_file(paths.train_main)
    train_split = parse_split_file(paths.train_split)
    dev_split = parse_split_file(paths.dev_split)
    test_df = parse_test_file(paths.test_main)

    train_df = train_split.merge(
        all_train[["record_id", "article_id", "keyword", "country", "text"]],
        on="record_id",
        how="left",
    )
    dev_df = dev_split.merge(
        all_train[["record_id", "article_id", "keyword", "country", "text"]],
        on="record_id",
        how="left",
    )

    train_missing = train_df["text"].isna().sum()
    dev_missing = dev_df["text"].isna().sum()
    if train_missing or dev_missing:
        raise ValueError(
            "Missing text after merge with split files: "
            f"train_missing={train_missing}, dev_missing={dev_missing}"
        )

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
