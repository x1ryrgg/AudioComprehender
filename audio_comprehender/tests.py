import pandas as pd
import numpy as np
import json
from pathlib import Path

from .config import TEST_TSV


def check_columns(path):
    df = pd.read_csv(path, sep='\t')

    print("Columns:", list(df.columns), '\n\n\n')


    print("Data: \n", df.head(10))

    df = df.dropna(subset=["annotator_emo"])

    # Majority vote по каждому уникальному файлу
    aggregated = (
        df.groupby(["hash_id", "audio_path", "duration"])["annotator_emo"]
        .agg(lambda votes: votes.value_counts().idxmax())
        .reset_index()
        .rename(columns={"annotator_emo": "emotion"})
    )

    print("AGGREGATED: \n", aggregated.head(10))

    return


def check_json_report(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    print("DATA:")
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    return data


if __name__ == "__main__":
    check_json_report(path=Path("D:/crowd_data/crowd/crowd_test/dialogs/Phone_ARU_ON_report.json"))