#!/usr/bin/env python
# src/utils/plot_shard_distribution.py

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 0) Ensure the project root (two levels above utils/) is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_processing_agent import DataProcessingAgent
from src.agents.feature_extraction_agent import FeatureExtractionAgent

def main():
    # 1) Load & clean the dataset
    data_path = PROJECT_ROOT / "data" / "processed_diagnosed_cbc_data.csv"
    df        = DataProcessingAgent(data_path).run()

    # 2) Extract features & labels
    X, y      = FeatureExtractionAgent().run(df)

    # 3) Encode labels as integers for sharding
    y_enc     = pd.Categorical(y).codes
    idx       = np.arange(len(y_enc))
    np.random.seed(42)
    np.random.shuffle(idx)

    # 4) Split indices into 3 shards
    num_clients = 3
    shards      = np.array_split(idx, num_clients)

    # 5) Plot class distributions per shard
    plt.figure(figsize=(8,6))
    for i, shard in enumerate(shards):
        counts = Counter(y.iloc[shard])
        labels, vals = zip(*counts.items())
        plt.bar(
            [f"{lab}\n(C{i+1})" for lab in labels],
            vals,
            label=f"Client {i+1}"
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Samples per Class")
    plt.title("Class Distribution in Each Client Shard")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from collections import Counter
    main()

