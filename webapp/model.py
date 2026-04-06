import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_csv(file_path):
    # Skip Vicon's 3 metadata header rows
    df = pd.read_csv(file_path, skiprows=3, header=None)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    return df.select_dtypes(include=[np.number]).values


def build_reference_model(reference_files):
    sequences = []
    for f in reference_files:
        data = load_csv(f)
        if data.shape[0] > 0:
            sequences.append(data)

    if not sequences:
        raise ValueError("No valid reference data found")

    min_len = min(seq.shape[0] for seq in sequences)
    sequences = [seq[:min_len] for seq in sequences]
    stacked = np.stack(sequences)

    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0) + 1e-6

    return mean, std


def compute_similarity(user_file, reference_files):
    user = load_csv(user_file)

    if user.shape[0] == 0:
        raise ValueError("No numeric data found in uploaded CSV. Check file format.")

    mean, std = build_reference_model(reference_files)

    # Match to minimum number of columns
    min_cols = min(user.shape[1], mean.shape[1])
    user = user[:, :min_cols]
    mean = mean[:, :min_cols]
    std = std[:, :min_cols]

    # Match to minimum number of rows
    min_len = min(user.shape[0], mean.shape[0])
    user = user[:min_len]
    mean = mean[:min_len]
    std = std[:min_len]

    z = (user - mean) / std
    score = np.exp(-0.5 * np.mean(z**2))

    return float(score * 100)