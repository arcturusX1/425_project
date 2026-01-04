import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def extract_mfcc_features(
    gtzan_path,
    n_mfcc=20,
    sr=22050,
    duration=30
):
    X = []
    y = []
    genre_to_id = {}

    genres = sorted(os.listdir(gtzan_path))
    for idx, genre in enumerate(genres):
        genre_to_id[genre] = idx
        genre_dir = os.path.join(gtzan_path, genre)

        for fname in tqdm(os.listdir(genre_dir), desc=f"Processing {genre}"):
            if not fname.endswith(".wav"):
                continue

            file_path = os.path.join(genre_dir, fname)
            try:
                audio, _ = librosa.load(
                    file_path, sr=sr, duration=duration
                )

                mfcc = librosa.feature.mfcc(
                    y=audio, sr=sr, n_mfcc=n_mfcc
                )

                mfcc_mean = mfcc.mean(axis=1)
                mfcc_std = mfcc.std(axis=1)

                feature = np.concatenate([mfcc_mean, mfcc_std])
                X.append(feature)
                y.append(idx)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, genre_to_id


if __name__ == "__main__":
    GTZAN_DIR = "data/gtzan/genres_original"

    X, y, genre_map = extract_mfcc_features(GTZAN_DIR)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    with open("data/processed/genre_map.json", "w") as f:
        json.dump(genre_map, f, indent=2)

    print("Feature extraction complete.")
    print("X shape:", X.shape)
