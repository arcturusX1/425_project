import os
import json
import numpy as np
import librosa
from tqdm import tqdm


def extract_melspectrograms(gtzan_path, n_mels=64, sr=22050, duration=30, hop_length=512, n_fft=2048, time_steps=128):
    X = []
    y = []
    genre_to_id = {}

    genres = sorted(os.listdir(gtzan_path))
    for idx, genre in enumerate(genres):
        genre_to_id[genre] = idx
        genre_dir = os.path.join(gtzan_path, genre)

        for fname in tqdm(os.listdir(genre_dir), desc=f"Processing {genre}"):
            if not fname.endswith('.wav'):
                continue

            file_path = os.path.join(genre_dir, fname)
            try:
                audio, _ = librosa.load(file_path, sr=sr, duration=duration)
                # compute mel-spectrogram
                S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                S_db = librosa.power_to_db(S, ref=np.max)

                # pad or truncate time axis to time_steps
                if S_db.shape[1] < time_steps:
                    pad_width = time_steps - S_db.shape[1]
                    S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=(S_db.min(),))
                elif S_db.shape[1] > time_steps:
                    S_db = S_db[:, :time_steps]

                X.append(S_db.astype(np.float32))
                y.append(idx)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    X = np.array(X)
    # reshape to (N, 1, n_mels, time_steps)
    X = X[:, None, :, :]
    y = np.array(y, dtype=np.int64)

    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/X_spec.npy', X)
    np.save('data/processed/y.npy', y)
    with open('data/processed/genre_map.json', 'w') as f:
        json.dump(genre_to_id, f, indent=2)

    print('Saved spectrograms to data/processed/X_spec.npy with shape', X.shape)
    return X, y, genre_to_id


if __name__ == '__main__':
    GTZAN_DIR = 'data/gtzan/genres_original'
    extract_melspectrograms(GTZAN_DIR)
