# requirements.txt
# ----------------
# tensorflow>=2.12.0
# librosa>=0.10.0
# numpy>=1.24.0
# pandas>=2.0.0
# matplotlib>=3.7.0
# scikit-learn>=1.3.0
# h5py>=3.8.0
#
# Expected dataset layout (already downloaded and extracted):
#   ./UrbanSound8K/audio/fold1 ... fold10/*.wav
#   ./UrbanSound8K/metadata/UrbanSound8K.csv
#
# Run: python urbansound_cnn.py
#
# Cache files written on first run (preprocessing is skipped if all exist):
#   X.npy      -> spectrograms, shape (N, 128, 128, 1), float32
#   y.npy      -> class IDs,    shape (N,), int64
#   folds.npy  -> fold numbers, shape (N,), int64 (needed to split by fold)

import os
import random

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import librosa
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models, callbacks, optimizers

# ----------------------------- Configuration --------------------------------
SEED = 42

DATASET_DIR = "./UrbanSound8K"
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata", "UrbanSound8K.csv")

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 128  # time axis is padded/truncated to this length
INPUT_SHAPE = (N_MELS, TARGET_FRAMES, 1)
NUM_CLASSES = 10

VAL_FOLD = 10  # folds 1-9 -> train, fold 10 -> validation
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 5

X_CACHE = "X.npy"
Y_CACHE = "y.npy"
FOLDS_CACHE = "folds.npy"
PROGRESS_EVERY = 500

MODEL_PATH = "audio_cnn_model.h5"
CM_PATH = "confusion_matrix.png"
CURVES_PATH = "training_curves.png"


# --------------------------------- Seeding ----------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)  # seeds python, numpy, and tf together
    except AttributeError:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # older TF versions: skip op-level determinism


# ------------------------------ Preprocessing -------------------------------
def audio_to_melspec(path: str) -> np.ndarray:
    """Load a wav file and return a (128, 128, 1) normalized dB Mel-spectrogram."""
    signal, _ = librosa.load(path, sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # range approx [-80, 0] dB

    # Pad (with the floor value, i.e. "silence") or truncate the time axis.
    n_frames = mel_db.shape[1]
    if n_frames < TARGET_FRAMES:
        pad = TARGET_FRAMES - n_frames
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant", constant_values=-80.0)
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]

    # Scale roughly from [-80, 0] dB to [0, 1] for stable training.
    mel_db = (np.clip(mel_db, -80.0, 0.0) + 80.0) / 80.0

    return mel_db.astype(np.float32)[..., np.newaxis]


def load_or_build_dataset(metadata: pd.DataFrame):
    """Return X, y, folds. Uses .npy cache if present, otherwise builds and caches."""
    if all(os.path.exists(p) for p in (X_CACHE, Y_CACHE, FOLDS_CACHE)):
        print(f"Found cache ({X_CACHE}, {Y_CACHE}, {FOLDS_CACHE}); skipping preprocessing.")
        X = np.load(X_CACHE)
        y = np.load(Y_CACHE)
        folds = np.load(FOLDS_CACHE)
        print(f"Loaded cached data: X={X.shape}, y={y.shape}, folds={folds.shape}")
        return X, y, folds

    total = len(metadata)
    print(f"No cache found. Preprocessing {total} audio files (this can take a while)...")

    X_list, y_list, fold_list = [], [], []
    skipped = 0

    for i, row in enumerate(metadata.itertuples(index=False), start=1):
        file_path = os.path.join(AUDIO_DIR, f"fold{row.fold}", row.slice_file_name)
        try:
            X_list.append(audio_to_melspec(file_path))
            y_list.append(int(row.classID))
            fold_list.append(int(row.fold))
        except Exception as exc:  # corrupt / missing file: skip but keep going
            skipped += 1
            print(f"  [warn] skipping {file_path}: {exc}")

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  Processed {i}/{total} files")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    folds = np.array(fold_list, dtype=np.int64)

    np.save(X_CACHE, X)
    np.save(Y_CACHE, y)
    np.save(FOLDS_CACHE, folds)
    print(f"Done. Saved cache: X={X.shape}, y={y.shape}, folds={folds.shape} (skipped {skipped})")
    return X, y, folds


# ---------------------------------- Model -----------------------------------
def build_model() -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            # Classifier head
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------- Plotting ---------------------------------
def save_training_curves(history, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="Train loss")
    axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, class_names, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (validation fold {VAL_FOLD})")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------- Main ------------------------------------
def main() -> None:
    seed_everything(SEED)

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. "
            "Expected UrbanSound8K extracted to ./UrbanSound8K/."
        )

    # Load metadata: slice_file_name, classID, fold (+ class name for plot labels).
    metadata = pd.read_csv(METADATA_PATH)
    class_map = (
        metadata[["classID", "class"]].drop_duplicates().sort_values("classID")
    )
    class_names = [str(c) for c in class_map["class"].tolist()]
    metadata = metadata[["slice_file_name", "classID", "fold"]]
    print(f"Loaded metadata: {len(metadata)} rows, classes: {class_names}")

    X, y, folds = load_or_build_dataset(metadata)

    # Standard UrbanSound8K protocol: folds 1-9 train, fold 10 validation.
    # Split is by fold membership only; nothing is shuffled across folds.
    train_mask = folds != VAL_FOLD
    val_mask = folds == VAL_FOLD
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Train (folds 1-9): {X_train.shape[0]} samples | "
          f"Validation (fold {VAL_FOLD}): {X_val.shape[0]} samples")

    model = build_model()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    # shuffle=True only reorders batches within the training folds each epoch.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        shuffle=True,
        verbose=2,
    )

    # Evaluation (weights already restored to best epoch by EarlyStopping).
    val_loss, val_acc = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    print(f"\nValidation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

    y_pred = np.argmax(model.predict(X_val, batch_size=BATCH_SIZE, verbose=0), axis=1)
    save_confusion_matrix(y_val, y_pred, class_names, CM_PATH)
    print(f"Confusion matrix saved to {CM_PATH}")

    save_training_curves(history, CURVES_PATH)
    print(f"Training curves saved to {CURVES_PATH}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
