# ===============================
# Spiral Dataset + NN (Best Generalization: Train 2 turns, Test 4+ turns)
# Requirements: tensorflow, numpy, matplotlib
# ===============================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# -----------------------------
# 1) Generate Spiral Dataset (2 classes)
#    FIX: turns มากขึ้น => เกลียว "ยาวออกไป" ไม่ใช่ "ถี่ขึ้น" ใน r เดิม
# -----------------------------
def generate_spiral(n_per_class=800, turns=2, noise=0.04, seed=42, clockwise=True, base_turns=2):
    """
    Archimedean-like spiral with constant pitch:
      r = t / (2π*base_turns)
    => Train turns=2 จะมี r_max≈1
       Test  turns=4 จะมี r_max≈2 (ต่อออกไปด้านนอก)
    """
    rng = np.random.default_rng(seed)

    max_theta = 2.0 * np.pi * turns
    t = np.linspace(0.0, max_theta, n_per_class)

    direction = -1.0 if clockwise else 1.0
    theta0 = direction * t
    theta1 = direction * (t + np.pi)  # phase shift => 2 classes not overlapping

    # constant pitch based on base_turns (keep same "pattern speed")
    r = t / (2.0 * np.pi * base_turns)
    r = np.maximum(r, 0.05)


    x0 = r * np.cos(theta0)
    y0 = r * np.sin(theta0)
    x1 = r * np.cos(theta1)
    y1 = r * np.sin(theta1)

    # noise (same scale for train/test)
    x0 += rng.normal(0, noise, size=n_per_class)
    y0 += rng.normal(0, noise, size=n_per_class)
    x1 += rng.normal(0, noise, size=n_per_class)
    y1 += rng.normal(0, noise, size=n_per_class)

    X = np.vstack([np.c_[x0, y0], np.c_[x1, y1]]).astype(np.float32)
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.float32)

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# -----------------------------
# Feature Engineering (Playground-style + polar Fourier features)
# -----------------------------
def featurize_xy(X, base_turns=2, clockwise=True):
    x = X[:, 0:1]
    y = X[:, 1:2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    direction = -1.0 if clockwise else 1.0
    phase = theta - direction * (2.0 * np.pi * base_turns) * r

    feats = np.hstack([
        x, y,
        r,
        np.sin(phase), np.cos(phase),
        np.sin(2*phase), np.cos(2*phase),
        x**2, y**2, x*y
    ]).astype(np.float32)
    return feats


def standardize_train_apply_test(X_train_f, X_test_f):
    mu = X_train_f.mean(axis=0, keepdims=True)
    sd = X_train_f.std(axis=0, keepdims=True) + 1e-6
    return (X_train_f - mu) / sd, (X_test_f - mu) / sd


# -----------------------------
# 3) Build NN Model
# -----------------------------
def build_model(input_dim, l2=1.5e-3, lr=3e-3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ])
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    

    return model



def main():
    np.random.seed(0)
    tf.random.set_seed(0)

    # -----------------------------
    # 2) Training: 2 turns
    # -----------------------------
    X_train, y_train = generate_spiral(
        n_per_class=900,
        turns=2,
        noise=0.05,
        seed=1,
        clockwise=True,
        base_turns=2
    )

    # -----------------------------
    # 4) Testing: 4 turns (or more)
    # -----------------------------
    X_test, y_test = generate_spiral(
        n_per_class=900,
        turns=4,
        noise=0.05,     # ให้ noise เท่ากันเพื่อวัด generalization จริง
        seed=2,
        clockwise=True,
        base_turns=2
    )

# Feature engineering
    X_train_f = featurize_xy(X_train, base_turns=2, clockwise=True)
    X_test_f  = featurize_xy(X_test,  base_turns=2, clockwise=True)


    # Standardize (train stats only)
    mu = X_train_f.mean(axis=0, keepdims=True)
    sd = X_train_f.std(axis=0, keepdims=True) + 1e-6

    X_train_fn = (X_train_f - mu) / sd
    X_test_fn  = (X_test_f  - mu) / sd

    # Clip AFTER normalization (prevents out-of-range features on test)
    X_train_fn = np.clip(X_train_fn, -5, 5)
    X_test_fn  = np.clip(X_test_fn,  -5, 5)



    model = build_model(input_dim=X_train_fn.shape[1], l2=8e-4, lr=3e-3)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_fn, y_train,
        validation_split=0.2,
        epochs=400,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    train_loss, train_acc = model.evaluate(X_train_fn, y_train, verbose=0)
    test_loss,  test_acc  = model.evaluate(X_test_fn,  y_test,  verbose=0)
    print(f"Train -> loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Test  -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # ---- Plot boundary ต้อง normalize grid ด้วย mu/sd ด้วย ----
    def predict_on_raw_grid(grid_raw):
        gf = featurize_xy(grid_raw, base_turns=2, clockwise=True)
        gfn = (gf - mu) / sd
        gfn = np.clip(gfn, -5, 5)   # ให้เหมือนตอน train/test ด้วย (สำคัญ)
        return model.predict(gfn, verbose=0)

    # Patch plot function quickly (local)
    def plot_db_norm(X_raw, y, title, grid_step=0.02):
        x_min, x_max = X_raw[:, 0].min() - 0.3, X_raw[:, 0].max() + 0.3
        y_min, y_max = X_raw[:, 1].min() - 0.3, X_raw[:, 1].max() + 0.3
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_step),
            np.arange(y_min, y_max, grid_step)
        )
        grid_raw = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
        probs = predict_on_raw_grid(grid_raw).reshape(xx.shape)

        plt.figure(figsize=(7, 6))
        plt.contourf(xx, yy, probs, levels=50, alpha=0.75)
        plt.contour(xx, yy, probs, levels=[0.5], linewidths=2)
        plt.scatter(X_raw[y == 0, 0], X_raw[y == 0, 1], s=15)
        plt.scatter(X_raw[y == 1, 0], X_raw[y == 1, 1], s=15)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()

    plot_db_norm(X_train, y_train, "Decision Boundary (Training Data)")
    plot_db_norm(X_test,  y_test,  "Decision Boundary (Testing Data)")
    plt.show()


if __name__ == "__main__":
    main()
