# ===============================
# Spiral Dataset + NN (Windows-first)
# Requirements: tensorflow, numpy, matplotlib
# ===============================

import os

# Force CPU (prevents CUDA init noise / issues)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Reduce TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable XLA devices (helps on some machines)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# -----------------------------
# 1) Function: Generate Spiral Dataset (2 classes)
# -----------------------------
def generate_spiral(n_per_class=600, turns=2, noise=0.04, seed=42, clockwise=True):
    """
    Two-class spiral dataset (Neural Network Playground style)
    - turns: number of revolutions
    - clockwise: rotation direction
    Returns:
      X: shape (2*n_per_class, 2)
      y: shape (2*n_per_class,) in {0,1}
    """
    rng = np.random.default_rng(seed)

    max_theta = 2.0 * np.pi * turns
    t = np.linspace(0.0, max_theta, n_per_class)
    r = np.linspace(0.05, 1.0, n_per_class)

    direction = -1.0 if clockwise else 1.0

    # class 0
    theta0 = direction * t
    x0 = r * np.cos(theta0)
    y0 = r * np.sin(theta0)

    # class 1 (phase shift by pi to avoid overlap)
    theta1 = direction * (t + np.pi)
    x1 = r * np.cos(theta1)
    y1 = r * np.sin(theta1)

    # noise
    x0 += rng.normal(0, noise, size=n_per_class)
    y0 += rng.normal(0, noise, size=n_per_class)
    x1 += rng.normal(0, noise, size=n_per_class)
    y1 += rng.normal(0, noise, size=n_per_class)

    X = np.vstack([np.c_[x0, y0], np.c_[x1, y1]]).astype(np.float32)
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.float32)

    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# -----------------------------
# 3) Build NN Model
# -----------------------------
def build_model(l2=2e-4, lr=1e-2, run_eagerly=False):
    model = keras.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2)),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
        run_eagerly=run_eagerly,   # set True only if your machine "hangs" at epoch 1
    )
    return model


# -----------------------------
# 6) Plot decision boundary
# -----------------------------
def plot_decision_boundary(model, X, y, title, grid_step=0.01):
    x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )

    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    probs = model.predict(grid, verbose=0).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, probs, levels=50, alpha=0.75)
    plt.contour(xx, yy, probs, levels=[0.5], linewidths=2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=15)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()


def main():
    # Reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # -----------------------------
    # 2) Training Data: 2 turns, 2 classes non-overlapping
    # -----------------------------
    X_train, y_train = generate_spiral(
        n_per_class=800, turns=2, noise=0.04, seed=1, clockwise=True
    )

    # -----------------------------
    # 4) Testing Data: 4 turns or more
    # -----------------------------
    X_test, y_test = generate_spiral(
        n_per_class=800, turns=4, noise=0.04, seed=2, clockwise=True
    )

    model = build_model(l2=2e-4, lr=1e-2, run_eagerly=False)

    # Early stopping helps prevent overfit (as required by note)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True
    )

    # -----------------------------
    # 3) Train
    # -----------------------------
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # -----------------------------
    # 5) Evaluate (Train vs Test error should be close)
    # -----------------------------
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Train -> loss: {train_loss:.4f}, acc: {train_acc:.4f}")
    print(f"Test  -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    # -----------------------------
    # 6) Plot (two figures)
    # -----------------------------
    plot_decision_boundary(model, X_train, y_train, "Decision Boundary (Training Data)")
    plot_decision_boundary(model, X_test, y_test, "Decision Boundary (Testing Data)")
    plt.show()


if __name__ == "__main__":
    main()
