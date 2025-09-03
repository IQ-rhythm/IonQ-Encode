# data/preprocessing/preprocess_fmnist.py

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

# === Config ===
DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

IMG_SIZE = 8
PCA_COMPONENTS = [16, 32, 64]

# Label mapping for T2 and T4
T2_CLASSES = [0, 2]  # 0: T-shirt/top, 2: Pullover
T4_CLASSES = [0, 2, 4, 6]  # T-shirt/top, Pullover, Coat, Shirt


def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train = datasets.FashionMNIST(root="data/raw", train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root="data/raw", train=False, download=True, transform=transform)

    x_train = train.data.numpy()
    y_train = train.targets.numpy()
    x_test = test.data.numpy()
    y_test = test.targets.numpy()

    # Resize to 8x8
    x_train = np.array([np.array(transforms.Resize((IMG_SIZE, IMG_SIZE))(img.unsqueeze(0))) for img in train.data])
    x_test = np.array([np.array(transforms.Resize((IMG_SIZE, IMG_SIZE))(img.unsqueeze(0))) for img in test.data])

    return x_train, y_train, x_test, y_test


def flatten_and_scale(x):
    x = x.reshape(x.shape[0], -1) / 255.0
    return x


def apply_pca(x_train, x_test, n_components):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    return x_train_pca, x_test_pca


def save_dataset(name, x_train, y_train, x_test, y_test):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Saved {name} to {path}")


def create_subset(x, y, classes):
    idx = np.isin(y, classes)
    x_sub = x[idx]
    y_sub = y[idx]
    # Reindex labels
    class_map = {c: i for i, c in enumerate(classes)}
    y_sub = np.array([class_map[label] for label in y_sub])
    return x_sub, y_sub


if __name__ == "__main__":
    print("Loading Fashion-MNIST...")
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    x_train = flatten_and_scale(x_train)
    x_test = flatten_and_scale(x_test)

    for k in PCA_COMPONENTS:
        print(f"Applying PCA with {k} components...")
        x_train_pca, x_test_pca = apply_pca(x_train, x_test, k)

        # T10 full dataset
        save_dataset(f"fashion_mnist_pca{k}_T10", x_train_pca, y_train, x_test_pca, y_test)

        # T2 dataset
        x2_train, y2_train = create_subset(x_train_pca, y_train, T2_CLASSES)
        x2_test, y2_test = create_subset(x_test_pca, y_test, T2_CLASSES)
        save_dataset(f"fashion_mnist_pca{k}_T2", x2_train, y2_train, x2_test, y2_test)

        # T4 dataset
        x4_train, y4_train = create_subset(x_train_pca, y_train, T4_CLASSES)
        x4_test, y4_test = create_subset(x_test_pca, y_test, T4_CLASSES)
        save_dataset(f"fashion_mnist_pca{k}_T4", x4_train, y4_train, x4_test, y4_test)

    print("All datasets saved in data/processed/")
