import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits

#1a

digits = load_digits()
data, labels = digits.data, digits.target

n = len(data)
np.random.seed(42)
# shuffles the index .permutation
indices = np.random.permutation(n)
data, labels = data[indices], labels[indices]

train_size = int(0.7 * n)
test_size = int(0.2 * n)

train_data = data[:train_size]
train_labels = labels[:train_size]

test_data = data[train_size:train_size + test_size]
test_labels = labels[train_size:train_size + test_size]

val_data = data[train_size + test_size:]
val_labels = labels[train_size + test_size:]

print("Train:", train_data.shape, "Test:", test_data.shape, "Val:", val_data.shape)


#b

def evaluate(y_pred, y_gt):
    return np.mean(y_pred == y_gt)


#c
# we use pca to compress 64 to 4 features as KNN use distance so it would be hard considering 64 features
def pca_fit_transform(X, num_components):
    mean = np.mean(X, axis=0) #mean contains the average value of each picture location
    X_centered = X - mean

    cov = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)
    # print(X.shape, X.shape[0])
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    #eigen value is amount of variance in that direction
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:num_components]]

    return np.dot(X_centered, components), mean, components


def pca_transform(X, mean, components):
    return np.dot(X - mean, components)


# Apply PCA → 4D
train_pca, mean, comp = pca_fit_transform(train_data, 4)
test_pca = pca_transform(test_data, mean, comp)
val_pca = pca_transform(val_data, mean, comp)


#d

def euclidean_distance(x, X):
    return np.sqrt(np.sum((X - x) ** 2, axis=1))


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        preds = []
        for x in X_test:
            distances = euclidean_distance(x, self.X)
            idx = np.argsort(distances)[:self.k]
            labels = self.y[idx]

            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])

        return np.array(preds)


# Validation to choose k
k_values = [1, 4, 8, 16]
results = {}

print("\nValidation Results:")
for k in k_values:
    model = KNN(k)
    model.fit(train_pca, train_labels)

    acc = evaluate(model.predict(val_pca), val_labels)
    results[k] = acc
    print(f"k={k}: {acc:.4f}")

best_k = max(results, key=results.get)
print("Best k:", best_k)


#e


model = KNN(best_k)
model.fit(train_pca, train_labels)

test_acc = evaluate(model.predict(test_pca), test_labels)
print("Test Accuracy:", test_acc)


#f


test_2d, _, _ = pca_fit_transform(test_data, 2)
test_pred = model.predict(test_pca)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Predicted
for label in np.unique(test_pred):
    idx = test_pred == label
    axes[0].scatter(test_2d[idx, 0], test_2d[idx, 1], label=label)

axes[0].set_title("Predicted Labels")
axes[0].legend()

# Ground truth
for label in np.unique(test_labels):
    idx = test_labels == label
    axes[1].scatter(test_2d[idx, 0], test_2d[idx, 1], label=label)

axes[1].set_title("Ground Truth Labels")
axes[1].legend()

plt.tight_layout()
plt.show()


#2a

train_df = pd.read_pickle("classification_fit.pkl")
test_df = pd.read_pickle("classification_test.pkl")
#train_df.head()
X_train = train_df[['x1', 'x2']].values
X_test = test_df[['x1', 'x2']].values

#b

# Add bias term
X_train_tilde = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test_tilde = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

#c

# Ground truth
t_train = np.where(train_df['label'] == 0, -1, 1)


#d

def fit(X, t):
    return np.linalg.inv(X.T @ X) @ X.T @ t


#e

def predict(X, w):
    y = X @ w
    #vector multiplication to compute scores for all samples
    return np.where(y >= 0, 1, -1)


#f

w = fit(X_train_tilde, t_train)
t_pred = predict(X_test_tilde, w)

print("\nWeights:", w)


#g

t_train = np.where(train_df['label'] == 0, -1, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Linear Classification")

# Train plot
for label in [-1, 1]:
    idx = t_train == label
    axes[0].scatter(X_train[idx, 0], X_train[idx, 1], label=label)

axes[0].set_title("Train Data")
axes[0].legend()

# Test plot + boundary
for label in [-1, 1]:
    idx = t_pred == label
    axes[1].scatter(X_test[idx, 0], X_test[idx, 1], label=label)

w1, w2, b = w
x_vals = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
y_vals = -(w1 * x_vals + b) / w2

axes[1].plot(x_vals, y_vals, color="blue")
axes[1].set_title("Test Data")
axes[1].legend()

plt.tight_layout()
plt.show()
