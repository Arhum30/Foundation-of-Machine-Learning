#Assignment_5_ZarrarKhan_AftabNabi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#a

class KNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = euclidean_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            # majority vote
            values, counts = np.unique(k_labels, return_counts=True)
            pred = values[np.argmax(counts)]
            predictions.append(pred)

        return np.array(predictions)


#b

def euclidean_distance(xi, X):
    return np.sqrt(np.sum((X - xi) ** 2, axis=1))
# we are adding across each row this is why we use axis = 1

#c

def evaluate(y_pred, y_gt):
    correct = np.sum(y_pred == y_gt)
    return correct / len(y_gt)


#d

blobs = pd.read_pickle("data_blobs.pkl")
moons = pd.read_pickle("data_moons.pkl")
full = pd.read_pickle("data_full.pkl")


def prepare_data(df):

    train_df = df[['x1_train', 'x2_train', 'y_train']].dropna()
    test_df  = df[['x1_test', 'x2_test', 'y_test']].dropna()

    X_train = train_df[['x1_train', 'x2_train']].values
    y_train = train_df['y_train'].values

    X_test = test_df[['x1_test', 'x2_test']].values
    y_test = test_df['y_test'].values

    return X_train, y_train, X_test, y_test


datasets = {
    "Blobs": prepare_data(blobs),
    "Moons": prepare_data(moons),
    "Full": prepare_data(full)
}


#e

k_values = [1, 15, 30]

for name, (X_train, y_train, X_test, y_test) in datasets.items():
    print(f"\nDataset: {name}")
    for k in k_values:
        model = KNeighborsClassifier(k=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = evaluate(preds, y_test)
        print(f"k = {k}, Accuracy = {acc:.4f}")


#f

colors = ['red', 'blue', 'green', 'yellow']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("k-NN Performance on different dataset (k = 15)")

for col_idx, (name, (X_train, y_train, X_test, y_test)) in enumerate(datasets.items()):

    model = KNeighborsClassifier(k=15)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # TOP ROW: GROUND TRUTH

    ax = axes[0, col_idx]

    # combine train + test for dense GT
    X_all = np.vstack((X_train, X_test))
    y_all = np.hstack((y_train, y_test))

    for label in np.unique(y_all):
        idx = y_all == label
        ax.scatter(X_all[idx, 0], X_all[idx, 1],
                   c=colors[int(label)], marker='x', label=str(int(label)))

    ax.set_title(f"{name} GT")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    # BOTTOM ROW: PREDICTIONS

    ax = axes[1, col_idx]

    # TRAIN DATA (x marker)
    for label in np.unique(y_train):
        idx = y_train == label
        ax.scatter(X_train[idx, 0], X_train[idx, 1],
                   c=colors[int(label)], marker='x')

    # TEST PREDICTIONS (>)
    for label in np.unique(y_pred):
        idx = y_pred == label
        ax.scatter(X_test[idx, 0], X_test[idx, 1],
                   c=colors[int(label)], marker='>')

    ax.set_title(f"{name} pred")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
plt.show()


#Exercise 2:

#(a)

def gkernel(d_m, h):
    constant = 1.0 / (h * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * (d_m / h) ** 2
    return constant * math.exp(exponent)


#(b)

def weight_m(K):
    M = len(K)

    denominator = 0.0
    for n in range(M):
        denominator += K[n]

    w = []
    for m in range(M):
        w.append(K[m] / denominator)

    return w



#(c)

data = {
    "DayID": [11, 22, 33, 44, 50, 56, 67, 70, 78, 89, 90, 100],
    "Price": [2337, 2750, 2301, 2500, 1700, 2100, 1100, 1750, 1000, 1642, 2000, 1932]
}

df = pd.DataFrame(data)

X = df["DayID"].values
Y = df["Price"].values


#(d)

h = 10

x_range = list(range(1, 111))
y_pred = []

for x in x_range:

    # compute kernel values
    K = []
    for m in range(len(X)):
        d_m = x - X[m]
        K.append(gkernel(d_m, h))

    # compute weights
    w = weight_m(K)

    # weighted sum
    y_hat = 0.0
    for m in range(len(Y)):
        y_hat += w[m] * Y[m]

    y_pred.append(y_hat)


#plot

plt.figure(figsize=(6, 4))

# scatter points
plt.scatter(X, Y)

# regression curve
plt.plot(x_range, y_pred)

# labels and title
plt.title("Kernel Regression (DayID-Price)")
plt.xlabel("DayID")
plt.ylabel("Price")

plt.show()

#e

print("\nThe bandwidth h controls how smooth the curve is. \n"
      "If h is small, the curve follows the data very closely and looks wiggly (overfitting). \n"
      "If h is large, the curve becomes smoother but may miss important details (underfitting). \n"
      "So, h decides how closely the model follows the data.")

