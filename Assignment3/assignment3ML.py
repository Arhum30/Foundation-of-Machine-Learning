
#Muhammad Ahmed Khan muk9044
#Muhammad Arhum Siddiqui mus8857

# Sheet 3
#%% Exercise 2a

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
img = digits.images
labels = digits.target
done_digits = []
plt.figure(figsize=(6,6))
counter = 1

for i in range(9):
    digit = labels[i]
    if digit != 9 and digit not in done_digits:
        plt.subplot(3,3,counter)
        plt.imshow(img[i])
        plt.axis('off')
        done_digits.append(digit)
        counter = counter + 1
    if counter > 9:
        break
plt.show()

#2b
data = digits.data
M = len(data)
N = len(data[0]) #gives numbers of pixels per image
print("Number of samples (M):", M)
print("Number of features (N):", N)
#2ci
means = []
for n in range(N):
    col_sum = 0
    for m in range(M):
        col_sum += data[m][n]
    means.append(col_sum / M)
centered_data = []
for m in range(M):
    new = []
    for n in range(N):
        value = data[m][n] - means[n]
        new.append(value)
    centered_data.append(new)

#2cii
cov_matrix = []

for i in range(N):
    new_2 = []
    for j in range(N): 
        cov = 0
        for m in range(M): # for loop for each image
            cov += centered_data[m][i] * centered_data[m][j]
        cov = cov / (M - 1)
        new_2.append(cov)
    cov_matrix.append(new_2)

#2ciii

import numpy as np
C = np.array(cov_matrix)   
eigenvalues, eigenvectors = np.linalg.eigh(C)

# print(eigenvalues, eigenvectors)

#2civ

print("The columns of U are the eigenvectors of the covariance matrix, i.e. the principal component directions. \nThe rows of U show how each original feature contributes to those eigenvectors.")
#2cv
sorted_indices = np.argsort(eigenvalues)[::-1]

index_1 = sorted_indices[0]
index_2 = sorted_indices[1]

#2cv
first_vector = eigenvectors[:, index_1]
second_vector = eigenvectors[:, index_2]
U_K = np.column_stack((first_vector, second_vector))


#2cvi and 2cvii
V_proj = np.dot(centered_data, U_K)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
scatter = plt.scatter(V_proj[:, 0], V_proj[:, 1], c=digits.target, cmap="viridis", s=10)
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.title("Principal Components of Digits (K=2)")
plt.colorbar(scatter, label="Digit Label")

plt.show()


# %% Exercise 3
import math
#3a
def compute_p_j_given_m(V, m, j, sigma_m): #m is the reference point
    numerator = 0 #j is candidate neighbour we are compyting probability for
    denominator = 0
    for k in range(len(V)):
        if k == m:
            continue
        distance_squared = 0
        for n in range(len(V[0])):
            distance_squared += (V[m][n] - V[k][n]) ** 2
        exponent = -distance_squared / (2 * sigma_m ** 2)
        value = math.exp(exponent)
        denominator += value
        if k == j:
            numerator = value
    if denominator == 0:
        return 0
    return numerator / denominator
#3b
def compute_perplexity(V, m, sigma_m):
    M = len(V)
    probs = []

    for j in range(M):
        if j == m:
            continue
        p = compute_p_j_given_m(V, m, j, sigma_m)
        probs.append(p)
    entropy = 0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    perplexity = 2 ** entropy
    return perplexity

#3c
def binary_search(V, m, target, l_min=0, l_max=100, e=0.001, iter=100):
    low = l_min
    high = l_max
    for i in range(iter):
        mid = (low + high) / 2
        perp = compute_perplexity(V, m, mid)

        if abs(perp - target) < e:
            return mid
        if perp > target:
            high = mid
        else:
            low = mid
        if (high - low) < e:
            break
    return mid
#3d
import pandas as pd
df = pd.read_pickle(r"D:\university\fourth_semester\Machine Learning\labs\lab3ML\tsne_data.pkl")
V = df.values
# print(V)
sigmas = []
for m in range(len(V)):
    sigma_m = binary_search(V, m, target=10, l_min=0.15, l_max=50)
    sigmas.append(sigma_m)

print("Total sigmas calculated:", len(sigmas))

print(sigmas)
# %%
