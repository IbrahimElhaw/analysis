# import numpy as np
# from scipy.stats import wasserstein_distance
#
#
# X = np.load("training__data/original.npy")
# Y = np.load("training__data/represented.npy")
# cost_matrix = np.zeros((len(X), len(Y)))
#
# for i, gesture_X in enumerate(X):
#     print(i)
#     for j, gesture_Y in enumerate(Y):
#         gesture_X_flat = gesture_X.flatten()
#         gesture_Y_flat = gesture_Y.flatten()
#         cost_matrix[i, j] = wasserstein_distance(gesture_X_flat, gesture_Y_flat)
#
# min_distances = cost_matrix.min(axis=1)
# closest_matches = cost_matrix.argmin(axis=1)
#
# print("Cost Matrix (Wasserstein Distances):")
# print(cost_matrix)
# print("\nMinimum Wasserstein Distances for each gesture in X:")
# print(min_distances)
# print("\nIndex of Closest Matching Gesture in Y for each gesture in X:")
# print(closest_matches)
#
# np.save("costs matrix represented.npy", cost_matrix)


import numpy as np
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed

# Load data
X = np.load("training__data/original.npy")
Y = np.load("training__data/represented.npy")

Y=Y[:10]

# Define function to compute distances for a single row
def compute_row_distances(i):
    gesture_X_flat = X[i].flatten()
    print(i)
    return [wasserstein_distance(gesture_X_flat, Y[j].flatten()) for j in range(len(Y))]

# Use Parallel processing to compute cost matrix
cost_matrix = Parallel(n_jobs=-1)(delayed(compute_row_distances)(i) for i in range(len(X)))
cost_matrix = np.array(cost_matrix)

# Find minimum distances and closest matches
min_distances = cost_matrix.min(axis=1)
closest_matches = cost_matrix.argmin(axis=1)

# Save results
np.save("costs_matrix_represented.npy", cost_matrix)

