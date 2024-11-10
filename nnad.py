from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

from physical_characteristics import load_data, filter_data
from scipy.spatial.distance import cdist

import numpy as np

arranged_original = "arranged/original"
arranged_manipulated = "arranged/manipulated"
arranged_represented = "arranged/represented"
arranged_manipulated_2_modes = "arranged/manipulated_2_modes"
arranged_edge_data = "arranged/second mode edge data"
number = 4


Xo, Yo, To, Vo, infos_o = load_data(arranged_original)
data_o = filter_data(Xo, infos_o, Yo, To, Vo,  number)

Xr, Yr, Tr, Vr, infos_r = load_data(arranged_represented)
data_r = filter_data(Xr, infos_r, Yr, Tr, Vr, number)
s=0


def interpolate(y_values, n_points, interp="linear"):
    time = np.linspace(0, len(y_values) - 1, len(y_values), endpoint=True)
    time_inter = np.linspace(0, len(y_values) - 1, n_points, endpoint=True)
    f = interp1d(time, y_values, kind=interp)
    return f(time_inter)

def prepare_data(X, Y):
    prepared_data = []

    for x_sample, y_sample in zip(X, Y):
        if len(x_sample)>6:
            print("more than 6, ignored")
            continue
        x_padded = [interpolate(stroke, 360//len(x_sample)) for stroke in x_sample]
        y_padded = [interpolate(stroke, 360//len(y_sample)) for stroke in y_sample]

        x_padded = np.concatenate(x_padded)
        y_padded = np.concatenate(y_padded)

        sample_vector = np.concatenate([x_padded, y_padded])
        prepared_data.append(sample_vector)

    # Convert list of arrays to a single 2D array (n_samples, 700)
    return np.array(prepared_data)



def calculate_nnad(synthetic_data, target_data):
    # Fit NearestNeighbors for the single nearest neighbor
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_data)

    # Calculate distances to the nearest neighbor
    distances, _ = nbrs.kneighbors(synthetic_data)

    # Average of the nearest neighbor distances
    nnad = np.mean(distances)
    return nnad


if __name__ == '__main__':
    print("in")
    mat_r = prepare_data(Xr, Yr)
    mat_o = prepare_data(Xo, Yo)
    print("out")
    print("Shape of mat_r:", mat_r.shape)
    print("Shape of mat_o:", mat_o.shape)

    distance = calculate_nnad(mat_o, mat_r)
    print(distance)
    # 369.461357
