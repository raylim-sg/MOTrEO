# This code contains the source-to-target mapping procedure of the MOTrEO+Ms for permutation-encoded vehicle route optimization tasks

import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from MOEA_routing_operators import index_of
import copy




def centroid_alignment(s_dim, dim, s_x, s_y, t_x, t_y):
    s_x_aligned = []
    s_y_aligned = []
    s_x_mean = float((sum(s_x) - s_x[0]) / s_dim)
    s_y_mean = float((sum(s_y) - s_y[0]) / s_dim)
    t_x_mean = float((sum(t_x) - t_x[0]) / dim)
    t_y_mean = float((sum(t_y) - t_y[0]) / dim)
    d_x = s_x_mean - t_x_mean
    d_y = s_y_mean - t_y_mean

    if d_x > 0:
        delta_x = 0 - d_x
    else:
        delta_x = abs(d_x)

    if d_y > 0:
        delta_y = 0 - d_y
    else:
        delta_y = abs(d_y)

    for i in range(s_dim + 1):
        s_xa = s_x[i] + delta_x
        s_ya = s_y[i] + delta_y
        s_x_aligned.append(float(s_xa))
        s_y_aligned.append(float(s_ya))
    return s_x_aligned, s_y_aligned



def scaling_PC_alignment(task, s_dim, dim, s_x, s_y, t_x, t_y):
    s_x_aligned = []
    s_y_aligned = []
    t_x_aligned = []
    t_y_aligned = []
    s_vec = copy.deepcopy(task.vec)

    # Source alignment
    s_nodes = []
    for i in range(0, s_dim + 1):
        s_nodes.append(str(i))

    xs = ['x_s']
    ys = ['y_s']

    s_graph = pd.DataFrame(index=s_nodes, columns=[*xs, *ys])

    s = 0
    for node in s_graph.index:
        s_graph.loc[node, 'x_s'] = float(s_x[s])
        s_graph.loc[node, 'y_s'] = float(s_y[s])
        s += 1

    s_graph_scaled = preprocessing.scale(s_graph)

    s_pca = PCA()
    s_pca.fit(s_graph_scaled)
    s_graph_data = s_pca.transform(s_graph_scaled)

    for k in range(0, s_dim + 1):
        s_x_data = s_graph_data[k][0]
        s_y_data = s_graph_data[k][1]
        s_graph_data[k][0] = s_vec[0] * s_x_data
        s_graph_data[k][1] = s_vec[1] * s_y_data

    for i in range(s_dim + 1):
        s_x_aligned.append(float(s_graph_data[i][0]))
        s_y_aligned.append(float(s_graph_data[i][1]))


    # Target alignment
    t_nodes = []
    t_nodes.append(int(0))

    for i in range(dim):
        t_nodes.append(str(i))

    xt = ['x_t']
    yt = ['y_t']

    t_graph = pd.DataFrame(index=t_nodes, columns=[*xt, *yt])

    t = 0
    for node in t_graph.index:
        t_graph.loc[node, 'x_t'] = float(t_x[t])
        t_graph.loc[node, 'y_t'] = float(t_y[t])
        t += 1

    t_graph_scaled = preprocessing.scale(t_graph)

    t_pca = PCA()
    t_pca.fit(t_graph_scaled)
    t_graph_data = t_pca.transform(t_graph_scaled)

    for i in range(dim + 1):
        t_x_aligned.append(float(t_graph_data[i][0]))
        t_y_aligned.append(float(t_graph_data[i][1]))
    return s_x_aligned, s_y_aligned, t_x_aligned, t_y_aligned



def source_target_distance_matrix(s_dim, dim, s_x, s_y, t_x, t_y):
    distance_matrix = [[0 for _ in range(s_dim + 1)] for _ in range(dim + 1)]

    for i in range(1, dim + 1):
        target_x = float(t_x[i])
        target_y = float(t_y[i])

        for j in range(1, s_dim + 1):
            source_x = float(s_x[j])
            source_y = float(s_y[j])
            distance = math.sqrt((target_x - source_x) **2 + (target_y - source_y) **2)
            distance_matrix[i][j] = distance
    return distance_matrix



def copy_target_index(task, s_dim, dim, distance_matrix, s_data):

    minimum_distances = []
    indices1 = []
    distances = []
    indices2 = [0 for _ in range(dim)]

    for j in range(1, s_dim + 1):
        distance_list = []
        for i in range(1, dim + 1):
            distance_list.append(float(distance_matrix[i][j]))
        distances.append(distance_list)

    if s_dim > dim:
        for i in range(dim):
            min_distance = float(min(distances[i]))
            minimum_distances.append(min_distance)
            index = index_of(min_distance, distances[i])
            indices1.append(index + 1)

        for j in range(dim):
            indices2[indices1[j] - 1] = j + 1

    else:
        for i in range(s_dim):
            min_distance = float(min(distances[i]))
            minimum_distances.append(min_distance)
            index = index_of(min_distance, distances[i])
            indices1.append(index + 1)

        for j in range(s_dim):
            indices2[indices1[j] - 1] = j + 1


    not_sampled = [int(i) for i in range(1, s_dim + 1)]
    indices3 = []

    if s_dim > dim:
        for i in range(dim):
            node = indices2[i]

            if not_sampled[node - 1] == 9999:
                indices3.append(9999)
            else:
                indices3.append(not_sampled[node - 1])
                not_sampled[node - 1] = 9999

        for j in range(dim):
            if indices3[j] == 9999:
                for k in range(s_dim):
                    if not_sampled[k] != 9999:
                        indices3[j] = not_sampled[k]
                        not_sampled[k] = 9999
                        break

        indices4 = indices3[:]

        for i in range(dim, s_dim):
            for k in range(s_dim):
                if not_sampled[k] != 9999:
                    indices4.append(not_sampled[k])
                    not_sampled[k] = 9999
                    break

    else:
        for i in range(s_dim):
            node = indices2[i]

            if not_sampled[node - 1] == 9999:
                indices3.append(9999)
            else:
                indices3.append(not_sampled[node - 1])
                not_sampled[node - 1] = 9999

        for j in range(s_dim):
            if indices3[j] == 9999:
                for k in range(s_dim):
                    if not_sampled[k] != 9999:
                        indices3[j] = not_sampled[k]
                        not_sampled[k] = 9999
                        break

        indices4 = indices3[:]


    transformed_s_data = copy.deepcopy(s_data)

    for i in range(task.pop_size_source):
        for j in range(s_dim):
            for k in range(s_dim):
                if s_data[i][k] == indices4[j] - 1:
                    transformed_s_data[i][k] = j
                    break
    return transformed_s_data



def map(task, s_dim, dim, s_x, s_y, t_x, t_y, s_data):

    s_x1, s_y1, t_x1, t_y1 = scaling_PC_alignment(task, s_dim, dim, s_x, s_y, t_x, t_y)

    s_x2, s_y2 = centroid_alignment(s_dim, dim, s_x1, s_y1, t_x1, t_y1)

    distance_matrix = source_target_distance_matrix(s_dim, dim, s_x2, s_y2, t_x1, t_y1)

    transformed_s_data = copy_target_index(task, s_dim, dim, distance_matrix, s_data)

    return transformed_s_data
