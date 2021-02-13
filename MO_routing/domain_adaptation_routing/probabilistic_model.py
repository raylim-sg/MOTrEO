# This file contains the codes for probabilistic model-based search and transfer

import numpy as np
import random




def edge_histogram(dim, parents, parent_size):
    edge_historam_model = [[0 for _ in range(dim)] for _ in range(dim)]
    edge_histogram_model_noisy = [[0 for _ in range(dim)] for _ in range(dim)]
    tmp = [[0 for _ in range(dim)] for _ in range(dim)]
    noise_size = int(0.1 * parent_size)
    noise = [list(np.random.permutation(dim)) for _ in range(noise_size)]
    bratio = 0.004

    exp = (2 * parent_size * bratio) / (dim - 1)

    for t in range(parent_size):
        for i in range(dim):
            u = parents[t][i]
            v = parents[t][(i + 1) % dim]
            tmp[u][v] += 1

    for i in range(dim):
        for j in range(dim):
            if (i != j):
                edge_historam_model[i][j] = (tmp[i][j] + tmp[j][i] + exp) / (parent_size)

    for t in range(0, noise_size):
        for i in range(dim):
            u = noise[t][i]
            v = noise[t][(i + 1) % dim]
            tmp[u][v] += 1

    for i in range(dim):
        for j in range(dim):
            if (i != j):
                edge_histogram_model_noisy[i][j] = (tmp[i][j] + tmp[j][i] + exp) / (parent_size + noise_size)
    return edge_historam_model, edge_histogram_model_noisy



def roulette_wheel(dim, probability_array):
    for i in range(1, dim):
        probability_array[i] += probability_array[i - 1]
    r = random.random()

    if r < probability_array[0]:
        return 0

    for i in range(0, dim - 1):
        if (r >= probability_array[i] and r < probability_array[i + 1]):
            return i + 1
    return 0



def sample_solution(dim, edge_histogram_model):
    solution = [0] * dim
    mark = [0] * dim
    solution[0] = 0
    mark[solution[0]] = 1

    for i in range(1, dim):
        probability_array = [0] * dim
        u = solution[i - 1]
        sum = 0
        for j in range(dim):
            if (mark[j] == 0):
                probability_array[j] = edge_histogram_model[u][j]
                sum += probability_array[j]
        for j in range(dim):
            probability_array[j] = probability_array[j]/sum
        solution[i] = roulette_wheel(dim, probability_array)
        mark[solution[i]] = 1
    return solution



def add_variables(dim_source, dim_new, edge_histogram_model, edge_histogram_model_noisy):
    new_model = [[0 for _ in range(dim_new)] for _ in range(dim_new)]
    new_model_noisy = [[0 for _ in range(dim_new)] for _ in range(dim_new)]
    for i in range(dim_source):
        for j in range(dim_source):
            new_model[i][j] = edge_histogram_model[i][j]
            new_model_noisy[i][j] = edge_histogram_model_noisy[i][j]

    for i in range(dim_source, dim_new):
        for j in range(dim_new):
            new_model[i][j] = np.random.uniform(0.00001, 0.00009)
            new_model_noisy[i][j] = np.random.uniform(0.00001, 0.00009)
            new_model[j][i] = np.random.uniform(0.00001, 0.00009)
            new_model_noisy[j][i] = np.random.uniform(0.00001, 0.00009)
    return new_model, new_model_noisy



def remove_variables(dim_source, dim_new, edge_historam_model, edge_historam_model_noisy):
    new_model = [[0 for _ in range(dim_new)] for _ in range(dim_new)]
    new_model_noisy = [[0 for _ in range(dim_source)] for _ in range(dim_source)]
    for i in range(dim_new):
        sum1 = 0
        sum2 = 0
        for j in range(dim_new):
            new_model[i][j] = edge_historam_model[i][j]
            new_model_noisy[i][j] = edge_historam_model_noisy[i][j]
        for k in range(dim_new, dim_source):
            sum1 += edge_historam_model[i][k]
            sum2 += edge_historam_model_noisy[i][k]
        prob1 = sum1 / dim_new
        prob2 = sum2 / dim_new
        for j in range(dim_new):
            new_model[i][j] += prob1
            new_model_noisy[i][j] += prob2
    return new_model, new_model_noisy



def sample_mixture_model(dim, pop_size, mixture_model, total_models, transfer_coefficients):
    solutions_per_model = []
    for i in range(total_models):
        solutions_per_model.append(int(np.ceil(transfer_coefficients[i] * pop_size)))
    tmp_candidates = []
    for i in range(total_models):
        if (solutions_per_model[i] == 0):
            continue
        for k in range(solutions_per_model[i]):
            sol = sample_solution(dim, mixture_model[i])
            tmp_candidates.append(sol)
    if len(tmp_candidates) < pop_size:
        gap = pop_size - len(tmp_candidates)
        for i in range(gap):
            add_sol = sample_solution(dim, mixture_model[total_models - 1])
            tmp_candidates.append(add_sol)
    return tmp_candidates[:pop_size]



def mutate_mixture_model(coefficients, total_models):
    modi_coefficients = np.add(coefficients, np.random.normal(0, 0.01, total_models))
    new_coefficients = coefficients[:]
    for i in range(total_models):
        if modi_coefficients[i] < 0:
            modi_coefficients[i] = 0
    pusum = sum(modi_coefficients)
    if (pusum == 0):
        transfer_coefficients = [0] * total_models
        transfer_coefficients[total_models - 1] = 1
    else:
        new_coefficients = modi_coefficients / pusum
    return new_coefficients



def EM(mixture_probability_matrix, parent_size, total_models, iter):
    coefficients = [1 / total_models] * total_models
    for t in range(0, iter):
        sum = [0] * parent_size
        for i in range(0, parent_size):
            for k in range(0, total_models):
                sum[i] += coefficients[k] * mixture_probability_matrix[i][k]
        for k in range(0, total_models):
            tmp = 0
            for i in range(0, parent_size):
                if (sum[i] != 0):
                    tmp += (coefficients[k] * mixture_probability_matrix[i][k]) / sum[i]
            coefficients[k] = tmp / parent_size
    return coefficients



def eval_PDF(dim, edge_histogram_model_noisy, solution):
    probability = 1
    for i in range(dim):
        u = solution[i]
        v = solution[(i + 1) % dim]
        probability = probability * edge_histogram_model_noisy[u][v]
    return probability



def build_mixture_probability_matrix(dim, parents, v_fold, parent_size, total_models, source_models_noisy):
    test_size = int(parent_size / v_fold)
    training_size = int(parent_size - test_size)
    index = np.random.permutation(parent_size)
    matrix = [[0 for _ in range(total_models)] for _ in range(parent_size)]
    for k in range(0, v_fold):
        training_data = []
        start_index = k * test_size
        end_index = (k + 1) * test_size
        for i in range(0, start_index):
            training_data.append(parents[index[i]])
        for i in range(end_index, parent_size):
            training_data.append(parents[index[i]])
        target_model_intermediate = edge_histogram(dim, training_data, training_size)
        for i in range(start_index, end_index):
            for j in range(0, total_models - 1):
                matrix[index[i]][j] = eval_PDF(dim, source_models_noisy[j], parents[index[i]])
            matrix[index[i]][total_models - 1] = eval_PDF(dim, target_model_intermediate[1], parents[index[i]])
    return matrix



def build_mixture_model(dim, parents, parent_size, total_models, source_models, source_models_noisy):
    v_fold = 5
    mixture_probability_matrix = build_mixture_probability_matrix(dim, parents, v_fold, parent_size, total_models, source_models_noisy)
    mixture_model = []
    iterations = 100
    coefficients = EM(mixture_probability_matrix, parent_size, total_models, iterations)
    for i in range(0, total_models - 1):
        mixture_model.append(source_models[i])
    target_model = edge_histogram(dim, parents, parent_size)
    mixture_model.append(target_model[0])
    return mixture_model, coefficients
