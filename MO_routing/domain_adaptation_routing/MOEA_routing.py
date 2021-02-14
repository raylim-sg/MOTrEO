# This code contains multi-objective evolutionary algorithms (MOEAs)
# The MOEAs are for solving the multi-objective drone routing problems detailed in "MO_routing_tasks.pdf"

import numpy as np
from MOEA_routing_operators import selection, fast_non_dominated_sort, crowding_distance, sort_distance, filtration, \
    optimizedCX, swap_node, mutation
from domain_adaptation_routing.probabilistic_model import edge_histogram, sample_solution, remove_variables, add_variables,\
    build_mixture_model, mutate_mixture_model, sample_mixture_model
from domain_adaptation_routing import source_to_target_mapping
import copy
import math




def NSGA2_routing(task_function, max_gen, pop_size, dim, pc, pm, pf_target):

    """
    By default, this algorithm encodes each solution as a permutation of integers.
    Solution decoding is thus needed during evaluations.

    ## Description of input arguments ##
    task_function: objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality (i.e., number of customer nodes)
    pc: crossover rate
    pm: mutation rate
    pf_target: approximated target Pareto front
    """

    gen_no = 0
    igd_values = []
    parent_size = int(pop_size / 2)

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)

    if pf_ref_len > 1:
        max_obj = [0] * n_obj
        min_obj = [0] * n_obj

        for i in range(n_obj):
            pf_ref = sorted(pf_ref, key=lambda obj: obj[i])
            max_obj[i] = pf_ref[pf_ref_len - 1][i]
            min_obj[i] = pf_ref[0][i]
        for i in range(pf_ref_len):
            for j in range(n_obj):
                pf_ref[i][j] = (pf_ref[i][j] - min_obj[j]) / (max_obj[j] - min_obj[j])
        pf_ref_f1 = []
        pf_ref_f2 = []
        for i in range(pf_ref_len):
            pf_ref_f1.append(pf_ref[i][0])
            pf_ref_f2.append(pf_ref[i][1])
    else:
        pf_ref_f1 = [pf_ref[0][0]]
        pf_ref_f2 = [pf_ref[0][1]]


    # Initialize search population
    solution = [list(np.random.permutation(dim)) for _ in range(pop_size)]
    function1_values = [0 for _ in range(pop_size)]
    function2_values = [0 for _ in range(pop_size)]
    for i in range(pop_size):
        function1_values[i], function2_values[i] = task_function(solution[i])

    while (gen_no < max_gen):
        if (gen_no > 0 and gen_no % 50 == 0):
            solution = filtration(dim, solution)
            function1_values = [0 for _ in range(pop_size)]
            function2_values = [0 for _ in range(pop_size)]
            for i in range(pop_size):
                function1_values[i], function2_values[i] = task_function(solution[i])

        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("NSGA-II Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        if pf_ref_len > 1:
            parent_front_f1 = []
            parent_front_f2 = []
            for i in range(len(parent_front_f11)):
                parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
                parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        else:
            parent_front_f1 = parent_front_f11[:]
            parent_front_f2 = parent_front_f22[:]
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                v1 = parent_front_f1[j] - pf_ref_f1[i]
                v2 = parent_front_f2[j] - pf_ref_f2[i]
                value1 = max(v1, 0)
                value2 = max(v2, 0)
                dist2 = pow(value1, 2.0) + pow(value2, 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generate offsprings
        parents = selection(solution, parent_size, function1_values[:], function2_values[:])
        solution2 = solution[:]

        for j in range(int(pop_size / 2)):
            r = np.random.random()
            if (r < pc):
                if j != int(pop_size / 2) - 1:
                    offspring = optimizedCX(task_function, dim, parents[j], parents[j + 1])
                else:
                    offspring = optimizedCX(task_function, dim, parents[0], parents[j])
            else:
                if j != int(pop_size / 2) - 1:
                    offspring1 = swap_node(dim, parents[j])
                    offspring2 = swap_node(dim, parents[j + 1])
                    offspring = [offspring1, offspring2]
                else:
                    offspring1 = swap_node(dim, parents[0])
                    offspring2 = swap_node(dim, parents[j])
                    offspring = [offspring1, offspring2]
            offspring[0] = mutation(dim, offspring[0], pm)
            offspring[1] = mutation(dim, offspring[1], pm)
            solution2.append(offspring[0])
            solution2.append(offspring[1])

        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(0)
            function2_values2.append(0)
            function1_values2[i], function2_values2[i] = task_function(solution2[i])
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

        # Environmental selection
        new_solution = []
        function1_values = []
        function2_values =[]
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2[i].sort()
            front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
            front.reverse()
            for index in front:
                new_solution.append(solution2[index])
                function1_values.append(function1_values2[index])
                function2_values.append(function2_values2[index])
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        # Survival
        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return igd_values




def EHBSA_routing(task_function, max_gen, pop_size, dim, pf_target):

    """
    By default, this algorithm encodes each solution as a permutation of integers.
    Solution decoding is thus needed during evaluations.

    ## Description of input arguments ##
    task_function: objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality (i.e., number of customer nodes)
    pf_target: approximated target Pareto front
    """

    gen_no = 0
    igd_values = []
    parent_size = int(pop_size / 2)

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)

    if pf_ref_len > 1:
        max_obj = [0] * n_obj
        min_obj = [0] * n_obj

        for i in range(n_obj):
            pf_ref = sorted(pf_ref, key=lambda obj: obj[i])
            max_obj[i] = pf_ref[pf_ref_len - 1][i]
            min_obj[i] = pf_ref[0][i]
        for i in range(pf_ref_len):
            for j in range(n_obj):
                pf_ref[i][j] = (pf_ref[i][j] - min_obj[j]) / (max_obj[j] - min_obj[j])
        pf_ref_f1 = []
        pf_ref_f2 = []
        for i in range(pf_ref_len):
            pf_ref_f1.append(pf_ref[i][0])
            pf_ref_f2.append(pf_ref[i][1])
    else:
        pf_ref_f1 = [pf_ref[0][0]]
        pf_ref_f2 = [pf_ref[0][1]]


    # Initialize search population
    solution = [list(np.random.permutation(dim)) for _ in range(pop_size)]
    function1_values = [0 for _ in range(pop_size)]
    function2_values = [0 for _ in range(pop_size)]
    for i in range(pop_size):
        function1_values[i], function2_values[i] = task_function(solution[i])

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("EHBSA Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        if pf_ref_len > 1:
            parent_front_f1 = []
            parent_front_f2 = []
            for i in range(len(parent_front_f11)):
                parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
                parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        else:
            parent_front_f1 = parent_front_f11[:]
            parent_front_f2 = parent_front_f22[:]
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                v1 = parent_front_f1[j] - pf_ref_f1[i]
                v2 = parent_front_f2[j] - pf_ref_f2[i]
                value1 = max(v1, 0)
                value2 = max(v2, 0)
                dist2 = pow(value1, 2.0) + pow(value2, 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generate offsprings
        parents = selection(solution, parent_size, function1_values[:], function2_values[:])
        solution2 = solution[:]

        model = edge_histogram(dim, parents, parent_size)

        for j in range(pop_size):
            offspring = sample_solution(dim, model[0])
            solution2.append(offspring)

        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(0)
            function2_values2.append(0)
            function1_values2[i], function2_values2[i] = task_function(solution2[i])
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

        # Environmental selection
        new_solution = []
        function1_values = []
        function2_values = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2[i].sort()
            front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
            front.reverse()
            for index in front:
                new_solution.append(solution2[index])
                function1_values.append(function1_values2[index])
                function2_values.append(function2_values2[index])
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        # Survival
        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return igd_values




def AMTEA_routing(task, task_function, max_gen, pop_size, dim, pc, pm, tr_int, pf_target):

    """
    By default, this algorithm encodes each solution as a permutation of integers.
    Solution decoding is thus needed during evaluations.

    ## Description of input arguments ##
    task: the current task being optimized
    task_function: objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality (i.e., number of customer nodes)
    pc: crossover rate
    pm: mutation rate
    tr_int: predefined transfer interval
    pf_target: approximated target Pareto front
    """

    # Build source model(s)
    models = []
    models_noisy = []
    s1_data = copy.deepcopy(task.data_s1)
    model1 = edge_histogram(task.dim_source1, s1_data, task.pop_size_source)
    models.append(model1[0])
    models_noisy.append(model1[1])
    if task.number_of_sources > 1:
        s2_data = copy.deepcopy(task.data_s2)
        model2 = edge_histogram(task.dim_source2, s2_data, task.pop_size_source)
        models.append(model2[0])
        models_noisy.append(model2[1])
    if task.number_of_sources > 2:
        s3_data = copy.deepcopy(task.data_s3)
        model3 = edge_histogram(task.dim_source3, s3_data, task.pop_size_source)
        models.append(model3[0])
        models_noisy.append(model3[1])

    # Ensure that each source model's dimensionality matches that of the target model
    source_models = []
    source_models_noisy = []
    if task.dim_source1 > dim:
        new_model1, new_model1n = remove_variables(task.dim_source1, dim, models[0], models_noisy[0])
    elif task.dim_source1 < dim:
        new_model1, new_model1n = add_variables(task.dim_source1, dim, models[0], models_noisy[0])
    else:
        new_model1 = models[0]
        new_model1n = models_noisy[0]
    source_models.append(new_model1)
    source_models_noisy.append(new_model1n)
    if task.number_of_sources > 1:
        if task.dim_source2 > dim:
            new_model2, new_model2n = remove_variables(task.dim_source2, dim, models[1], models_noisy[1])
        elif task.dim_source2 < dim:
            new_model2, new_model2n = add_variables(task.dim_source2, dim, models[1], models_noisy[1])
        else:
            new_model2 = models[1]
            new_model2n = models_noisy[1]
        source_models.append(new_model2)
        source_models_noisy.append(new_model2n)
    if task.number_of_sources > 2:
        if task.dim_source3 > dim:
            new_model3, new_model3n = remove_variables(task.dim_source3, dim, models[2], models_noisy[2])
        elif task.dim_source1 < dim:
            new_model3, new_model3n = add_variables(task.dim_source3, dim, models[2], models_noisy[2])
        else:
            new_model3 = models[2]
            new_model3n = models_noisy[2]
        source_models.append(new_model3)
        source_models_noisy.append(new_model3n)

    igd_values = []
    parent_size = int(pop_size / 2)
    gen_no = 0
    total_models = len(models) + 1
    all_coefficients = []

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)

    if pf_ref_len > 1:
        max_obj = [0] * n_obj
        min_obj = [0] * n_obj

        for i in range(n_obj):
            pf_ref = sorted(pf_ref, key=lambda obj: obj[i])
            max_obj[i] = pf_ref[pf_ref_len - 1][i]
            min_obj[i] = pf_ref[0][i]
        for i in range(pf_ref_len):
            for j in range(n_obj):
                pf_ref[i][j] = (pf_ref[i][j] - min_obj[j]) / (max_obj[j] - min_obj[j])
        pf_ref_f1 = []
        pf_ref_f2 = []
        for i in range(pf_ref_len):
            pf_ref_f1.append(pf_ref[i][0])
            pf_ref_f2.append(pf_ref[i][1])
    else:
        pf_ref_f1 = [pf_ref[0][0]]
        pf_ref_f2 = [pf_ref[0][1]]


    # Initialize search population
    solution = [list(np.random.permutation(dim)) for _ in range(pop_size)]
    function1_values = [0 for _ in range(pop_size)]
    function2_values = [0 for _ in range(pop_size)]
    for i in range(pop_size):
        function1_values[i], function2_values[i] = task_function(solution[i])

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("AMTEA Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        if pf_ref_len > 1:
            parent_front_f1 = []
            parent_front_f2 = []
            for i in range(len(parent_front_f11)):
                parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
                parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        else:
            parent_front_f1 = parent_front_f11[:]
            parent_front_f2 = parent_front_f22[:]

        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                v1 = parent_front_f1[j] - pf_ref_f1[i]
                v2 = parent_front_f2[j] - pf_ref_f2[i]
                value1 = max(v1, 0)
                value2 = max(v2, 0)
                dist2 = pow(value1, 2.0) + pow(value2, 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generate offsprings
        parents = selection(solution, parent_size, function1_values[:], function2_values[:])
        solution2 = solution[:]

        # Perform probabilistic model-based transfer at specified transfer intervals
        if gen_no % tr_int == 0:
            mixture_model, coefficients = build_mixture_model(dim, parents, parent_size, total_models, source_models, source_models_noisy)
            transfer_coefficients = mutate_mixture_model(coefficients, total_models)
            all_coefficients.append(transfer_coefficients)
            offspring = sample_mixture_model(dim, pop_size, mixture_model, total_models, transfer_coefficients)
            for j in range(pop_size):
                solution2.append(offspring[j])
            print("Transfer coefficients = ", transfer_coefficients)

        # Perform standard reproduction during non-transfer intervals
        else:
            for j in range(int(pop_size / 2)):
                r = np.random.random()
                if (r < pc):
                    if j != int(pop_size / 2) - 1:
                        offspring = optimizedCX(task_function, dim, parents[j], parents[j + 1])
                    else:
                        offspring = optimizedCX(task_function, dim, parents[0], parents[j])
                else:
                    if j != int(pop_size / 2) - 1:
                        offspring1 = swap_node(dim, parents[j])
                        offspring2 = swap_node(dim, parents[j + 1])
                        offspring = [offspring1, offspring2]
                    else:
                        offspring1 = swap_node(dim, parents[0])
                        offspring2 = swap_node(dim, parents[j])
                        offspring = [offspring1, offspring2]
                offspring[0] = mutation(dim, offspring[0], pm)
                offspring[1] = mutation(dim, offspring[1], pm)
                solution2.append(offspring[0])
                solution2.append(offspring[1])

        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(0)
            function2_values2.append(0)
            function1_values2[i], function2_values2[i] = task_function(solution2[i])
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

        # Environmental selection
        new_solution = []
        function1_values = []
        function2_values = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2[i].sort()
            front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
            front.reverse()
            for index in front:
                new_solution.append(solution2[index])
                function1_values.append(function1_values2[index])
                function2_values.append(function2_values2[index])
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        # Survival
        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return igd_values, all_coefficients




def MOTrEO_Ms_routing(task, task_function, max_gen, pop_size, dim, pc, pm, tr_int, pf_target):

    """
    By default, this algorithm encodes each solution as a permutation of integers.
    Solution decoding is thus needed during evaluations.

    ## Description of input arguments ##
    task: the current task being optimized
    task_function: objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality (i.e., number of customer nodes)
    pc: crossover rate
    pm: mutation rate
    tr_int: predefined transfer interval
    pf_target: approximated target Pareto front
    """

    # Perform source-to-target search space mapping
    t_x = copy.deepcopy(task.x_t)
    t_y = copy.deepcopy(task.y_t)
    # Source1
    s1_x = copy.deepcopy(task.x_s1)
    s1_y = copy.deepcopy(task.y_s1)
    s1_data = copy.deepcopy(task.data_s1)
    s1_data_transformed = source_to_target_mapping.map(task, task.dim_source1, dim, s1_x, s1_y, t_x, t_y, s1_data)
    # Source2
    if task.number_of_sources > 1:
        s2_x = copy.deepcopy(task.x_s2)
        s2_y = copy.deepcopy(task.y_s2)
        s2_data = copy.deepcopy(task.data_s2)
        s2_data_transformed = source_to_target_mapping.map(task, task.dim_source2, dim, s2_x, s2_y, t_x, t_y, s2_data)
    # Source3
    if task.number_of_sources > 2:
        s3_x = copy.deepcopy(task.x_s3)
        s3_y = copy.deepcopy(task.y_s3)
        s3_data = copy.deepcopy(task.data_s3)
        s3_data_transformed = source_to_target_mapping.map(task, task.dim_source3, dim, s3_x, s3_y, t_x, t_y, s3_data)

    # Build source model(s)
    models = []
    models_noisy = []
    model1 = edge_histogram(task.dim_source1, s1_data_transformed, task.pop_size_source)
    models.append(model1[0])
    models_noisy.append(model1[1])
    if task.number_of_sources > 1:
        model2 = edge_histogram(task.dim_source2, s2_data_transformed, task.pop_size_source)
        models.append(model2[0])
        models_noisy.append(model2[1])
    if task.number_of_sources > 2:
        model3 = edge_histogram(task.dim_source3, s3_data_transformed, task.pop_size_source)
        models.append(model3[0])
        models_noisy.append(model3[1])

    # Ensure that each source model's dimensionality matches that of the target model
    source_models = []
    source_models_noisy = []
    if task.dim_source1 > dim:
        new_model1, new_model1n = remove_variables(task.dim_source1, dim, models[0], models_noisy[0])
    elif task.dim_source1 < dim:
        new_model1, new_model1n = add_variables(task.dim_source1, dim, models[0], models_noisy[0])
    else:
        new_model1 = models[0]
        new_model1n = models_noisy[0]
    source_models.append(new_model1)
    source_models_noisy.append(new_model1n)
    if task.number_of_sources > 1:
        if task.dim_source2 > dim:
            new_model2, new_model2n = remove_variables(task.dim_source2, dim, models[1], models_noisy[1])
        elif task.dim_source2 < dim:
            new_model2, new_model2n = add_variables(task.dim_source2, dim, models[1], models_noisy[1])
        else:
            new_model2 = models[1]
            new_model2n = models_noisy[1]
        source_models.append(new_model2)
        source_models_noisy.append(new_model2n)
    if task.number_of_sources > 2:
        if task.dim_source3 > dim:
            new_model3, new_model3n = remove_variables(task.dim_source3, dim, models[2], models_noisy[2])
        elif task.dim_source1 < dim:
            new_model3, new_model3n = add_variables(task.dim_source3, dim, models[2], models_noisy[2])
        else:
            new_model3 = models[2]
            new_model3n = models_noisy[2]
        source_models.append(new_model3)
        source_models_noisy.append(new_model3n)

    igd_values = []
    parent_size = int(pop_size / 2)
    gen_no = 0
    total_models = len(models) + 1
    all_coefficients = []

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)

    if pf_ref_len > 1:
        max_obj = [0] * n_obj
        min_obj = [0] * n_obj

        for i in range(n_obj):
            pf_ref = sorted(pf_ref, key=lambda obj: obj[i])
            max_obj[i] = pf_ref[pf_ref_len - 1][i]
            min_obj[i] = pf_ref[0][i]
        for i in range(pf_ref_len):
            for j in range(n_obj):
                pf_ref[i][j] = (pf_ref[i][j] - min_obj[j]) / (max_obj[j] - min_obj[j])
        pf_ref_f1 = []
        pf_ref_f2 = []
        for i in range(pf_ref_len):
            pf_ref_f1.append(pf_ref[i][0])
            pf_ref_f2.append(pf_ref[i][1])
    else:
        pf_ref_f1 = [pf_ref[0][0]]
        pf_ref_f2 = [pf_ref[0][1]]


    # Initialize search population
    solution = [list(np.random.permutation(dim)) for _ in range(pop_size)]
    function1_values = [0 for _ in range(pop_size)]
    function2_values = [0 for _ in range(pop_size)]
    for i in range(pop_size):
        function1_values[i], function2_values[i] = task_function(solution[i])

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("MOTrEO+Ms Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        if pf_ref_len > 1:
            parent_front_f1 = []
            parent_front_f2 = []
            for i in range(len(parent_front_f11)):
                parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
                parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        else:
            parent_front_f1 = parent_front_f11[:]
            parent_front_f2 = parent_front_f22[:]

        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                v1 = parent_front_f1[j] - pf_ref_f1[i]
                v2 = parent_front_f2[j] - pf_ref_f2[i]
                value1 = max(v1, 0)
                value2 = max(v2, 0)
                dist2 = pow(value1, 2.0) + pow(value2, 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generate offsprings
        parents = selection(solution, parent_size, function1_values[:], function2_values[:])
        solution2 = solution[:]

        # Perform probabilistic model-based transfer at specified transfer intervals
        if gen_no % tr_int == 0:
            mixture_model, coefficients = build_mixture_model(dim, parents, parent_size, total_models, source_models, source_models_noisy)
            transfer_coefficients = mutate_mixture_model(coefficients, total_models)
            all_coefficients.append(transfer_coefficients)
            offspring = sample_mixture_model(dim, pop_size, mixture_model, total_models, transfer_coefficients)
            for j in range(pop_size):
                solution2.append(offspring[j])
            print("Transfer coefficients = ", transfer_coefficients)

        # Perform standard reproduction during non-transfer intervals
        else:
            for j in range(int(pop_size / 2)):
                r = np.random.random()
                if (r < pc):
                    if j != int(pop_size / 2) - 1:
                        offspring = optimizedCX(task_function, dim, parents[j], parents[j + 1])
                    else:
                        offspring = optimizedCX(task_function, dim, parents[0], parents[j])
                else:
                    if j != int(pop_size / 2) - 1:
                        offspring1 = swap_node(dim, parents[j])
                        offspring2 = swap_node(dim, parents[j + 1])
                        offspring = [offspring1, offspring2]
                    else:
                        offspring1 = swap_node(dim, parents[0])
                        offspring2 = swap_node(dim, parents[j])
                        offspring = [offspring1, offspring2]
                offspring[0] = mutation(dim, offspring[0], pm)
                offspring[1] = mutation(dim, offspring[1], pm)
                solution2.append(offspring[0])
                solution2.append(offspring[1])

        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(0)
            function2_values2.append(0)
            function1_values2[i], function2_values2[i] = task_function(solution2[i])
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))

        # Environmental selection
        new_solution = []
        function1_values = []
        function2_values = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2[i].sort()
            front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
            front.reverse()
            for index in front:
                new_solution.append(solution2[index])
                function1_values.append(function1_values2[index])
                function2_values.append(function2_values2[index])
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        # Survival
        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return igd_values, all_coefficients
