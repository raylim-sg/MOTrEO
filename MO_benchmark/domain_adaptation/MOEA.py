# This code contains multi-objective evolutionary algorithms (MOEAs) for continuous optimization

import numpy as np
import random
from MOEA_operators import SBX_crossover, polynomial_mutation, binary_tournament, crowding_distance, \
    sort_distance, fast_non_dominated_sort, check_bounds, \
    binary_tournament_3D, crowding_distance_3D, fast_non_dominated_sort_3D
from domain_adaptation.probabilisticModel import MixtureModel
from domain_adaptation.nlmap import learn_map, solution_transform, model_transform
import copy
import math




# Original NSGA-II (2 objectives)
def NSGA2(function1, function2, max_gen, pop_size, dim, pf_target):

    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.

    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: approximated target Pareto front
    """

    gen_no = 0
    igd_values = []

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)
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

    # Initialize search population
    solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("NSGA-II Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        parent_front_f1 = []
        parent_front_f2 = []
        for i in range(len(parent_front_f11)):
            parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
            parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generating offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            a1 = random.randint(0, pop_size - 1)
            a2 = random.randint(0, pop_size - 1)
            a = binary_tournament(a1, a2, function1_values[:], function2_values[:])
            b1 = random.randint(0, pop_size - 1)
            b2 = random.randint(0, pop_size - 1)
            b = binary_tournament(b1, b2, function1_values[:], function2_values[:])
            c1, c2 = SBX_crossover(solution[a], solution[b])
            c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
            solution2.append(c1_mutated)
            solution2.append(c2_mutated)
        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(function1(solution2[i]))
            function2_values2.append(function2(solution2[i]))
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

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return igd_values




# Original NSGA-II (3 objectives)
def NSGA2_3D(function1, function2, function3, max_gen, pop_size, dim, pf_target):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.

    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    function3: third objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: target Pareto front
    """

    gen_no = 0
    igd_values = []

    n_obj = 3
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)
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
    pf_ref_f3 = []
    for i in range(pf_ref_len):
        pf_ref_f1.append(pf_ref[i][0])
        pf_ref_f2.append(pf_ref[i][1])
        pf_ref_f3.append(pf_ref[i][2])


    # Initialize search population
    solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]
    function3_values = [function3(solution[i]) for i in range(0, pop_size)]

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort_3D(function1_values[:], function2_values[:], function3_values[:])
        print("NSGA-II Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        parent_front_f33 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])
            parent_front_f33.append(function3_values[index])

        # Compute IGD values
        parent_front_f1 = []
        parent_front_f2 = []
        parent_front_f3 = []
        for i in range(len(parent_front_f11)):
            parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
            parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
            parent_front_f3.append((parent_front_f33[i] - min_obj[2]) / (max_obj[2] - min_obj[2]))
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0) \
                        + pow(parent_front_f3[j] - pf_ref_f3[i], 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generating offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            a1 = random.randint(0, pop_size - 1)
            a2 = random.randint(0, pop_size - 1)
            a = binary_tournament_3D(a1, a2, function1_values[:], function2_values[:], function3_values[:])
            b1 = random.randint(0, pop_size - 1)
            b2 = random.randint(0, pop_size - 1)
            b = binary_tournament_3D(b1, b2, function1_values[:], function2_values[:], function3_values[:])
            c1, c2 = SBX_crossover(solution[a], solution[b])
            c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
            solution2.append(c1_mutated)
            solution2.append(c2_mutated)
        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        function3_values2 = function3_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(function1(solution2[i]))
            function2_values2.append(function2(solution2[i]))
            function3_values2.append(function3(solution2[i]))
        non_dominated_sorted_solution2 = fast_non_dominated_sort_3D(function1_values2[:], function2_values2[:], function3_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance_3D(function1_values2[:], function2_values2[:], function3_values2[:], non_dominated_sorted_solution2[i][:]))

        # Environmental selection
        new_solution = []
        function1_values = []
        function2_values = []
        function3_values = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2[i].sort()
            front = sort_distance(non_dominated_sorted_solution2[i], crowding_distance_values2[i])
            front.reverse()
            for index in front:
                new_solution.append(solution2[index])
                function1_values.append(function1_values2[index])
                function2_values.append(function2_values2[index])
                function3_values.append(function3_values2[index])
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")

    return igd_values




# Adaptive Model-Based Transfer Evolutionary Algorithm (AMTEA)
def AMTEA(function1, function2, max_gen, pop_size, dim, pf_target, source_model_list=None,
            tr_int=None, verbose=False):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.

    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: target Pareto front
    source_model_list: list storing available source model(s)
    tr_int: predefined transfer interval
    """

    if tr_int is None:
        tr_int = 2
    sm_list = copy.deepcopy(source_model_list)

    gen_no = 0
    igd_values = []
    transfer_coefficients = []

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)
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


    # Initialize search population
    solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]

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
        parent_front_f1 = []
        parent_front_f2 = []
        for i in range(len(parent_front_f11)):
            parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
            parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generating offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            if (gen_no + 1) % tr_int == 0:
                # offspring generated by sampling the target probabilistic mixture model at specified transfer intervals
                for i in range(len(sm_list)):
                    if sm_list[i].dim < dim or sm_list[i].dim > dim:
                        sm_list[i].modify(dim)

                mm = MixtureModel(sm_list)
                mm.createTable(np.array(solution), True, 'mvarnorm')
                mm.EMstacking()
                mm.mutate()
                coefficients = np.around(mm.alpha, decimals=5)
                transfer_coefficients.append(coefficients[0])
                if verbose: print("Transfer coefficients = ", coefficients)

                offspring_A = mm.sample(pop_size)
                for i in range(0, len(offspring_A)):
                    offspring_B = check_bounds(offspring_A[i])
                    solution2.append(offspring_B)

            else: # Offspring generated via standard reproduction during non-transfer intervals
                a1 = random.randint(0, pop_size - 1)
                a2 = random.randint(0, pop_size - 1)
                a = binary_tournament(a1, a2, function1_values[:], function2_values[:])
                b1 = random.randint(0, pop_size - 1)
                b2 = random.randint(0, pop_size - 1)
                b = binary_tournament(b1, b2, function1_values[:], function2_values[:])
                c1, c2 = SBX_crossover(solution[a], solution[b])
                c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
                solution2.append(c1_mutated)
                solution2.append(c2_mutated)
        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(function1(solution2[i]))
            function2_values2.append(function2(solution2[i]))
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

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")


    return igd_values, transfer_coefficients




# NSGA-II with the injection of transformed source solutions
def NSGA2_Ms(function1, function2, max_gen, pop_size, dim, pf_target, source_data=None, source_model_list=None,
                tr_int=None, reg=None):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.

    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: target Pareto front
    source_data: solutions of previously encountered (optimized) source problem
    source_model_list: list storing available source model(s)
    tr_int: predefined transfer interval
    reg: regularization coefficient of neural network parameters when learning source-to-target mapping Ms
    """

    if tr_int is None:
        tr_int = 2
    if pop_size <= 50:
        target_popsize_smaller = True
    else:
        target_popsize_smaller = False

    gen_no = 0
    igd_values = []
    source_data_size = len(source_data[0])

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)
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


    # Initialize search population
    solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("NSGA-II+Ms Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        parent_front_f1 = []
        parent_front_f2 = []
        for i in range(len(parent_front_f11)):
            parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
            parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generating offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            a1 = random.randint(0, pop_size - 1)
            a2 = random.randint(0, pop_size - 1)
            a = binary_tournament(a1, a2, function1_values[:], function2_values[:])
            b1 = random.randint(0, pop_size - 1)
            b2 = random.randint(0, pop_size - 1)
            b = binary_tournament(b1, b2, function1_values[:], function2_values[:])
            c1, c2 = SBX_crossover(solution[a], solution[b])
            c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
            solution2.append(c1_mutated)
            solution2.append(c2_mutated)
        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(function1(solution2[i]))
            function2_values2.append(function2(solution2[i]))

        if (gen_no + 1) % tr_int == 0:
            # Learn non-linear source-to-target mapping at specified transfer intervals
            if target_popsize_smaller:
                generational_solutions = np.array(solution)
                source_data_subpop = []
                source_data_indices = random.sample(range(0, source_data_size), pop_size)
                source_data_indices.sort()
                for i in range(pop_size):
                    source_data_subpop.append(source_data[gen_no][source_data_indices[i]])
                source_data_subpop = np.array(source_data_subpop)
                map = learn_map(source_data_subpop, generational_solutions, reg)
            else:
                target_data_subpop = []
                target_data_indices = random.sample(range(0, pop_size), source_data_size)
                target_data_indices.sort()
                for i in range(source_data_size):
                    target_data_subpop.append(solution[target_data_indices[i]])
                generational_solutions = np.array(target_data_subpop)
                map = learn_map(source_data[gen_no], generational_solutions, reg)
            # Perform solution transfer
            transfer_size = tr_int
            source_solutions = source_model_list[0].sample(transfer_size)
            indx = random.sample(range(0, 2 * pop_size), transfer_size)
            for i in range(transfer_size):
                transferred_sol = solution_transform(source_solutions[i], map)
                solution2[indx[i]] = transferred_sol
                function1_values2[indx[i]] = function1(transferred_sol)
                function2_values2[indx[i]] = function2(transferred_sol)

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

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")


    return igd_values



# Proposed multi-objective TrEO algorithm with source-to-target search space mapping Ms (MOTrEO+Ms)
# Equipped with a probabilistic model-based mechanism to capture source-target similarity
def MOTrEO_Ms(function1, function2, max_gen, pop_size, dim, pf_target, source_data=None, source_model_list=None,
               tr_int=None, reg=None, verbose=False):

    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.

    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: target Pareto front
    source_data: solutions of previously encountered (optimized) source problem
    source_model_list: list storing available source model(s)
    tr_int: predefined transfer interval
    reg: regularization coefficient of neural network parameters when learning source-to-target mapping Ms
    """

    if tr_int is None:
        tr_int = 2
    sm_list = copy.deepcopy(source_model_list)
    if pop_size <= 50:
        target_popsize_smaller = True
    else:
        target_popsize_smaller = False

    gen_no = 0
    igd_values = []
    transfer_coefficients = []
    source_data_size = len(source_data[0])

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)
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


    # Initialize search population
    solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]

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
        parent_front_f1 = []
        parent_front_f2 = []
        for i in range(len(parent_front_f11)):
            parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
            parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generate offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            if (gen_no + 1) % tr_int == 0:
                # Learn non-linear source-to-target mapping at specified transfer intervals
                if target_popsize_smaller:
                    generational_solutions = np.array(solution)
                    source_data_subpop = []
                    source_data_indices = random.sample(range(0, source_data_size), pop_size)
                    source_data_indices.sort()
                    for i in range(pop_size):
                        source_data_subpop.append(source_data[gen_no][source_data_indices[i]])
                    source_data_subpop = np.array(source_data_subpop)
                    map = learn_map(source_data_subpop, generational_solutions, reg)
                else:
                    target_data_subpop = []
                    target_data_indices = random.sample(range(0, pop_size), source_data_size)
                    target_data_indices.sort()
                    for i in range(source_data_size):
                        target_data_subpop.append(solution[target_data_indices[i]])
                    generational_solutions = np.array(target_data_subpop)
                    map = learn_map(source_data[gen_no], generational_solutions, reg)

                # offspring generated by sampling the target probabilistic mixture model
                new_models = [model_transform(sm_list[0], map)]
                mm = MixtureModel(new_models)
                mm.createTable(np.array(solution), True, 'mvarnorm')
                mm.EMstacking()
                mm.mutate()
                coefficients = np.around(mm.alpha, decimals=5)
                transfer_coefficients.append(coefficients[0])
                if verbose: print("Transfer coefficients = ", coefficients)

                offspring_A = mm.sample(pop_size)
                for i in range(0, len(offspring_A)):
                    offspring_B = check_bounds(offspring_A[i])
                    solution2.append(offspring_B)

            else:
                # Offspring creation via standard reproduction during non-transfer intervals
                a1 = random.randint(0, pop_size - 1)
                a2 = random.randint(0, pop_size - 1)
                a = binary_tournament(a1, a2, function1_values[:], function2_values[:])
                b1 = random.randint(0, pop_size - 1)
                b2 = random.randint(0, pop_size - 1)
                b = binary_tournament(b1, b2, function1_values[:], function2_values[:])
                c1, c2 = SBX_crossover(solution[a], solution[b])
                c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
                solution2.append(c1_mutated)
                solution2.append(c2_mutated)
        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(function1(solution2[i]))
            function2_values2.append(function2(solution2[i]))
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

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")


    return igd_values, transfer_coefficients




# Multi-objective TrEO algorithm which transfers solutions from a random source model (RSM) (MOTrEO (RSM))
# The extent of solution transfers is according to the transfer coefficients obtained in the MOTrEO+Ms
def MoTrEO_RSM(function1, function2, max_gen, pop_size, dim, pf_target, tr_int=None,
                transfer_coefficients=None):
    """
    By default, this algorithm encodes each variable in the range [0,1]. Decoding step is thus needed during evaluations.

    ## Description of input arguments ##
    function1: first objective function name/handle passed for solution evaluation
    function2: second objective function name/handle passed for solution evaluation
    max_gen: maximum number of generations of EA
    pop_size: population size of EA
    dim: search space dimensionality
    pf_target: target Pareto front
    tr_int: predefined transfer interval
    transfer_coefficients: coefficients obtained in the MOTrEO+Ms used to determine the extent of knowledge transfers in the MOTrEO(RSM)
    """

    if tr_int is None:
        tr_int = 2

    gen_no = 0
    igd_values = []
    transfer_counter = 0

    n_obj = 2
    pf_ref = copy.deepcopy(pf_target)
    pf_ref_len = len(pf_ref)
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


    # Initialize search population
    solution = [[random.random() for _ in range(dim)] for _ in range(0, pop_size)]
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]

    while (gen_no < max_gen):
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        print("MOTrEO (RSM) Output for Generation ", gen_no, " :")
        parent_front_f11 = []
        parent_front_f22 = []
        non_dominated_sorted_solution[0].sort()
        for index in non_dominated_sorted_solution[0]:
            parent_front_f11.append(function1_values[index])
            parent_front_f22.append(function2_values[index])

        # Compute IGD values
        parent_front_f1 = []
        parent_front_f2 = []
        for i in range(len(parent_front_f11)):
            parent_front_f1.append((parent_front_f11[i] - min_obj[0]) / (max_obj[0] - min_obj[0]))
            parent_front_f2.append((parent_front_f22[i] - min_obj[1]) / (max_obj[1] - min_obj[1]))
        sum_dist = 0
        for i in range(pf_ref_len):
            min_dist = math.inf
            for j in range(len(parent_front_f1)):
                dist2 = pow(parent_front_f1[j] - pf_ref_f1[i], 2.0) + pow(parent_front_f2[j] - pf_ref_f2[i], 2.0)
                dist = math.sqrt(dist2)
                if dist < min_dist:
                    min_dist = dist
            sum_dist += min_dist
        igd = sum_dist / pf_ref_len
        igd_values.append(igd)
        print('IGD = ', igd)

        # Generating offsprings
        solution2 = solution[:]
        while (len(solution2) < 2 * pop_size):
            a1 = random.randint(0, pop_size - 1)
            a2 = random.randint(0, pop_size - 1)
            a = binary_tournament(a1, a2, function1_values[:], function2_values[:])
            b1 = random.randint(0, pop_size - 1)
            b2 = random.randint(0, pop_size - 1)
            b = binary_tournament(b1, b2, function1_values[:], function2_values[:])
            c1, c2 = SBX_crossover(solution[a], solution[b])
            c1_mutated, c2_mutated = polynomial_mutation(c1, c2)
            solution2.append(c1_mutated)
            solution2.append(c2_mutated)

        function1_values2 = function1_values[:]
        function2_values2 = function2_values[:]
        for i in range(pop_size, 2 * pop_size):
            function1_values2.append(function1(solution2[i]))
            function2_values2.append(function2(solution2[i]))

        # Knowledge transfers from a RSM based on the transfer coefficients obtained in the MOTrEO+Ms
        if (gen_no + 1) % tr_int == 0:
            offspring_sample_size = np.ceil(pop_size * transfer_coefficients[transfer_counter]).astype(int)
            if offspring_sample_size >= 1:
                indx = random.sample(range(0, 2 * pop_size), offspring_sample_size)
                for i in range(offspring_sample_size):
                    transferred_sol = [random.uniform(0, 1) for _ in range(dim)]
                    solution2[indx[i]] = transferred_sol
                    function1_values2[indx[i]] = function1(transferred_sol)
                    function2_values2[indx[i]] = function2(transferred_sol)
            transfer_counter += 1

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

        solution = new_solution[:]
        gen_no = gen_no + 1
        print("\n")


    return igd_values
