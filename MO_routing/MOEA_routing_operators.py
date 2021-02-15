# This code contains the evolutionary search operators used by the MOEAs in MOEA_routing.py

import random
import numpy as np
import math




def binary_tournament(p1, p2, f1, f2):
    if (f1[p1] < f1[p2] and f2[p1] < f2[p2]) or (f1[p1] <= f1[p2] and f2[p1] < f2[p2]) or (f1[p1] < f1[p2] and f2[p1] <= f2[p2]):
        return p1
    elif (f1[p1] > f1[p2] and f2[p1] > f2[p2]) or (f1[p1] >= f1[p2] and f2[p1] > f2[p2]) or (f1[p1] > f1[p2] and f2[p1] >= f2[p2]):
        return p2
    else:
        r = random.random()
        if r < 0.5:
            return p1
        else:
            return p2



def selection(pop, parent_size, function1, function2):
    pop_size = len(pop)
    parent_pop = []
    for i in range(parent_size):
        p1 = np.random.randint(0, pop_size)
        p2 = np.random.randint(0, pop_size)
        parent = binary_tournament(p1, p2, function1, function2)
        parent_pop.append(pop[parent])
    return parent_pop



def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i



def index(non_dom_sol, min_distance, c_distance):
    for i in range(0, len(c_distance)):
        if c_distance[i] == min_distance:
            return non_dom_sol[i]



def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list



def sort_distance(non_dom_sol, c_distance):
    sorted_distance = []
    while len(sorted_distance) != len(c_distance):
        sorted_distance.append(index(non_dom_sol, min(c_distance), c_distance))
        c_distance[index_of(min(c_distance), c_distance)] = math.inf
    return sorted_distance



def fast_non_dominated_sort(f1, f2):
    S = [[] for i in range(0, len(f1))]
    front = [[]]
    n = [0 for i in range(0, len(f1))]
    rank = [0 for i in range(0, len(f1))]

    for p in range(0, len(f1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(f1)):
            if (f1[p] < f1[q] and f2[p] < f2[q]) or (f1[p] <= f1[q] and f2[p] < f2[q]) or (f1[p] < f1[q] and f2[p] <= f2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (f1[q] < f1[p] and f2[q] < f2[p]) or (f1[q] <= f1[p] and f2[q] < f2[p]) or (f1[q] < f1[p] and f2[q] <= f2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front



def crowding_distance(f1, f2, front):
    front.sort()
    distance = [0 for i in range(len(f1))]
    c_distance = []
    sorted1 = sort_by_values(front, f1[:])
    sorted2 = sort_by_values(front, f2[:])
    distance[sorted1[0]] = 999999999
    distance[sorted1[len(front) - 1]] = 999999999
    distance[sorted2[0]] = 999999999
    distance[sorted2[len(front) - 1]] = 999999999

    for k in range(1, len(front) - 1):
        distance[sorted1[k]] = distance[sorted1[k]] + (f1[sorted1[k + 1]] - f1[sorted1[k - 1]]) / (max(f1) - min(f1) + 0.001) + 0.001
    for k in range(1, len(front) - 1):
        distance[sorted2[k]] = distance[sorted2[k]] + (f2[sorted2[k + 1]] - f2[sorted2[k - 1]]) / (max(f2) - min(f2) + 0.001) + 0.001

    for j in distance:
        if j == 0:
            continue
        else:
            c_distance.append(j)
    return c_distance



def sols_equal(dim, sol1, sol2):
    for i in range(dim):
        if (sol1[i] != sol2[i]):
            return False
    return True



def filtration(dim, population):
    sorted(population, key=lambda population:population)
    new_population = []
    new_population.append(population[0])
    for i in range(1, len(population)):
        if sols_equal(dim, population[i-1], population[i]):
            new_sol = np.random.permutation(dim)
            new_population.append(new_sol)
        else:
            new_population.append(population[i])
    return new_population



class Pair:
    def __init__(self, _u, _v):
        self.u = _u
        self.v = _v



def optimizedCX(obj_function, dim, parent1, parent2):
    bipartiteMatrix = [[[0 for _ in range(2)] for _ in range(dim)] for _ in range(dim)]
    intersectionMatrix = [[0 for _ in range(dim)] for _ in range(dim)]
    unionMatrix = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        bipartiteMatrix[parent1[i]][i][0] = 1
        bipartiteMatrix[i][parent1[i]][1] = 1
        unionMatrix[parent1[i]][i] = 1
    for i in range(dim):
        if (bipartiteMatrix[parent2[i]][i][0] == 1):
            intersectionMatrix[parent2[i]][i] = 1
        unionMatrix[parent2[i]][i] = 1
        bipartiteMatrix[parent2[i]][i][0] = 1
        bipartiteMatrix[i][parent2[i]][1] = 1

    allCycles = []
    boo = [0] * dim
    numberCycle = 0
    for i in range(dim):
        if (boo[i] == 0):
            t = 0
            start = i
            cycle = []
            next = -1
            while (boo[start] == 0):
                boo[start] = 1
                for j in range(dim):
                    if (bipartiteMatrix[start][j][0] == 1 and j != next):
                        p = Pair(start, j)
                        cycle.append(p)
                        next = j
                        break
                for j in range(dim):
                    if (bipartiteMatrix[next][j][1] == 1 and j != start):
                        p = Pair(j, next)
                        cycle.append(p)
                        start = j
                        break
            if (len(cycle) > 1):
                numberCycle += 1
            allCycles.append(cycle)

    size = len(allCycles)
    if (size > 5):
        number = 32
    else:
        number = int(math.pow(2, numberCycle))
    offspring = []
    for i in range(number):
        sol = [0 for _ in range(dim)]
        for j in range(size):
            u = np.random.randint(0, 2)
            lenC = len(allCycles[j])
            if (lenC == 1):
                u = 0
            while (u < lenC):
                p = allCycles[j][u]
                sol[p.v] = p.u
                u = u + 2
        offspring.append(sol)

    function1_offspring = [0 for _ in range(number)]
    function2_offspring = [0 for _ in range(number)]
    for k in range(number):
        function1_offspring[k], function2_offspring[k] = obj_function(offspring[k])
    non_dominated_sorted_offspring = fast_non_dominated_sort(function1_offspring, function2_offspring)
    crowding_distance_offspring = crowding_distance(function1_offspring[:], function2_offspring[:], non_dominated_sorted_offspring[0][:])
    non_dominated_sorted_offspring[0].sort()
    front = sort_distance(non_dominated_sorted_offspring[0], crowding_distance_offspring)
    front.reverse()
    o_child = offspring[front[0]]

    edgeE_Child = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            edgeE_Child[i][j] = unionMatrix[i][j]
    for i in range(dim):
        edgeE_Child[o_child[i]][i] = 0
    for i in range(dim):
        for j in range(dim):
            if (intersectionMatrix[i][j] == 1):
                edgeE_Child[i][j] = 1
    e_child = [0 for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if (edgeE_Child[i][j] == 1):
                e_child[j] = i
    return [o_child, e_child]



def swap_node(dim, solution):
    offspring = solution[:]
    pos1 = np.random.randint(0, dim - 1)
    pos2 = np.random.randint(pos1 + 1, dim)
    tmp = offspring[pos1]
    offspring[pos1] = offspring[pos2]
    offspring[pos2] = tmp
    return offspring



def mutation_inversion(dim, solution):
    sol = solution[:]
    pos1 = np.random.randint(0, dim - 1)
    pos2 = np.random.randint(pos1 + 1, dim)
    i = pos1
    j = pos2
    while i < j:
        tmp = sol[i]
        sol[i] = sol[j]
        sol[j] = tmp
        i += 1
        j -= 1
    return sol



def mutation_swap_sequence(dim, solution):
    pos1 = np.random.randint(0, dim - 3)
    pos2 = np.random.randint(pos1 + 1, dim - 2)
    pos3 = np.random.randint(pos2 + 1, dim - 1)
    pos4 = np.random.randint(pos3 + 1, dim)
    sol = []
    i = 0
    while i < pos1:
        sol.append(solution[i])
        i += 1
    i = pos3
    while i <= pos4:
        sol.append(solution[i])
        i += 1
    i = pos2 + 1
    while i < pos3:
        sol.append(solution[i])
        i += 1
    i = pos1
    while i <= pos2:
        sol.append(solution[i])
        i += 1
    i = pos4 + 1
    while i < dim:
        sol.append(solution[i])
        i += 1
    return sol



def mutation(dim, solution, pm):
    p = np.random.random_sample()
    if (p < pm/2):
        sol = mutation_inversion(dim, solution)
    elif (p < pm):
        sol = mutation_swap_sequence(dim, solution)
    else:
        sol = solution
    return sol
