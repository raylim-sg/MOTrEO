# This code contains the multi-objective benchmark problems and their properties
# A decoding step is included for the target function evaluations
# A concise description of the benchmark tasks is given in the file "MO_benchmark_tasks.pdf"

import math



class CIHS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 100
        self.pf_target = 'pf_concave.txt'
        self.source_model = 's_model_CIHS.txt'
        self.source_data = 's_data_CIHS.txt'
        self.reg = 1000
        self.shift_vector_target = []
        self.rotation_matrix_target = []


    def function1_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        f1 = x[0]
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        q = 0
        for i in range(1, len(x)):
            q += abs(x[i])
        q = 1 + ((9 / (len(x) - 1)) * q)
        h = 1 - pow(x[0] / q, 2.0)
        f2 = q * h
        return f2




class CIMS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 5
        self.pf_target = 'pf_circle.txt'
        self.source_model = 's_model_CIMS.txt'
        self.source_data = 's_data_CIMS.txt'
        self.reg = 100
        shift_vector_t = [0 for _ in range(20 - 1)]
        rotation_matrix_t = [[0 for _ in range(20 - 1)] for _ in range(20 - 1)]
        file1 = open("M_CIMS_2.txt")
        file2 = open("S_CIMS_2.txt")
        value2 = file2.readline().split(" ")
        for i in range(0, 20 - 1):
            value1 = file1.readline().split(" ")
            for j in range(0, 20 - 1):
                rotation_matrix_t[i][j] = float(value1[j])
            shift_vector_t[i] = float(value2[i])
        self.shift_vector_target = shift_vector_t
        self.rotation_matrix_target = rotation_matrix_t


    def function1_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        z = [0 for _ in range(1, len(x))]
        for k in range(1, len(x)):
            z[k - 1] = x[k] - self.shift_vector_target[k - 1]

        j = 0
        while j < len(x) - 1:
            matrix = 0
            solution = 0
            value = 0
            for variable in range(1, len(x)):
                value += self.rotation_matrix_target[j][matrix] * z[solution]
                matrix += 1
                solution += 1
            z[j] = value
            j += 1

        q = 0
        for i in range(1, len(x)):
            q += abs(z[i - 1])
        q = (q * 9 / (len(x) - 1)) + 1
        f1 = q * math.cos(math.pi * x[0] / 2.0)
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        z = [0 for _ in range(1, len(x))]
        for k in range(1, len(x)):
            z[k - 1] = x[k] - self.shift_vector_target[k - 1]

        j = 0
        while j < len(x) - 1:
            matrix = 0
            solution = 0
            value = 0
            for variable in range(1, len(x)):
                value += self.rotation_matrix_target[j][matrix] * z[solution]
                matrix += 1
                solution += 1
            z[j] = value
            j += 1

        q = 0
        for i in range(1, len(x)):
            q += abs(z[i - 1])
        q = (q * 9 / (len(x) - 1)) + 1
        f2 = q * math.sin(math.pi * x[0] / 2.0)
        return f2




class CILS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 50
        self.pf_target = 'pf_circle.txt'
        self.source_model = 's_model_CILS.txt'
        self.source_data = 's_data_CILS.txt'
        self.reg = 100
        self.shift_vector_target = []
        self.rotation_matrix_target = []


    def function1_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        q = 1
        for i in range(1, len(x)):
            q += pow(x[i], 2.0) - (10 * math.cos(2 * math.pi * x[i])) + 10
        f1 = q * math.cos(math.pi * x[0] / 2.0)
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        q = 1
        for i in range(1, len(x)):
            q += pow(x[i], 2.0) - (10 * math.cos(2 * math.pi * x[i])) + 10
        f2 = q * math.sin(math.pi * x[0] / 2.0)
        return f2




class PIHS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 100
        self.pf_target = 'pf_convex.txt'
        self.source_model = 's_model_PIHS.txt'
        self.source_data = 's_data_PIHS.txt'
        self.reg = 1000
        shift_vector_t = [0 for _ in range(20 - 1)]
        file = open("S_PIHS_2.txt")
        value = file.readline().split(" ")
        for i in range(0, 20 - 1):
            shift_vector_t[i] = float(value[i])
        self.shift_vector_target = shift_vector_t
        self.rotation_matrix_target = []


    def function1_target(self, solution):
        x = solution[:]

        f1 = x[0]
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        z = [0 for _ in range(1, len(x))]
        for k in range(1, len(x)):
            z[k - 1] = x[k] - self.shift_vector_target[k - 1]

        q = 0
        for i in range(1, len(x)):
            q += pow(z[i - 1], 2.0) - (10 * math.cos(2 * math.pi * z[i - 1])) + 10
        q = 1 + q
        f2 = q * (1 - math.sqrt(x[0] / q))
        return f2




class PIMS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 1
        self.pf_target = 'pf_concave.txt'
        self.source_model = 's_model_PIMS.txt'
        self.source_data = 's_data_PIMS.txt'
        self.reg = 500
        self.shift_vector_target = []
        rotation_matrix_t = [[0 for _ in range(20 - 1)] for _ in range(20 - 1)]
        file = open("M_PIMS_2.txt")
        for i in range(0, 20 - 1):
            value = file.readline().split(" ")
            for j in range(0, 20 - 1):
                rotation_matrix_t[i][j] = float(value[j])
        self.rotation_matrix_target = rotation_matrix_t


    def function1_target(self, solution):
        x = solution[:]

        f1 = x[0]
        return f1

    def function2_target(self, solution):
        x = solution[:]

        z = [0 for _ in range(1, len(x))]

        j = 0
        while j < len(x) - 1:
            matrix = 0
            value = 0
            for variable in range(1, len(x)):
                value += self.rotation_matrix_target[j][matrix] * x[variable]
                matrix += 1
            z[j] = value
            j += 1

        q = 0
        for i in range(1, len(x)):
            q += pow(z[i - 1], 2.0) - (10 * math.cos(2 * math.pi * z[i - 1])) + 10
        q = 1 + q
        f2 = q * (1 - pow(x[0] / q, 2.0))
        return f2




class PILS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 100
        self.pf_target = 'pf_circle.txt'
        self.source_model = 's_model_PILS.txt'
        self.source_data = 's_data_PILS.txt'
        self.reg = 100
        shift_vector_t = [0 for _ in range(20 - 1)]
        file = open("S_PILS_2.txt")
        value = file.readline().split(" ")
        for i in range(0, 20 - 1):
            shift_vector_t[i] = float(value[i])
        self.shift_vector_target = shift_vector_t
        self.rotation_matrix_target = []


    def function1_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        z = [0 for _ in range(1, len(x))]
        for k in range(1, len(x)):
            z[k - 1] = x[k] - self.shift_vector_target[k - 1]

        q = 0
        h = 0
        for i in range(1, len(x)):
            q += pow(z[i - 1], 2.0)
            h += math.cos(2 * math.pi * z[i - 1])
        q = math.sqrt((1 / (len(x) - 1)) * q)
        h = (1/ (len(x) - 1)) * h
        q = 21 + math.e - (20 * math.exp(-0.2 * q)) - math.exp(h)

        f1 = q * math.cos(math.pi * x[0] / 2.0)
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        z = [0 for _ in range(1, len(x))]
        for k in range(1, len(x)):
            z[k - 1] = x[k] - self.shift_vector_target[k - 1]

        q = 0
        h = 0
        for i in range(1, len(x)):
            q += pow(z[i - 1], 2.0)
            h += math.cos(2 * math.pi * z[i - 1])
        q = math.sqrt((1 / (len(x) - 1)) * q)
        h = (1/ (len(x) - 1)) * h
        q = 21 + math.e - (20 * math.exp(-0.2 * q)) - math.exp(h)

        f2 = q * math.sin(math.pi * x[0] / 2.0)
        return f2




class NIHS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 80
        self.pf_target = 'pf_circle.txt'
        self.source_model = 's_model_NIHS.txt'
        self.source_data = 's_data_NIHS.txt'
        self.reg = 1000
        self.shift_vector_target = []
        self.rotation_matrix_target = []


    def function1_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        q = 1
        for i in range(1, len(x) - 1):
            q += (100 * pow(pow(x[i], 2.0) - x[i + 1], 2.0)) + pow(1 - x[i], 2.0)
        q += (100 * pow(pow(x[len(x) - 1], 2.0) - 1, 2.0)) + pow(1 - x[len(x) - 1], 2.0)
        f1 = q * math.cos(math.pi * x[0] / 2.0)
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(1, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        q = 1
        for i in range(1, len(x) - 1):
            q += (100 * pow(pow(x[i], 2.0) - x[i + 1], 2.0)) + pow(1 - x[i], 2.0)
        q += (100 * pow(pow(x[len(x) - 1], 2.0) - 1, 2.0)) + pow(1 - x[len(x) - 1], 2.0)
        f2 = q * math.sin(math.pi * x[0] / 2.0)
        return f2




class NIMS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 20
        self.pf_target = 'pf_concave.txt'
        self.source_model = 's_model_NIMS.txt'
        self.source_data = 's_data_NIMS.txt'
        self.reg = 100
        self.shift_vector_target = []
        rotation_matrix_t = [[0 for _ in range(20 - 2)] for _ in range(20 - 2)]
        file = open("M_NIMS_2.txt")
        for i in range(0, 20 - 2):
            value = file.readline().split(" ")
            for j in range(0, 20 - 2):
                rotation_matrix_t[i][j] = float(value[j])
        self.rotation_matrix_target = rotation_matrix_t


    def function1_target(self, solution):
        x = solution[:]

        f1 = 0.5 * (x[0] + x[1])
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(2, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        z = [0 for _ in range(2, len(x))]

        j = 0
        while j < len(x) - 2:
            matrix = 0
            value = 0
            for variable in range(2, len(x)):
                value += self.rotation_matrix_target[j][matrix] * x[variable]
                matrix += 1
            z[j] = value
            j += 1

        q = 1
        for i in range(0, len(x) - 2):
            q += pow(z[i], 2.0)

        f2 = q * (1 - (pow((x[0] + x[1]) / (2 * q), 2.0)))
        return f2




class NILS(object):
    def __init__(self):
        self.dim_target = 20
        self.upper_bound_target = 100
        self.pf_target = 'pf_concave.txt'
        self.source_model = 's_model_NILS.txt'
        self.source_data = 's_data_NILS.txt'
        self.reg = 100
        self.shift_vector_target = []
        self.rotation_matrix_target = []


    def function1_target(self, solution):
        x = solution[:]
        f1 = 0.5 * (x[0] + x[1])
        return f1

    def function2_target(self, solution):
        x = solution[:]
        for k in range(2, len(x)):
            x[k] = (2 * self.upper_bound_target * solution[k]) - self.upper_bound_target

        q1 = 0
        h = 0
        for i in range(2, len(x)):
            q1 += pow(x[i], 2.0)
            h += math.cos(2 * math.pi * x[i])
        q1 = math.sqrt((1 / (len(x) - 2)) * q1)
        h = (1/ (len(x) - 2)) * h
        q = 21 + math.e - (20 * math.exp(-0.2 * q1)) - math.exp(h)

        f2 = q * (1 - (pow((x[0] + x[1]) / (2 * q), 2.0)))
        return f2
