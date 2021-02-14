# This is the main code for running computational experiments for the multi-objective optimization of drone routing
# A summary of the target and source problems is given in the file "MO_routing_tasks.pdf"

import math
import xlsxwriter
import matplotlib.pyplot as plt
from domain_adaptation_routing import MOEA_routing
from MO_routing_tasks import M_151, X_214, Golden_17, Golden_18, CMT4, X_162
import numpy as np




if __name__ == "__main__":

    # Specify routing task (require user input)
    task = M_151()
    plt.title('M-n151', fontsize=15)


    # Set experimental parameters (require user input)
    number_of_runs = 20
    max_gen_target = 100
    transfer_interval = 5
    pop_size_target = 100


    # Save MOEA results to Excel files (require user input)
    wb1a = xlsxwriter.Workbook("M_151_AMTEA_IGD.xlsx")
    sheet1a = wb1a.add_worksheet()
    wb1c = xlsxwriter.Workbook("M_151_AMTEA_trCoefficients.xlsx")
    sheet1c = wb1c.add_worksheet("Source1")
    sheet1c2 = wb1c.add_worksheet("Source2")
    sheet1c3 = wb1c.add_worksheet("Source3")

    wb2a = xlsxwriter.Workbook("M_151_EHBSA_IGD.xlsx")
    sheet2a = wb2a.add_worksheet()

    wb3a = xlsxwriter.Workbook("M_151_MOTrEO_Ms_IGD.xlsx")
    sheet3a = wb3a.add_worksheet()
    wb3c = xlsxwriter.Workbook("M_151_MOTrEO_Ms_trCoefficients.xlsx")
    sheet3c = wb3c.add_worksheet("Source1")
    sheet3c2 = wb3c.add_worksheet("Source2")
    sheet3c3 = wb3c.add_worksheet("Source3")

    wb4a = xlsxwriter.Workbook("M_151_NSGA2_IGD.xlsx")
    sheet4a = wb4a.add_worksheet()


    # Load target Pareto front
    pf_t = open(task.pf_target)
    pf_target = []
    for line in pf_t:
        pf_target.append([float(i) for i in line.split()])


    # Solve the target problem
    best_IGD_each_run_AMTEA = []
    best_IGD_each_run_EHBSA = []
    best_IGD_each_run_MO_TrEO_Ms = []
    best_IGD_each_run_NSGA2 = []
    AMTEA_result = []
    EHBSA_result = []
    MOTrEO_Ms_result = []
    NSGA2_result = []

    for run in range(number_of_runs):

    #############################################################

        print('AMTEA Results for run ', run)
        print("\n")

        best_IGD = math.inf
        result = []

        igd_values, transfer_coefficients = MOEA_routing.AMTEA_routing(
            task=task, task_function=task.target_functions, max_gen=max_gen_target, pop_size=pop_size_target,
            dim=task.dim_target, pc=task.cx_rate, pm=task.mut_rate, tr_int=transfer_interval, pf_target=pf_target)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet1a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_AMTEA.append(best_IGD)
        AMTEA_result.append(result)

        for m in range(0, len(transfer_coefficients)):
            sheet1c.write(m, run, transfer_coefficients[m][0])
            if task.number_of_sources > 1:
                sheet1c2.write(m, run, transfer_coefficients[m][1])
            if task.number_of_sources > 2:
                sheet1c3.write(m, run, transfer_coefficients[m][2])


    ##############################################################

        print('EHBSA Results for run ', run)
        print("\n")

        best_IGD = math.inf
        result = []

        igd_values = MOEA_routing.EHBSA_routing(task_function=task.target_functions,
            max_gen=max_gen_target, pop_size=pop_size_target, dim=task.dim_target, pf_target=pf_target)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet2a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_EHBSA.append(best_IGD)
        EHBSA_result.append(result)


    #############################################################

        print('MOTrEO+Ms Results for run ', run)
        print("\n")
        
        best_IGD = math.inf
        result = []

        igd_values, transfer_coefficients = MOEA_routing.MOTrEO_Ms_routing(task=task, task_function=task.target_functions,
            max_gen=max_gen_target, pop_size=pop_size_target, dim=task.dim_target, pc=task.cx_rate, pm=task.mut_rate,
            tr_int=transfer_interval, pf_target=pf_target)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet3a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_MO_TrEO_Ms.append(best_IGD)
        MOTrEO_Ms_result.append(result)

        for m in range(0, len(transfer_coefficients)):
            sheet3c.write(m, run, transfer_coefficients[m][0])
            if task.number_of_sources > 1:
                sheet3c2.write(m, run, transfer_coefficients[m][1])
            if task.number_of_sources > 2:
                sheet3c3.write(m, run, transfer_coefficients[m][2])


    ############################################################

        print('NSGA-II Results for run ', run)
        print("\n")
    
        best_IGD = math.inf
        result = []
    
        igd_values = MOEA_routing.NSGA2_routing(task_function=task.target_functions,
                                                max_gen=max_gen_target, pop_size=pop_size_target, dim=task.dim_target,
                                                pc=task.cx_rate, pm=task.mut_rate, pf_target=pf_target)
    
        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet4a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_NSGA2.append(best_IGD)
        NSGA2_result.append(result)


    #############################################################

    best_IGD_average_AMTEA = float(sum(best_IGD_each_run_AMTEA) / number_of_runs)
    print("AMTEA average IGD over ", number_of_runs, " runs = ", best_IGD_average_AMTEA)

    best_IGD_average_EHBSA = float(sum(best_IGD_each_run_EHBSA) / number_of_runs)
    print("EHBSA average IGD over ", number_of_runs, " runs = ", best_IGD_average_EHBSA)

    best_IGD_average_MO_TrEO_Ms = float(sum(best_IGD_each_run_MO_TrEO_Ms) / number_of_runs)
    print("MOTrEO+Ms average IGD over ", number_of_runs, " runs = ", best_IGD_average_MO_TrEO_Ms)

    best_IGD_average_NSGA2 = float(sum(best_IGD_each_run_NSGA2) / number_of_runs)
    print("NSGA-II average IGD over ", number_of_runs, " runs = ", best_IGD_average_NSGA2)


    wb1a.close()
    wb1c.close()
    wb2a.close()
    wb3a.close()
    wb3c.close()
    wb4a.close()

    
    ############################################################

    # Plot the convergence curves
    AMTEA_curve = []
    EHBSA_curve = []
    MOTrEO_Ms_curve = []
    NSGA2_curve = []

    for j in range(40):
        AMTEA = []
        NSGA2_Ms = []
        MOTrEO_Ms = []
        NSGA2 = []
        for i in range(number_of_runs):
            AMTEA.append(AMTEA_result[i][j])
            NSGA2_Ms.append(EHBSA_result[i][j])
            MOTrEO_Ms.append(MOTrEO_Ms_result[i][j])
            NSGA2.append(NSGA2_result[i][j])
        AMTEA_curve.append(sum(AMTEA) / number_of_runs)
        EHBSA_curve.append(sum(NSGA2_Ms) / number_of_runs)
        MOTrEO_Ms_curve.append(sum(MOTrEO_Ms) / number_of_runs)
        NSGA2_curve.append(sum(NSGA2) / number_of_runs)


    eval = np.arange(0, 4000, 100)
    plt.plot(eval, AMTEA_curve, 'k')
    plt.plot(eval, EHBSA_curve, 'r')
    plt.plot(eval, MOTrEO_Ms_curve, 'b')
    plt.plot(eval, NSGA2_curve, 'm')
    plt.xlabel("Number of Function Evaluations", fontsize=12)
    plt.ylabel("Average IGD", fontsize=12)
    plt.legend(['AMTEA', 'EHBSA', 'MOTrEO+Ms', 'NSGA-II'])
    plt.show()


    ############################################################
