# This is the main code for running computational experiments for the complex multi-objective optimization benchmark tasks
# A concise description of the benchmark tasks is given in the file "MO_benchmark_tasks.pdf"

import pickle
import numpy as np
import math
import xlsxwriter
from domain_adaptation import MOEA
from MO_benchmark_tasks import CIHS, CIMS, CILS, PIHS, PIMS, PILS, NIHS, NIMS, NILS
import matplotlib.pyplot as plt




if __name__ == "__main__":

    # Specify benchmark task (require user input)
    task = NILS()
    plt.title('NILS', fontsize=15)


    # Set experimental parameters (require user input)
    number_of_runs = 20
    max_gen_target = 1000
    transfer_interval = 2
    pop_size_target = 50


    # Save MOEA results to Excel files (require user input)
    wb1a = xlsxwriter.Workbook("NILS_AMTEA_IGD.xlsx")
    sheet1a = wb1a.add_worksheet()
    wb1c = xlsxwriter.Workbook("NILS_AMTEA_trCoefficients.xlsx")
    sheet1c = wb1c.add_worksheet()

    wb2a = xlsxwriter.Workbook("NILS_NSGA2_Ms_IGD.xlsx")
    sheet2a = wb2a.add_worksheet()

    wb3a = xlsxwriter.Workbook("NILS_MOTrEO_Ms_IGD.xlsx")
    sheet3a = wb3a.add_worksheet()
    wb3c = xlsxwriter.Workbook("NILS_MOTrEO_Ms_trCoefficients.xlsx")
    sheet3c = wb3c.add_worksheet()

    wb4a = xlsxwriter.Workbook("NILS_NSGA2_IGD.xlsx")
    sheet4a = wb4a.add_worksheet()

    wb6a = xlsxwriter.Workbook("NILS_MOTrEO_RSM_IGD.xlsx")
    sheet6a = wb6a.add_worksheet()


    # Load target Pareto front
    pf_t = open(task.pf_target)
    pf_target = []
    for line in pf_t:
        pf_target.append([float(i) for i in line.split()])


    # Load available source model and data
    fp = open(task.source_model, 'rb')
    model = pickle.load(fp)
    fp.close()
    fp = open(task.source_data, 'rb')
    source_data = pickle.load(fp)
    fp.close()
    print('Source model dimension =', model.dim)
    print("\n")


    # Solve the target problem
    best_IGD_each_run_AMTEA = []
    best_IGD_each_run_NSGA2_Ms = []
    best_IGD_each_run_MO_TrEO_Ms = []
    best_IGD_each_run_NSGA2 = []
    best_IGD_each_run_MO_TrEO_RSM = []
    transfer_coefficients_MO_TrEO_Ms = []
    AMTEA_result = []
    NSGA2_Ms_result = []
    MOTrEO_Ms_result = []
    NSGA2_result = []
    MOTrEO_RSM_result = []

    for run in range(number_of_runs):

    #############################################################

        print('AMTEA results for run ', run)
        print("\n")

        best_IGD = math.inf
        result = []

        igd_values, transfer_coefficients = MOEA.AMTEA(task.function1_target, task.function2_target,
            max_gen=max_gen_target,pop_size=pop_size_target, dim=task.dim_target, pf_target=pf_target, 
            source_model_list=[model], tr_int=transfer_interval, verbose=True)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet1a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_AMTEA.append(best_IGD)
        AMTEA_result.append(result)

        for i in range(0, len(transfer_coefficients)):
            sheet1c.write(i, run, transfer_coefficients[i]) 


    ##############################################################

        print('NSGA-II+Ms results for run ', run)
        print("\n")

        best_IGD = math.inf
        result = []

        igd_values = MOEA.NSGA2_Ms(task.function1_target, task.function2_target, max_gen=max_gen_target,
            pop_size=pop_size_target, dim=task.dim_target, pf_target=pf_target, source_data=source_data,
            source_model_list=[model], tr_int=transfer_interval, reg=task.reg,)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet2a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_NSGA2_Ms.append(best_IGD)
        NSGA2_Ms_result.append(result)


    #############################################################

        print('MOTrEO+Ms results for run ', run)
        print("\n")
        
        best_IGD = math.inf
        result = []

        igd_values, transfer_coefficients = MOEA.MOTrEO_Ms(task.function1_target, task.function2_target,
            max_gen=max_gen_target,pop_size=pop_size_target, dim=task.dim_target, pf_target=pf_target, 
            source_data=source_data, source_model_list=[model], tr_int=transfer_interval, reg=task.reg, verbose=True)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet3a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_MO_TrEO_Ms.append(best_IGD)
        MOTrEO_Ms_result.append(result)

        for i in range(0, len(transfer_coefficients)):
            sheet3c.write(i, run, transfer_coefficients[i])

        transfer_coefficients_MO_TrEO_Ms.append(transfer_coefficients)


    ############################################################

        print('NSGA-II results for run ', run)
        print("\n")

        best_IGD = math.inf
        result = []

        igd_values = MOEA.NSGA2(task.function1_target, task.function2_target, max_gen=max_gen_target,
            pop_size=pop_size_target, dim=task.dim_target, pf_target=pf_target)

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet4a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_NSGA2.append(best_IGD)
        NSGA2_result.append(result)


    #############################################################

        print('MOTrEO(RSM) results for run ', run)
        print("\n")

        best_IGD = math.inf
        result = []

        igd_values = MOEA.MOTrEO_RSM(task.function1_target, task.function2_target, max_gen=max_gen_target,
                        pop_size=pop_size_target, dim=task.dim_target, pf_target=pf_target, tr_int=transfer_interval,
                        transfer_coefficients=transfer_coefficients_MO_TrEO_Ms[run])

        for j in range(max_gen_target):
            if float(igd_values[j]) < best_IGD:
                best_IGD = float(igd_values[j])
            sheet6a.write(j, run, best_IGD)
            result.append(best_IGD)
        best_IGD_each_run_MO_TrEO_RSM.append(best_IGD)
        MOTrEO_RSM_result.append(result)


    ############################################################

    best_IGD_average_AMTEA = float(sum(best_IGD_each_run_AMTEA) / number_of_runs)
    print("AMTEA average IGD over ", number_of_runs, " runs = ", best_IGD_average_AMTEA)

    best_IGD_average_NSGA2_Ms = float(sum(best_IGD_each_run_NSGA2_Ms) / number_of_runs)
    print("NSGA-II+Ms average IGD over ", number_of_runs, " runs = ", best_IGD_average_NSGA2_Ms)

    best_IGD_average_MO_TrEO_Ms = float(sum(best_IGD_each_run_MO_TrEO_Ms) / number_of_runs)
    print("MOTrEO+Ms average IGD over ", number_of_runs, " runs = ", best_IGD_average_MO_TrEO_Ms)

    best_IGD_average_NSGA2 = float(sum(best_IGD_each_run_NSGA2) / number_of_runs)
    print("NSGA-II average IGD over ", number_of_runs, " runs = ", best_IGD_average_NSGA2)

    best_IGD_average_MO_TrEO_RSM = float(sum(best_IGD_each_run_MO_TrEO_RSM) / number_of_runs)
    print("MOTrEO (RSM) average IGD over ", number_of_runs, " runs = ", best_IGD_average_MO_TrEO_RSM)

    wb1a.close()
    wb1c.close()
    wb2a.close()
    wb3a.close()
    wb3c.close()
    wb4a.close()
    wb6a.close()


    ############################################################

    # Plot the convergence curves
    AMTEA_curve = []
    NSGA2_Ms_curve = []
    MOTrEO_Ms_curve = []
    NSGA2_curve = []
    MOTrEO_RSM_curve = []

    for j in range(200):
        AMTEA = []
        NSGA2_Ms = []
        MOTrEO_Ms = []
        NSGA2 = []
        MOTrEO_RSM =[]
        for i in range(number_of_runs):
            AMTEA.append(AMTEA_result[i][j])
            NSGA2_Ms.append(NSGA2_Ms_result[i][j])
            MOTrEO_Ms.append(MOTrEO_Ms_result[i][j])
            NSGA2.append(NSGA2_result[i][j])
            MOTrEO_RSM.append(MOTrEO_RSM_result[i][j])
        AMTEA_curve.append(sum(AMTEA) / number_of_runs)
        NSGA2_Ms_curve.append(sum(NSGA2_Ms) / number_of_runs)
        MOTrEO_Ms_curve.append(sum(MOTrEO_Ms) / number_of_runs)
        NSGA2_curve.append(sum(NSGA2) / number_of_runs)
        MOTrEO_RSM_curve.append(sum(MOTrEO_RSM) / number_of_runs)


    eval = list(np.arange(0, 10000, 50))
    plt.plot(eval, AMTEA_curve, 'k')
    plt.plot(eval, NSGA2_Ms_curve, 'r')
    plt.plot(eval, MOTrEO_Ms_curve, 'b')
    plt.plot(eval, NSGA2_curve, 'm')
    plt.plot(eval, MOTrEO_RSM_curve, 'g')
    plt.xlabel("Number of Function Evaluations", fontsize=12)
    plt.ylabel("Average IGD", fontsize=12)
    plt.legend(['AMTEA', 'NSGA-II+Ms', 'MOTrEO+Ms', 'NSGA-II', 'MOTrEO(RSM)'])
    plt.show()


    ############################################################
