from __future__ import division, print_function, unicode_literals
__author__ = "R a f i j ..."
import time  # output some timings per evaluation
from collections import defaultdict
import os
import webbrowser  # to show post-processed results in the browser
import numpy as np  # for np.median
import cocoex  # experimentation module
import cocopp  # post-processing module
from solver_LiA_COCO import lia


def main_lia():
    # -------------------------------------------------------
    # set on COCO
    # -------------------------------------------------------
    suite_name = "bbob"
    solver = lia
    algorithm_name = "LiA_algorithm"  # no spaces allowed
    output_folder = algorithm_name  # no spaces allowed
    suite = cocoex.Suite(suite_name, "", "")
    observer = cocoex.Observer(suite_name, "result_folder: {0} algorithm_name: {1}".format(
                                   output_folder, algorithm_name))

    # ---------------------------------------------------------------------------
    # initial variables
    # ---------------------------------------------------------------------------
    minimal_print = cocoex.utilities.MiniPrint()
    stoppings = defaultdict(list)  # dict of lists, key is the problem index
    timings = defaultdict(list)    # key is the dimension

    print('------------------------<< Start LiA analysis >>-------------------------')
    # ------------------------------------------------------------------------------
    # start benchmark functions COCO
    # ------------------------------------------------------------------------------
    instances, instances_tot = 1, 15
    func_dim = 24 * instances_tot
    # --------------------------------------------------------------
    # ide => 0 -> 2D, 1 -> 3D, 2 -> 5D, 3 -> 10D, 4 -> 20D, 5 -> 40D
    # -------------------------------------------------------------
    ide, func = np.array([1, 2, 3, 4]).astype('int'), 1
    ini_dim = ide * func_dim
    ini_fun = ini_dim + (instances_tot * (func-1))
    fin_dim = ini_dim + func_dim
    # -------------------------------------------------
    # number of instances
    # -------------------------------------------------
    n_instance = 15
    ini_instance = ini_dim
    instances = n_instance + ini_instance
    fin_instance = ini_instance + 15
    # ------------------------------------------------
    steps_min, steps_max = [1e-1, 1e-1, 1e-1, 1e-1], [1e-8, 1e-8, 1e-9, 1e-9]
    n_groups = [7, 7, 8, 8]
    itr_max = [5e2, 2e3, 4e3, 5e3]
    act = 1

    time0 = time.time()
    for index, problem in enumerate(suite):
        if act > len(fin_instance)-1:
            break
        if index == fin_instance[act]:
            ini_instance[act] = index
            instances[act] = n_instance + ini_instance[act]
            fin_instance[act] = ini_instance[act] + 15

        if ini_dim[act] <= index < fin_dim[act] and ini_fun[act] <= index < instances[act]:
            print(index)
            print(problem)
            # --------------------------------------------
            # generate the data for cocopp post-processing
            # ---------------------------------------------
            problem.observe_with(observer)
            problem(np.zeros(problem.dimension))
            if not len(timings[problem.dimension]) and len(timings) > 1:
                print("\n   %s %d-D done in %.1e seconds/evaluations" % (minimal_print.stime, sorted(timings)[-2],
                                                                         np.median(timings[sorted(timings)[-2]])),
                      end='')
            # ---------------------------------------------------------------
            # star LiA algorithm
            # ---------------------------------------------------------------
            time1 = time.time()
            max_runs = int(itr_max[act])

            s_min, s_max = steps_min[act], steps_max[act]

            output = solver(problem, stp_min=s_min, stp_max=s_max, itr=max_runs, ide_dim=act, n_gr=n_groups[act])
            stoppings[problem.index].append(output[1:])
            timings[problem.dimension].append((time.time() - time1) / problem.evaluations if problem.evaluations else 0)

            with open(output_folder + '_stopping_conditions.pydict', 'wt') as file_:
                file_.write("# code to read in these data:\n"
                            "# import ast\n"
                            "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
                            "# stoppings = ast.literal_eval(file_.read())\n"
                            % output_folder)
                file_.write(repr(dict(stoppings)))

            # ----------------------------------------------------------
            # timings
            # ----------------------------------------------------------
            timings[problem.dimension].append((time.time() - time1) / problem.evaluations if problem.evaluations else 0)
            minimal_print(problem, final=problem.index == len(suite) - 1)

        if index > fin_dim[act]:
            act += 1

    # ----------------------------------------------------------
    # print timings and final message
    # ----------------------------------------------------------
    print("\n   %s %d-D done in %.1e seconds/evaluations"
          % (minimal_print.stime, sorted(timings)[-1], np.median(timings[sorted(timings)[-1]])))
    print("*** Full experiment done in %s ***" % cocoex.utilities.ascetime(time.time() - time0))

    print("Timing summary:\n" "  dimension  median seconds/evaluations\n" "  -------------------------------------")
    for dimension in sorted(timings):
        print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
    print("  -------------------------------------")

    # -----------------------------------------------------------------------
    # post-process data
    # -----------------------------------------------------------------------
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001"
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")


if __name__ == "__main__":
    main_lia()


