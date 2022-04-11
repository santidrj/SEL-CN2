import os
import time

import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from CN2 import CN2
from datasets import load_heart, load_iris, load_rice

N_ITERS = 10

if __name__ == "__main__":
    with open(os.path.join("results", "metrics.txt"), "w") as f:
        print(
            """
#####################
# LOAD IRIS DATASET #
#####################
    """
        )
        best_cn = None
        best_acc = -1
        avg_acc = 0
        avg_time = 0
        for _ in range(N_ITERS):
            cn = CN2(5, 0.5)
            x, y = load_iris()
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=1
            )
            start = time.perf_counter()
            cn.fit(x_train, y_train, n_bins=8, fixed_bin_size=True)
            avg_time += time.perf_counter() - start
            pred_class = cn.predict(x_test)
            acc_score = accuracy_score(y_test, pred_class.astype(y_test.dtype))

            if acc_score > best_acc:
                best_cn = cn
                best_acc = acc_score

            avg_acc += acc_score

        avg_acc /= N_ITERS
        avg_time /= N_ITERS
        best_cn.print_rules()
        best_cn.save_rules("iris_rules")
        best_cn.save_rules("iris_rules", "latex")

        seconds = f'Average training time {avg_time:.3f}s\n'
        best_accuracy = f"Best classification accuracy: {best_acc}\n"
        accuracy = f"Average classification accuracy: {acc_score}\n"
        f.write("Iris metrics\n")
        f.write(seconds)
        f.write(best_accuracy)
        f.write(accuracy)
        f.write("\n")
        print(seconds)
        print(best_accuracy)
        print(accuracy)

        print(
            """
######################
# LOAD HEART DATASET #
######################
    """
        )
        cn = CN2(5, 0.5)
        x, y = load_heart()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        start = time.perf_counter()
        cn.fit(x_train, y_train, n_bins=7)
        end = time.perf_counter() - start
        cn.print_rules()
        cn.save_rules("heart_rules")
        cn.save_rules("heart_rules", "latex")
        pred_class = cn.predict(x_test)

        seconds = f'Training took: {end:.3f} seconds\n'
        accuracy = f"Classification accuracy: {accuracy_score(y_test, pred_class.astype(y_test.dtype))}\n"
        f.write("Heart metrics\n")
        f.write(seconds)
        f.write(accuracy)
        f.write("\n")
        print(seconds)
        print(accuracy)

        print(
            """
#####################################
# LOAD RICE OSMANCIK CAMMEO DATASET #
#####################################
    """
        )
        cn = CN2(5, 0.5)
        x, y = load_rice()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        start = time.perf_counter()
        cn.fit(x_train, y_train, n_bins=4)
        end = time.perf_counter() - start
        cn.print_rules()
        cn.save_rules("rice_rules")
        cn.save_rules("rice_rules", "latex")
        pred_class = cn.predict(x_test)

        seconds = f'Training took: {end:.3f} seconds\n'
        accuracy = f"Classification accuracy: {accuracy_score(y_test, pred_class.astype(y_test.dtype))}\n"
        f.write("Rice metrics\n")
        f.write(seconds)
        f.write(accuracy)
        f.write("\n")
        print(seconds)
        print(accuracy)
