import argparse
import os
import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from source.CN2 import CN2
from source.datasets import load_heart, load_iris, load_rice

parser = argparse.ArgumentParser()
group = parser.add_argument_group("Datasets", "By default all datasets are used")
group.add_argument(
    "--short", "-s", action="store_true", help="run CN2 with short dataset"
)
group.add_argument(
    "--medium", "-m", action="store_true", help="run CN2 with medium dataset"
)
group.add_argument(
    "--long", "-l", action="store_true", help="run CN2 with long dataset"
)
parser.add_argument(
    "--iterations",
    "-i",
    type=int,
    default=5,
    help="number of times the algorithm is executed (default 5)",
)
parser.add_argument(
    "--seed", type=int, default=None, help="seed for reproducible results"
)

args = parser.parse_args()

N_ITER = args.iterations
SEED = args.seed


def run_short(file):
    print(
        """
#########################
# RUN WITH IRIS DATASET #
#########################
    """
    )
    best_cn = None
    best_acc = -1
    avg_acc = 0
    avg_time = 0
    for _ in range(N_ITER):
        cn = CN2(5, 0.7, SEED)
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

    avg_acc /= N_ITER
    avg_time /= N_ITER
    best_cn.print_rules()
    best_cn.save_rules("iris_rules")
    best_cn.save_rules("iris_rules", "latex")

    seconds = f"Average training time {avg_time:.3f}s\n"
    best_accuracy = f"Best classification accuracy: {best_acc}\n"
    accuracy = f"Average classification accuracy: {avg_acc}\n"
    file.write("Iris metrics\n")
    file.write(seconds)
    file.write(best_accuracy)
    file.write(accuracy)
    file.write("\n")
    print(seconds)
    print(best_accuracy)
    print(accuracy)


def run_medium(file):
    print(
        """
##########################
# RUN WITH HEART DATASET #
##########################
"""
    )
    best_cn = None
    best_acc = -1
    avg_acc = 0
    avg_time = 0
    for _ in range(N_ITER):
        cn = CN2(5, 0.7, SEED)
        x, y = load_heart()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        start = time.perf_counter()
        cn.fit(x_train, y_train, n_bins=7)
        avg_time += time.perf_counter() - start
        pred_class = cn.predict(x_test)
        acc_score = accuracy_score(y_test, pred_class.astype(y_test.dtype))

        if acc_score > best_acc:
            best_cn = cn
            best_acc = acc_score

        avg_acc += acc_score

    avg_acc /= N_ITER
    avg_time /= N_ITER

    best_cn.print_rules()
    best_cn.save_rules("heart_rules")
    best_cn.save_rules("heart_rules", "latex")

    seconds = f"Average training time {avg_time:.3f}s\n"
    best_accuracy = f"Best classification accuracy: {best_acc}\n"
    accuracy = f"Average classification accuracy: {avg_acc}\n"
    file.write("Heart metrics\n")
    file.write(seconds)
    file.write(best_accuracy)
    file.write(accuracy)
    file.write("\n")
    print(seconds)
    print(best_accuracy)
    print(accuracy)


def run_long(file):
    print(
        """
#########################################
# RUN WITH RICE OSMANCIK CAMMEO DATASET #
#########################################
"""
    )
    best_cn = None
    best_acc = -1
    avg_acc = 0
    avg_time = 0
    for _ in range(N_ITER):
        cn = CN2(5, 0.7, SEED)
        x, y = load_rice()
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=1
        )
        start = time.perf_counter()
        cn.fit(x_train, y_train, n_bins=4)
        avg_time += time.perf_counter() - start
        pred_class = cn.predict(x_test)
        acc_score = accuracy_score(y_test, pred_class.astype(y_test.dtype))

        if acc_score > best_acc:
            best_cn = cn
            best_acc = acc_score

        avg_acc += acc_score

    avg_acc /= N_ITER
    avg_time /= N_ITER

    best_cn.print_rules()
    best_cn.save_rules("rice_rules")
    best_cn.save_rules("rice_rules", "latex")

    seconds = f"Average training time {avg_time:.3f}s\n"
    best_accuracy = f"Best classification accuracy: {best_acc}\n"
    accuracy = f"Average classification accuracy: {avg_acc}\n"
    file.write("Rice metrics\n")
    file.write(seconds)
    file.write(best_accuracy)
    file.write(accuracy)
    file.write("\n")
    print(seconds)
    print(best_accuracy)
    print(accuracy)


if not os.path.exists("results"):
    os.mkdir("results")

if not (args.short or args.medium or args.long):
    with open(os.path.join("results", "iris_metrics.txt"), "w") as f:
        run_short(f)
    with open(os.path.join("results", "heart_metrics.txt"), "w") as f:
        run_medium(f)
    with open(os.path.join("results", "rice_metrics.txt"), "w") as f:
        run_long(f)
else:
    if args.short:
        with open(os.path.join("results", "iris_metrics.txt"), "w") as f:
            run_short(f)
    if args.medium:
        with open(os.path.join("results", "heart_metrics.txt"), "w") as f:
            run_medium(f)
    if args.long:
        with open(os.path.join("results", "rice_metrics.txt"), "w") as f:
            run_long(f)
