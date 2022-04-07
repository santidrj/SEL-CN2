import os

import pandas as pd
from pandas.core.common import random_state
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from CN2 import CN2
from datasets import load_iris, load_heart

if __name__ == '__main__':
    cn = CN2(5, 0.5)
    print("""
#####################
# LOAD IRIS DATASET #
#####################
""")
    x, y = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    cn.fit(x_train, y_train, n_bins=5, fixed_bin_size=True)
    cn.print_rules()
    pred_class = cn.predict(x_test)
    print(f'Classification accuracy: {accuracy_score(y_test, pred_class.astype(y_test.dtype))}')

    cn = CN2(5, 0.5)
    print("""
######################
# LOAD HEART DATASET #
######################
""")
    x, y = load_heart()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    cn.fit(x_train, y_train, n_bins=7)
    cn.print_rules()
    pred_class = cn.predict(x_test)
    print(f'Classification accuracy: {accuracy_score(y_test, pred_class.astype(y_test.dtype))}')
