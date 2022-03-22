import os

import pandas as pd
from sklearn.metrics import classification_report

from CN2 import CN2

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('datasets', 'iris.csv'))
    cn = CN2(5, 0.2)
    x = df.drop(columns='class')
    y = df['class']
    cn.fit(x, y, n_bins=5, fixed_bin_size=True)
    cn.print_rules()
    pred_class = cn.predict(x)
    print('Classification report:')
    print(classification_report(y, pred_class.astype(y.dtype)))

