import os

import pandas as pd

from CN2 import CN2

if __name__ == '__main__':
    df = pd.read_csv(os.path.join('datasets', 'breast-cancer.csv'))
    cn = CN2(5, 0.7)
    cn.fit(df[:-1], df['class'])

