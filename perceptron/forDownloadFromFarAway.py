import os
import pandas as pd

# не работает на винде :(
#st  = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')

# поэтому пишем ссыль напрямую
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, encoding='utf-8')

print(df.tail())
df.to_csv('iris.csv',header=False, index=False)
