import pandas as pd
import numpy as np

df = pd.read_csv("maxvals.csv")

t_lists = df["t_list"].apply(lambda x: [int(i) for i in x[1:-1].split(",")])

grid = np.array(t_lists.to_list())

for i in grid:
    print(i)