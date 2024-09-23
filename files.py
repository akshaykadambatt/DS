import os
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as dates

for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname,filename))

def epoch_to_date(epoch_time):
    datetime_time = datetime.datetime.fromtimestamp(epoch_time)
    return datetime_time
print("1------------------")
def check_missing():
    missing_cols, missing_rows = (
        (df.isnull().sum(x) | df.eq('').sum(x)) for x in (0, 1)
    )
    print("3------------------")
    print((df.isnull().sum(axis=0) | df.eq('').sum(axis=0)).loc[lambda x: x.gt(0)].index)
    print("3------------------")

print("2------------------")
df = pd.read_csv('./dataset_final.csv')

print(df.loc[[2,2,3]].sum().loc[lambda x: x.gt(0)])
