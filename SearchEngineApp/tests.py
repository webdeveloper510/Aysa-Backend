from django.test import TestCase

import pandas as pd

df1 = pd.read_csv('/home/mandeep/Desktop/mandeep_personal/InsightHub/SearchMind/Data/profit_margin_merge.csv')
df2 = pd.read_csv('/home/mandeep/Desktop/mandeep_personal/InsightHub/SearchMind/Data/Tax_Avoidance.csv')
df3 = pd.read_csv('/home/mandeep/Desktop/mandeep_personal/InsightHub/SearchMind/Data/Worker_Pay_Gap.csv')


print(df1.columns.str.strip().tolist())
print()
print(df2.columns.str.strip().tolist())
print()
print(df3.columns.str.strip().tolist())