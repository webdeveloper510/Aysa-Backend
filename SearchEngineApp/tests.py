from django.test import TestCase

import pandas as pd

df1 = pd.read_csv("/home/mandeep/Desktop/mandeep_personal/InsightHub/SearchMind/test.csv")
df2 = pd.read_csv("/home/mandeep/Desktop/mandeep_personal/InsightHub/SearchMind/test1.csv")

print(df1.head())
print(df2.head())

df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.drop_duplicates(inplace=True, ignore_index=True)

print()
print(df_combined)



