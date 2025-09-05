import pandas as pd
import os

df1_file_path = os.path.join(os.getcwd() , "Data", 'profit_margins.csv')
df2_file_path = os.path.join(os.getcwd() , "Data", 'profit_margins2.csv')
merge_csv_path = os.path.join(os.getcwd() , "Data", 'profit_margin_merge.csv')



def remove_extra_space(df):
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    return df


# CODE HANDLE FOR PROFIT MARGIN 1 CSV
df1 = pd.read_csv(df1_file_path)
df1.columns = df1.columns.str.strip()
df1 = remove_extra_space(df1)
df1["Wholesale Price"] = "None"


# CODE HANDLE FOR PROFIT MARGIN 2 CSV
df2 = pd.read_csv(df2_file_path)
df2.columns = df2.columns.str.strip()
df2 = remove_extra_space(df2)
df2['Product Type'] = df2["Category"]
df2["Gender"] ="unisex"


df2 = df2.rename(columns={"Link to Product Pic": 'Link to Product Pictures', 'Retial Profit Margin': 'Profit Margin','Market Price (Ib)': "Release Price", 'Wholesale Price (Ib)':'Wholesale Price'})
df2['Release Price'] =df2["Release Price"].apply(lambda x: f"${x}" if "$" not in x else x )
df2['Wholesale Price'] =df2["Wholesale Price"].apply(lambda x: f"${x}" if "$" not in x else x )

order_columns_list = df1.columns.tolist()
df2 = df2[order_columns_list]


merge_df = pd.concat([df1 ,df2],ignore_index=True)