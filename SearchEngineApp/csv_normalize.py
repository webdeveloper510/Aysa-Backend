import pandas as pd
import os
import numpy as np


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

# merge_df.to_csv(merge_csv_path , index=False)
# print("merge_df : \n ", merge_df)


# df1 = pd.read_csv(merge_csv_path)
# df2 = pd.read_csv(latest_csv_path)

# # Clean columns
# df1.columns = df1.columns.str.strip()
# df2.columns = df2.columns.str.strip()

# df1 = df1.apply(lambda col: col.str.lower().str.strip().str.replace(r"\s+", " ", regex=True) if col.dtype == "object" else col)
# df2 = df2.apply(lambda col: col.str.lower().str.strip().str.replace(r"\s+", " ", regex=True) if col.dtype == "object" else col)



# # Update product names
# Update_product_name = []
# for idx, row_data in df2.iterrows():
#     category_value = str(row_data["Category"]).strip()
#     product_name = str(row_data["Product Name"]).strip()

#     if category_value in product_name:
#         product_name = product_name.replace(category_value, "").strip()
#     Update_product_name.append(product_name)


# df2["Copy Product Name"] = df2["Product Name"].tolist()

# df2 = df2.drop("Product Name", axis=1)
# df2["Product Name"] = Update_product_name

# # Move Category → Product Type, drop Category
# df2["Product Type"] = df2["Category"].tolist()
# df2 = df2.drop("Category", axis=1)

# print("Shape after cleaning:", df2.shape)

# # Build mapping dict from df1
# mapping = dict(zip(df1['Product Type'], df1['Category']))

# # Map Category from df1, fallback to Product Type if missing
# df2['Category'] = df2['Product Type'].map(mapping).fillna(df2['Product Type'])
# print("================> ", df2.columns.tolist())

# order_column = [
#     "Brand", "Product Name",'Copy Product Name', "Product Type", "Category","Gender",
#     "Production Year", "Profit Margin", "Profit Made",
#     "Release Price", "Wholesale Price", "Link to Product Pictures"
# ]
# df2 = df2[order_column]

# df2 = df2.apply(
#     lambda col: col.str.title().str.strip().str.replace(r"\s+", " ", regex=True)
#     if col.dtype == "object" else col
# )

# df2 = df2.drop("Product Name", axis=1)
# df2 = df2.rename({"Copy Product Name": "Product Name"}, axis=1)
# print("================> ", df2.columns.tolist())
# # Save to CSV
# df2.to_csv("new_useable1.csv", index=False)

