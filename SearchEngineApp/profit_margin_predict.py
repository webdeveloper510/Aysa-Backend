import pandas as pd
import os
import sys
import torch
from sentence_transformers import SentenceTransformer , util
from thefuzz import fuzz
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Detect device CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Take a  parameter to filter out dataframe
top_n = 80
similarity = 0.75


class ProfitMarginPreidction:

    def __init__(self, pickle_df , model, user_query):
        self.df = pickle_df
        self.model = model
        self.user_query = user_query
        self.original_df = pickle_df.copy()

    # function to convert profit margin api
    def convert_profit_margin(self,df: pd.DataFrame) -> pd.DataFrame:
        # Remove all characters except digits and dot
        df["Profit Margin"] = (
            df["Profit Margin"]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)  # keep digits, minus, dot
            .replace("", "0")  # if empty string, set to 0
            .astype(float)
        )
        return df
    

    # function to  change profit margin value...
    # Remove all characters except digits and dot
    def convert_profit_margin(self , df : pd.DataFrame) -> pd.DataFrame:
        df["Profit Margin"] = (
            df["Profit Margin"]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)  # keep digits, minus, dot
            .replace("", "0")  # if empty string, set to 0
            .astype(float)
        )
        return df
    
    
    # This function is return dataframe if user asked about only Brand Name
    def BrandDF(self, user_query: str , pickle_df: pd.DataFrame) -> pd.DataFrame:
        pickle_df.loc[:, "Brand"] = pickle_df["Brand"].astype(str).str.lower().str.strip()
        user_query = str(user_query.lower().strip())

        filtered_df = pd.DataFrame()
        if user_query in pickle_df["Brand"].values:  
            filtered_df = pickle_df.loc[(pickle_df["Brand"]==user_query)]
            filtered_df = filtered_df.sort_values('Production Year', ascending=False)  
            filtered_df = filtered_df.drop_duplicates(subset=["Production Year"], keep="first")       # Keep only first row of every brand

            if len(filtered_df) > 3:
                filtered_df = filtered_df.iloc[0:3]

            filtered_df = filtered_df.drop(columns=["Category" ,"Gender", "text", "text_embedding"],  errors="ignore")      # remove unneccessary dataframe
        return filtered_df
    # function to apply embedding 
    def apply_embedding(self):

        query_embedding = self.model.encode(self.user_query, convert_to_tensor=True).to(device)      # convert user query into vector   
        # Convert all full Text  embeddings to tensor
        # tensor([-0.0947, -0.0559, 0.0754, 0.0389, ...], device='cuda:0')
        full_embeddings = [torch.tensor(e).to(device) for e in self.df['text_embedding']]  
        full_text_embedding_tensor = torch.stack(full_embeddings)                #(num_rows, embedding_dimension)   like     [10000, 384]
        similarity_score = util.cos_sim(query_embedding, full_text_embedding_tensor)[0].cpu().numpy()
        self.df['similarity_score'] = similarity_score
        
        # remove unnessary columns from the Embeddingf df
        embedding_df = (
            self.df.drop(columns=["text_embedding"])
            .sort_values('similarity_score', ascending=False)
            .head(top_n)
            )
        
        return embedding_df

    # function to get matched row and get required paramter
    def GetMatchedRow_AndParameter(self , embedding_df : pd.DataFrame)-> dict:
        matched_row = embedding_df.loc[embedding_df["similarity_score"].idxmax()]           # Get Most highest similarity rows
        matched_row_data = matched_row.to_dict() 
                                                   # Convert Series row in dict
        # Get paramter from the dataframe
        matched_brand =str(matched_row_data.get("Brand")).lower().strip()
        matched_category = str(matched_row.get("Category")).lower().strip()
        matched_year = int(matched_row.get("Production Year"))
        matched_product_type = str(matched_row.get("Product Type")).lower().strip()
        matched_gender = str(matched_row.get("Gender")).lower().strip()

        # Make a response dict
        response_dict = {
                            "matched_brand" :  matched_brand,
                            "matched_category" :  matched_category,
                            "matched_year" :  matched_year,
                            "matched_product_type" :  matched_product_type,
                            "matched_gender" :  matched_gender,
                        }
        return response_dict , matched_row_data

    # function to return filter out dataframe based on the category value
    def Get_Category_based_df(self,response_dict : dict) -> pd.DataFrame:

        # Make lower case and remove extra space
        self.original_df.loc[:,"Category"] =  self.original_df["Category"].astype(str).str.lower().str.strip()
        
        # Get Category value from Matched Row dict
        matched_category = response_dict.get("matched_category")

        # filter df
        categorized_df = self.original_df.loc[self.original_df["Category"] == matched_category]
        return categorized_df

    # function to return filter out dataframe based on the matched year value
    def Get_year_based_df(self,response_dict : dict , categorized_df: pd.DataFrame) -> pd.DataFrame:
        
        # Get Year value from Matched Row dict
        matched_year = response_dict.get("matched_year")

        # filter df
        yearly_df = categorized_df.loc[categorized_df["Production Year"].astype(int) == matched_year]
        return yearly_df
    
    # function to return filter out dataframe based on the matched GENDER value
    def Get_gender_based_df(self,response_dict : dict , yearly_df: pd.DataFrame) -> pd.DataFrame:

        # Make lower case and remove extra space
        yearly_df.loc[:, "Gender"] = yearly_df["Gender"].astype(str).str.lower().str.strip()

        # Get Gender value from Matched Row dict
        matched_gender = response_dict.get("matched_gender")
        
        # filter df
        classify_gender_df = yearly_df.loc[yearly_df["Gender"] == matched_gender]
        return classify_gender_df
    
    # This function Returning Two lists
    # One is compare list which is filter rows based on the Brand value and Product Type Value
    # Second is brand list which is filter rows based on the only Brand Value
    def Filter_rows_list(self, response_dict : dict, genderly_df: pd.DataFrame) -> pd.DataFrame:

        brand_product_type =[]

        # Get values from paramter/ response dict
        matched_brand_name = response_dict.get("matched_brand") 
        matched_product_type= response_dict.get("matched_product_type") 

        # Iterate through the dataframe
        for idx , row_data in genderly_df.iterrows():
            filtered_brand_name = str(row_data.get("Brand")).lower().strip()
            filtered_product_type = str(row_data.get("Product Type")).lower().strip()
            
            # Skip same brand
            if ((matched_brand_name != filtered_brand_name and matched_product_type == filtered_product_type)
                    or
                (matched_brand_name != filtered_brand_name and matched_product_type in filtered_product_type)
                    or 
                (matched_brand_name != filtered_brand_name and filtered_product_type in matched_product_type)
                ):
                brand_product_type.append(row_data)
        
        return brand_product_type
    
    # This function is just filtering dataframe 
    # Based on the brand product type list
    # And only brand name list
    def Filtered_Dataframe(self,brand_product_type_list: list) -> pd.DataFrame:
 
        filtered_df = pd.DataFrame(brand_product_type_list)
        
        # Return Empty list when filtered df is empty 
        if filtered_df.empty:
            return []
        
        # Remove extra sign , percentage sign and extra spaces from profit margin csv
        filtered_df = self.convert_profit_margin(filtered_df)
        
        # Get min amd max value of specific brand
        agg_df = filtered_df.groupby("Brand")["Profit Margin"].agg(["min", "max"]).reset_index()

        # if there is multiple brand exist in dataframe
        if len(agg_df) >1 :

            # Remove extra spaces from the brand column value
            agg_df["Brand"] = agg_df["Brand"].str.strip()

            # Get max and min profit value of specific brand
            max_profit =    agg_df['max'].max()
            min_profit =    agg_df['min'].min()
            
            # Get the Brand(s) corresponding to max and min
            max_brands = agg_df[agg_df['max'] == max_profit]['Brand'].tolist()
            min_brands = agg_df[agg_df['min'] == min_profit]['Brand'].tolist()

            #Get rows from filtered_df where Profit Margin equals max/min and Brand matches
            highest_row = filtered_df[(filtered_df['Brand'].str.strip().isin(max_brands)) & (filtered_df['Profit Margin'].astype(float) == max_profit)]
            lowest_rows = filtered_df[(filtered_df['Brand'].str.strip().isin(min_brands)) & (filtered_df['Profit Margin'].astype(float) == min_profit)]

            filtered_df = pd.concat([highest_row, lowest_rows], ignore_index=True)                  # Concate dataframe with highest and lowest margin rows
            filtered_df = filtered_df.sort_values('Profit Margin', ascending=False)                 # Sort dataframe based on the profit margin descending value
            filtered_df = filtered_df.drop_duplicates(subset=["Profit Margin"], keep="first")       # Keep only first row of every brand


        elif len(agg_df) ==1 : 
            print('agg df elif condition is running ')
            # Only one brand exists, get max profit margin row
            max_profit = agg_df['max'].iloc[0]
            max_brand = agg_df['Brand'].iloc[0]

            filtered_df = filtered_df[(filtered_df['Brand'].str.strip() == max_brand) & 
                                        (filtered_df['Profit Margin'].astype(float) == max_profit)]

        
        return filtered_df

