from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from datetime import datetime , timedelta

# Import python packages
import os 
import jwt
import sys
from pathlib import Path
import pandas as pd
import torch
import csv
from datetime import date

# import Project files
from .response import *
from .models import *
from .utils import *
from .product_structure import *
from .tax_structure import *
from .profit_margin_predict import *
from .ceo_worker import *
from sentence_transformers import SentenceTransformer , util
from django.contrib.auth.hashers import make_password   , check_password
from SearchMind.settings import *
import requests


from dotenv import load_dotenv
load_dotenv()

# use in both  tax and ceo worker semantic search api
TAX_CEO_WORKER_SIMILARITY = round(float(os.getenv("TAX_CEO_WORKER_SIMILARITY")), 2)     
TAX_CEO_WORKER_YEAR_SIMILARITY = round(float(os.getenv("TAX_CEO_WORKER_YEAR_SIMILARITY")), 2)

# GET TOP_N MATCHED VALUE
TOP_N = int(os.getenv("TOP_N"))

""" ###############################          Profit Margin Data    ##################################"""
# API to inference product trained model
class ProductSemanticSearchView(APIView):

    def FilterUserQuery(self , input_text: str) -> list:
        split_text = input_text.split(" ")
        unique_list = list(set(split_text))
        return unique_list
    
    # FUNCTION TO GET PRODUCT OF SINGLE USER QUERY
    def get_brand_products(self ,pickle_df, user_query: str) -> pd.DataFrame:
        try:
            # Make a copy of df
            df = pickle_df.copy()

            # Drop Duplicate Rows from the dataframe
            df = df.drop_duplicates(subset=["Brand", "Product Name", "Product Type", "Category", "Production Year"])

            # Normalize Brand Column
            df['Brand'] = df['Brand'].astype(str).str.lower().str.strip()

            # Handle synonyms
            if user_query.lower().strip() in ['apple', 'iphone']:
                user_query = "apple"
            else:
                user_query = user_query.lower().strip()

            # Check if Brand exists
            mask = df['Brand'].str.contains(user_query, case=False, na=False)
            if not mask.any():
                return ProductResponse("error", f"No products found for brand '{user_query}'")

            # Drop unnecessary columns
            drop_cols = [col for col in ["text_embedding", "brand_embedding", "text", "brand"] if col in df.columns]
            df = df.drop(columns=drop_cols, errors="ignore")

            # Get Masked DF & Sort
            masked_df = df[mask]
            sorted_df = masked_df.sort_values("Production Year", ascending=False)

            # Get Latest Matched Row
            matched_row = sorted_df.iloc[0].to_dict()

            # Extract Values
            brand_name = str(matched_row.get("Brand")).lower().strip()
            product_name = str(matched_row.get("Product Name")).lower().strip()

            # Filter same brand but different products
            filtered_df = sorted_df.loc[
                (sorted_df["Brand"].str.lower().str.strip() == brand_name) &
                (sorted_df["Product Name"].str.lower().str.strip() != product_name)
            ].sort_values("Production Year", ascending=False)

            # Select one unique product per year
            selected_rows = []
            used_products = set()

            for year, group in filtered_df.groupby("Production Year", sort=False):
                row = group.loc[~group["Product Name"].str.lower().isin(used_products)].head(1)
                if not row.empty:
                    selected_rows.append(row)
                    used_products.add(row["Product Name"].iloc[0].lower())

            if not selected_rows:
                return ProductResponse("error", f"No alternative products found for '{user_query}'")

            filtered_unique = (
                pd.concat(selected_rows)
                .sort_values("Production Year", ascending=False)
                .reset_index(drop=True)
            )

            # Format output
            filtered_unique["Brand"] = filtered_unique["Brand"].str.title()
            if len(filtered_unique) > 3:
                filtered_unique = filtered_unique.iloc[0:3]

            return filtered_unique

        except Exception as e:
            exc_type , exc_obj , exc_tb = sys.exc_info()
            error_message = f"[ERROR] failed to get Products for matched single category , error is : {str(e)} in line no : {exc_tb.tb_lineno}"
            return error_message

    
   # Main function 
    def post(self, request, format=None):
        try:
            PROFIT_MARGIN_SIMILARITY_SCORE = os.getenv("PROFIT_MARGIN_SIMILARITY_SCORE")            # use this in produict semantic search api
            # Get threshold value from environemnt file
            if isinstance(PROFIT_MARGIN_SIMILARITY_SCORE, str):
                PROFIT_MARGIN_SIMILARITY_SCORE = round(float(PROFIT_MARGIN_SIMILARITY_SCORE),2)

            # Required Fields\
            required_fields= ['query','tab_type', 'device_type' , 'target_year']

            # Get Payload data
            payload = request.data
       

            # Handle missing field
            missing_fields = [field for field in required_fields if payload.get(field) is None  or not payload.get(field)]
            if missing_fields:
                return Response({
                    'message':f"{', '.join(missing_fields)}: key is required .",
                    'status':status.HTTP_400_BAD_REQUEST
                }, status=status.HTTP_400_BAD_REQUEST)
        
            # Handle device type value
            device_type =str(payload.get("device_type")).lower().strip()

            if device_type not in ["mobile", "desktop"]:
                return Response({
                    "message": "Invalid device type , Please choose one from them ['mobile' , 'desktop']" ,
                    "status": 400,
                }, status=status.HTTP_400_BAD_REQUEST)
            
          
            target_year =int(payload.get("target_year"))
            # print("target_year===================>",target_year)


            # Create a object of gloabl search APIVIEW
            global_search_obj = GlobalSearchAPIView()

            # get payload value in parameter
            user_query = str(payload.get("query")).lower().strip()

            # Define paths
            pickle_df_path = os.path.join(os.getcwd() ,"static", "media", "EmbeddingDir", "Profit Margin" ,"profit_embedding.pkl")
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')

            # Load model
            model = SentenceTransformer(transfer_model_path)
            pickle_df = pd.read_pickle(pickle_df_path)

            # CALL A CLASS TO PREDICT PROFIT MARGIN DATA 
            Profit_Obj  = ProfitMarginPreidction(pickle_df,model,user_query)

            # Handle when user has asked about only brand name
            user_query_filter_list = self.FilterUserQuery(user_query)

            if len(user_query_filter_list)  ==1:
                
                # call function to get dataframe
                result_df = self.get_brand_products(pickle_df , user_query)

                # HANDLE IF FUNCTION RETURN ERROR
                if isinstance(result_df ,str):
                    return Internal_server_response(result_df)
                
                # HANDLE IF FUNCTION RETURN ERROR
                elif isinstance(result_df ,pd.DataFrame):
                    json_output= result_df.to_dict(orient="records")
                    #print("json_output=================>",json_output)

                    CEO_WORKER_JSON_DATA=[]
                    if json_output:
                        brand_name = str(json_output[0]["Brand"]).lower().strip()
                        production_year = int(json_output[0]["Production Year"])

                        # GET CEO WORKER GAP DATA BASED ON THE PROFIT MARGIN DATA
                        CEO_WORKER_JSON_DATA = global_search_obj.Filter_CeoWorker_Data(device_type ,brand_name , production_year)

                    return ProfitProductResponse("success", json_output, CEO_WORKER_JSON_DATA)
                
            # Function -1
            Embedding_df  = Profit_Obj.apply_embedding()            # call function to get embedding df
            #print("Embedding_df : \n ", Embedding_df[["Brand", "Product Name", "Product Type", "Production Year", "Gender", "Category", "Type Mapped", "similarity_score"]].iloc[0:50])
            #print()

             # Filter out dataframe if similarity score greater than threshold Value
            Embedding_df = Embedding_df.loc[Embedding_df["similarity_score"] > PROFIT_MARGIN_SIMILARITY_SCORE]   
            
            if Embedding_df.empty:
                return ProfitProductResponse("No Data Matched", [], [])

            # Function -2
            paramter_dict , matched_row_data_dict = Profit_Obj.GetMatchedRow_AndParameter(Embedding_df,target_year)     # Get matched row parameter dict
            #print("matched_row_data_dict=========================>",paramter_dict,matched_row_data_dict)

            # create a dataframe from matched row data dict
            searched_df = pd.DataFrame([matched_row_data_dict])

            # if searched dataframe is empty  return empty json 
            if searched_df.empty:
                return ProfitProductResponse("failed",[], [])

            # Remove unneccary columns from searched dataframe
            searched_df = searched_df.drop(columns=["text", 'similarity_score','brand_embedding', 'brand'], errors="ignore", axis=1)
            matched_row_json = searched_df.to_dict(orient="records")            # convert json into dict


            # Get Required Parameter from the Matched Dataframe
            brand_name = str(matched_row_json[0]["Brand"]).lower().strip()
            production_year = int(matched_row_json[0]["Production Year"])
            searched_product_name = matched_row_json[0]["Product Name"]
            searched_product_type = matched_row_json[0]["Product Type"]
            ProductName = searched_product_name + searched_product_type


            # GET CEO WORKER GAP DATA BASED ON THE PROFIT MARGIN DATA
            CEO_WORKER_JSON_DATA = global_search_obj.Filter_CeoWorker_Data(device_type ,brand_name , production_year)
            
            # call function to update product track coubnt 
            vistor_track_res = ProductSearch_Object_create_func(brand_name , ProductName , payload.get("tab_type"))
            
            # Function -3
            Product_Category_df = Profit_Obj.Get_Category_based_df(paramter_dict)  

            #print("Product_Category_df : \n", Product_Category_df)
            #print()

            # Return Response if only matched row dataframe is true
            if Product_Category_df.empty:
                return ProfitProductResponse("success",matched_row_json, CEO_WORKER_JSON_DATA)

            # Function -4
            Product_Yearly_df = Profit_Obj.Get_year_based_df(paramter_dict , Product_Category_df) 
            #print("Product_Yearly_df : \n ", Product_Yearly_df[["Brand", "Product Name", "Product Type", "Production Year", "Gender", "Category", "Type Mapped"]])
            #print()

            # Return Response if only matched row dataframe is true
            if Product_Yearly_df.empty:
                return ProfitProductResponse("success",matched_row_json, CEO_WORKER_JSON_DATA)

            # Function -5
            Product_Gender_df = Profit_Obj.Get_gender_based_df(paramter_dict , Product_Yearly_df) 
            #print("Product_Gender_df : \n ", Product_Gender_df[["Brand", "Product Name", "Product Type", "Production Year", "Gender", "Category", "Type Mapped"]].iloc[50:90])

            if Product_Gender_df.empty:
                return ProfitProductResponse("success",matched_row_json, CEO_WORKER_JSON_DATA)

            # Function -6
            brand_product_type_list= Profit_Obj.Filter_rows_list(paramter_dict , Product_Gender_df) 

            # Function -7 
            filtered_df = Profit_Obj.Filtered_Dataframe(brand_product_type_list)

            if isinstance(filtered_df , list):
                print("-- Skipping There is no any compare data found")
                return ProfitProductResponse('success', searched_df.to_dict(orient="records"), [])

            # Drop Unneccessary columns if it filtered_df is dataframe
            if isinstance(filtered_df , pd.DataFrame) and not filtered_df.empty:
                filtered_df = filtered_df.drop(columns=["text","similarity_score", "text_embedding", "brand_embedding", "brand"],  errors="ignore")      # remove unneccessary dataframe

            #Add percentage sign
            filtered_df["Profit Margin"] = filtered_df["Profit Margin"].astype(float).map(lambda x: f"{x:.2f} %")
 
            # Merge bot dataframe
            merge_df = pd.concat([searched_df , filtered_df], ignore_index=True)    # concat both dataframe  
            
            # Only return three product in API
            if len(merge_df) > 3:
                merge_df = merge_df.iloc[0:3]

            return ProfitProductResponse('success', merge_df.to_dict(orient="records"), CEO_WORKER_JSON_DATA)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)

# API For get all profit margin data
class GetProfitMarginData(APIView):# #""
    def get(self,format=None):
        try:
            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd(), "static", "media" , "Profit Data", 'profit_margin.csv')
            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Read csv 
            df = pd.read_csv(input_csv_file_path)
            
            # Remove Extra spaces from the column Name
            df.columns = df.columns.str.strip()

            # Drop Unneccsary columns
            if "Unnamed: 8" in df.columns:
                df = df.drop("Unnamed: 8", axis=1)

            # Replace NaN/inf values with None so JSON can handle them
            df = df.replace([np.inf, -np.inf], np.nan)   # convert inf to NaN
            df = df.where(pd.notnull(df), None)          # convert NaN to None
            
            # Rename  column Name
            df= df.rename({"Product Type": "Type"}, axis=1)

            df = df.dropna(subset=['Product Name']) 
            df.drop_duplicates(inplace=True) # Remove duplicacy from dataframe  

            # Return Response
            return Response({
                "message": "success" if not df.empty else "failed",
                "status": status.HTTP_200_OK if not df.empty  else 404, 
                "data": df.to_dict(orient="records") if not df.empty else []
            }, status = status.HTTP_200_OK if not df.empty  else 404)
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)


""" ###############################          Tax Avenue Data    ##################################"""
# API to inference Tax trained model
class TaxSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Function to get most matched query 
    def GetMatchedRowDict(self , model , user_query: str ,Input_df : pd.DataFrame, similarity_score : float) -> dict:
        try:
            df =Input_df.copy()

            # Remove dupicate rows
            df = df.drop_duplicates(subset=["Company Name", "Year"])

            print("After ", df.shape)
            query_embedding = model.encode(user_query, convert_to_tensor=True).to(self.device)

            # Convert all full Text  embeddings to tensor
            tax_embeddings = [torch.tensor(e).to(self.device) for e in df['tax_text_embedding']]
            tax_embedding_tensor = torch.stack(tax_embeddings)

            # Cosine similarity on full Text
            fullText_similarities = util.cos_sim(query_embedding, tax_embedding_tensor)[0].cpu().numpy()
            df['tax_similarity'] = fullText_similarities

            # SORT VALUES 
            embedding_df = (df.sort_values('tax_similarity', ascending=False).head(TOP_N))
            
            # Filter Dataframe based on the threshold value

            matched_row = embedding_df.loc[embedding_df["tax_similarity"].idxmax()]
            

           
            filtered_df = embedding_df.loc[embedding_df["tax_similarity"].astype(float) >= similarity_score]
      
            if filtered_df.empty:

                # Clean matched text before searching
                clean_query = re.sub(r"\s+", " ", user_query.strip().lower())
                normalized_matched = embedding_df["text"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
                mask = normalized_matched.str.contains(clean_query, case=False, na=False)

                if mask.any():
                    matched_row = embedding_df[mask]
                    matched_row_data = matched_row.to_dict(orient="records")[0]
                    return matched_row_data
                return []
            
            matched_row = filtered_df.loc[filtered_df["tax_similarity"].idxmax()]
            matched_row_data = matched_row.to_dict()

            return matched_row_data

        except Exception as e:
            exc_type , exc_obj , exc_tb = sys.exc_info()
            error_message = f"[ERROR] Failed to get matched row from dataframe error is : {str(e)} in line no : {exc_tb.tb_lineno}"
            return error_message

    def post(self , request , format=None):
        try:
            # Get User query from POST Request
            required_fields= ['query','tab_type']
            
            # Get payload data 
            payload = request.data

            # Handle missing data 
            missing_fields = [field for field in required_fields if payload.get(field) is None  or not payload.get(field)]
            if missing_fields:
                return Response({
                    'message':f"{', '.join(missing_fields)}: key is required .",
                    'status':status.HTTP_400_BAD_REQUEST
                })
            
            # Take Payload query value in parameter
            user_query = payload.get("query")
         
            # Define paths
            tax_embedding_df_path = os.path.join(os.getcwd(),"static", "media",  "EmbeddingDir", "Tax", "tax_embedding.pkl")
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            
            # Load model
            model = SentenceTransformer(transfer_model_path)

            # Reaf full model and save mode
            df = pd.read_pickle(tax_embedding_df_path)
            original_df = df.copy()

            # Call function to get year status from  the user query ...
            FilterYear= get_year(user_query)
            
            if FilterYear != "None":
                print('filter year ', FilterYear)
                
                # Filter datfarame based on the year
                filtered_df = df.loc[df["Year"].astype(int) == int(FilterYear)]

                #print(filtered_df)

                if filtered_df.empty:
                    return DATA_NOT_FOUND(f"No Data Exist of Year : {FilterYear}")
                
                # call function to get most similar row
                MatchedRow = self.GetMatchedRowDict(model , user_query , filtered_df, TAX_CEO_WORKER_YEAR_SIMILARITY)
               
                # RETURN BAD RESPONSE IF MATCHED ROW VARIABLE GET STING ERROR MESSAGE
                if isinstance(MatchedRow , str):
                    return Internal_server_response(MatchedRow)
                
                # RETURN SUCCESS RESPONSE IF MATCHED ROW IS DICT
                elif isinstance(MatchedRow , dict):

                    # Get matched row data 
                    matched_company_name = MatchedRow.get("Company Name")
                    matched_year = MatchedRow.get("Year")
                    product_name = f"{matched_company_name} {matched_year}"

                    # call function to update product track coubnt 
                    vistor_track_res = ProductSearch_Object_create_func(matched_company_name , product_name , payload.get("tab_type"))
                
                    serached_df = pd.DataFrame([MatchedRow])
                    
                    # print("serached_df : \n ", serached_df[["Company Name", "Year", "Taxes Paid", "Taxes Avoided"]])
                    
                    serached_df = serached_df.drop(columns=["tax_similarity", "tax_text_embedding", "text"], axis=1)
                    
                    return ProductResponse("success",serached_df.to_dict(orient="records"))
                
                # IF THERE IS NO DATA RETURN DATA NOT FOUND RESPONSE
                else:
                    return DATA_NOT_FOUND('DATA NOT FOUND')
              
            else:
                
                # Add new column
                MatchedRow = self.GetMatchedRowDict(model , user_query , df , TAX_CEO_WORKER_SIMILARITY)
                  
                # RETURN BAD RESPONSE IF MATCHED ROW VARIABLE GET STING ERROR MESSAGE
                if isinstance(MatchedRow , str):
                    return BAD_RESPONSE(MatchedRow)
                
                # RETURN SUCCESS RESPONSE IF MATCHED ROW IS DICT
                elif isinstance(MatchedRow , dict):

                    CompanyName = str(MatchedRow.get("Company Name")).lower().strip()

                    # call function to update product track coubnt 
                    vistor_track_res = ProductSearch_Object_create_func(CompanyName.title() , CompanyName.title() , payload.get("tab_type"))

                    # FILTERED DATAFRAME BASED ON THE COMPANY NAME
                    filtered_df = original_df.loc[original_df["Company Name"].astype(str).str.lower().str.strip() == CompanyName]

                    # DROP COLUMNS
                    filtered_df = filtered_df.drop(columns=["text", "tax_text_embedding"]).reset_index(drop=True)

                    # SORT DATAFRAME BASED ON THE YEAR COLUMN
                    sorted_df = filtered_df.sort_values(by="Year" , ascending=False)

                    # IF LENGTH OF THE SORTED DATAFRAME GET ONLY FIRST 4 ROWS
                    if len(sorted_df) > 4:
                        sorted_df = sorted_df.iloc[0:4]

                    return ProductResponse("success",sorted_df.to_dict(orient="records"))
                
                
                # IF THERE IS NO DATA RETURN DATA NOT FOUND RESPONSE
                else:
                    return DATA_NOT_FOUND('DATA NOT FOUND')


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            return Internal_server_response(error_message)

# API For get all Tax Avenue data
class TaxAvenueView(APIView):
    def get(self, request):
        try:

            # Get Data from Tax model
            tax_csv_file_path = os.path.join(os.getcwd(), "static", "media" , "Tax Data", 'Tax_Avoidance.csv')

            # Read CSV
            df = pd.read_csv(tax_csv_file_path)

            # Clean NaN and infinity values
            if not df.empty:
                df.dropna(inplace=True)
            
            # Return Response
            return Response({
                "message": "Tax Tab Data get successfully ....",
                "status": status.HTTP_200_OK, 
                "data": df.to_dict(orient="records") if not df.empty else []
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Failed to get Tax Data error is {str(e)} in line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)

""" ###############################          CEO Worker Frontline Data    ##################################"""

# API to inference CEO Worker  data
class CEOWorkerSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def post(self , request , format=None):
        try:
            
            # GET USER QUERY FROM POST REQUEST 
            required_fields= ['query','tab_type', 'device_type']
            
            # Get payload data 
            payload = request.data

            # Handle missing data 
            missing_fields = [field for field in required_fields if payload.get(field) is None  or not payload.get(field)]
            if missing_fields:
                return Response({
                    'message':f"{', '.join(missing_fields)}: key is required .",
                    'status':status.HTTP_400_BAD_REQUEST
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Handle device type value
            device_type =str(payload.get("device_type")).lower().strip()

            if device_type not in ["mobile", "desktop"]:
                return Response({
                    "message": "Invalid device type , Please choose one from them ['mobile' , 'desktop']" ,
                    "status": 400,
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Take Payload query value in parameter
            user_query = payload.get("query")

            # Paths 
            ceo_phone_embedding_df_path = os.path.join(os.getcwd(), "static" , "media" , "EmbeddingDir", "CEO-Worker", "ceo_phone_embedding.pkl")
            ceo_desktop_embedding_df_path = os.path.join(os.getcwd(),  "static" , "media" , "EmbeddingDir", "CEO-Worker", "ceo_desktop_embedding.pkl")

            # Load model
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            model = SentenceTransformer(transfer_model_path)

            # MAKE OBJECT OF TAX SEMANTIC SEARCH CLASS FUNCTION 
            tax_obj = TaxSemanticSearchView()

            df = pd.DataFrame()
            # Reaf PHONE DATAFRAME
            if device_type =="mobile":
                df = pd.read_pickle(ceo_phone_embedding_df_path)
                df = df.rename(columns={'phone_text_embedding': 'tax_text_embedding', 'phone_text': 'text'})

            else:
                df = pd.read_pickle(ceo_desktop_embedding_df_path)
                df = df.rename(columns={'desktop_text_embedding': 'tax_text_embedding','desktop_text': 'text'})

            # Remove Unnamed: 0 columns
            if  'Unnamed: 0' in df.columns:
                df = df.drop("Unnamed: 0", axis=1)


            # Remove dupicate rows from the CEO WORKER csv
            df = df.drop_duplicates(subset=["Company Name", "Year", "CEO Name"])

            # make a copy of original dataframe
            original_df = df.copy()

            # Call function to get year status from  the user query ...
            FilterYear= get_year(user_query)

            if FilterYear != "None":
                print('filter year ', FilterYear)
                
                # Filter datfarame based on the year
                filtered_df = df.loc[df["Year"].astype(int) == int(FilterYear)]

                if filtered_df.empty:
                    return DATA_NOT_FOUND(f"No Data Exist of Year : {FilterYear}")
                
                # call function to get most similar row
                MatchedRow = tax_obj.GetMatchedRowDict(model , user_query , filtered_df, TAX_CEO_WORKER_YEAR_SIMILARITY)
                
                # RETURN BAD RESPONSE IF MATCHED ROW VARIABLE GET STING ERROR MESSAGE
                if isinstance(MatchedRow , str):
                    return Internal_server_response(MatchedRow)
                
                # RETURN SUCCESS RESPONSE IF MATCHED ROW IS DICT
                elif isinstance(MatchedRow , dict):

                    # Get matched row data 
                    matched_company_name = MatchedRow.get("Company Name")
                    matched_ceo_name = MatchedRow.get("CEO Name")
                    matched_year = MatchedRow.get("Year")
                    brand_name = f"{matched_company_name} {matched_year}"

                    # call function to update product track coubnt 
                    vistor_track_res = ProductSearch_Object_create_func(brand_name , matched_ceo_name , payload.get("tab_type"))
                    searched_df = pd.DataFrame([MatchedRow])
                    
                    #print("searched_df : \n ", searched_df[["Company Name", "Year", "CEO Name", "CEO Total Compensation", "Frontline Worker Salary"]])
                    searched_df = searched_df.drop(columns=["tax_similarity", "tax_text_embedding", "text"])
                    
                    return ProductResponse("success",searched_df.to_dict(orient="records"))
                
                # IF THERE IS NO DATA RETURN DATA NOT FOUND RESPONSE
                else:
                    return DATA_NOT_FOUND('DATA NOT FOUND')
            else:
                # Add new column
                MatchedRow = tax_obj.GetMatchedRowDict(model , user_query , df , TAX_CEO_WORKER_SIMILARITY)

                # RETURN BAD RESPONSE IF MATCHED ROW VARIABLE GET STING ERROR MESSAGE
                if isinstance(MatchedRow , str):
                    return Internal_server_response(MatchedRow)
                
                # RETURN SUCCESS RESPONSE IF MATCHED ROW IS DICT
                elif isinstance(MatchedRow , dict):

                    CompanyName = str(MatchedRow.get("Company Name")).lower().strip()
                    CEOWorkerName = str(MatchedRow.get("CEO Name")).lower().strip()

                    vistor_track_res = ProductSearch_Object_create_func(CompanyName.title() , CEOWorkerName.title() , payload.get("tab_type"))


                    # FILTERED DATAFRAME BASED ON THE COMPANY NAME
                    filtered_df = original_df.loc[original_df["Company Name"].astype(str).str.lower().str.strip() == CompanyName]

                    # DROP COLUMNS
                    filtered_df = filtered_df.drop(columns=["text", "tax_text_embedding"]).reset_index(drop=True)

                    # SORT DATAFRAME BASED ON THE YEAR COLUMN
                    sorted_df = filtered_df.sort_values(by="Year" , ascending=False)
                    
                    #print("sorted_df : \n ", sorted_df[["Company Name", "Year", "CEO Name", "CEO Total Compensation", "Frontline Worker Salary"]])

                    # IF LENGTH OF THE SORTED DATAFRAME GET ONLY FIRST 4 ROWS
                    if len(sorted_df) > 4:
                        sorted_df = sorted_df.iloc[0:4]

                    return ProductResponse("success",sorted_df.to_dict(orient="records"))
                
                # IF THERE IS NO DATA RETURN DATA NOT FOUND RESPONSE
                else:
                    return DATA_NOT_FOUND('DATA NOT FOUND')
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            return Internal_server_response(error_message)

# API For get CEO Worker data
class CeoWorkerView(APIView):
    def get(self, request):
        try:

            # Get CEO WORKER Data from DATABASE
            ceo_worker_file_path = os.path.join(os.getcwd() , "static", "media", "CEO Worker Data", "Website.csv")

            # Read csv 
            df = pd.read_csv(ceo_worker_file_path)
            
            # Handle if there is no data found
            if df.empty:
                return DATA_NOT_FOUND("No Data Found for CEO Worker Tab ")
            
            # Clean NaN and infinity values
            if not df.empty:
                df.dropna(inplace=True)
            
            # Return Response
            return Response({
                "message": "CEO Worker Tab Data get successfully ....",
                "status": status.HTTP_200_OK , 
                "data": df.to_dict(orient="records") if not df.empty else []
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)
        

"""                 #######################          CSV RELATED API's              ###########################################               """
# Use Django SECRET_KEY or define a custom one
class AdminAuthenticationView(APIView):
    def post(self, request, format=None):
        raw_password = request.data.get('password')

        if not raw_password:
            return Response(
                {
                    'message': "Password is required.",
                    'status': status.HTTP_400_BAD_REQUEST
                }
            )

        admin_obj = AdminAuthenticationModel.objects.first()
        if not admin_obj:
            return Response({"message": "Admin not found", "status": 400})

        if not check_password(raw_password, admin_obj.password):
            return Response({"message": "Incorrect Password", "status": 400})

        #  Payload for JWT
        payload = {
            "admin_id": admin_obj.id,
            "exp": datetime.utcnow() + timedelta(minutes=30),  # expires in 30 min
            "iat": datetime.utcnow()
        }

        secret_key = os.getenv("SECRET_KEY", "default_secret")  # fallback
        algorithm = os.getenv("ALGORITHM", "HS256")

        token = jwt.encode(payload, secret_key, algorithm=algorithm)

        # If jwt.encode returns bytes (older PyJWT)
        if isinstance(token, bytes):
            token = token.decode("utf-8")

        return Response({
            "message": "Login Successfully...",
            "status": 200,
            "token": token
        })

# API to validate token 
class TokenProtectedView(APIView):
    def get(self, request, *args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return Response({"error": "No token provided"}, status=status.HTTP_401_UNAUTHORIZED)

        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token.split(" ")[1]

        validation = validate_token(token)

        if not validation["valid"]:
            return Response({"error": validation["error"]}, status=status.HTTP_401_UNAUTHORIZED)

        return Response({"message": "Token is Valid", "payload": validation["payload"]},  status=status.HTTP_202_ACCEPTED)


# APi to get product Visitor Track count data
class TrackProductSearchCount(APIView):
    def get(self,request):
        try:

            date_str = request.GET.get("date")

            Product_Data_obj = ProductSearchTrack.objects.all().values()

            if not Product_Data_obj:
                return DATA_NOT_FOUND("No data found .")
            
            df = pd.DataFrame(list(Product_Data_obj))
            
            df["brand_name"] = df["brand_name"].str.title()
            df["product_name"] = df["product_name"].str.title()

            # Convert to datetime
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["date_only"] = df["created_at"].dt.date

            df = df.loc[df["date_only"].astype(str) == str(date_str)]
            df = df.drop(columns=["created_at","updated_at", "date_only"], axis=1)

            total_visits = df["search_count"].sum()
           
            grouped = df.groupby('tab_type').apply(lambda group: group.to_dict(orient='records')).to_dict()
            result_data = [{tab: records} for tab, records in grouped.items()]

            return Response({
                "message": f"No Data Found for Date : {date_str}" if df.empty else f"Data get successfully of date : {date_str}" ,
                "status": status.HTTP_200_OK,
                "total_search": total_visits,
                # "data": df.to_dict(orient="records"),
                "data": result_data
            })
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)

# Api for Track Visiotor count
class TrackVisitorCount(APIView):
    def post(self , request , format= None):
        try:
            user_browser_id = request.data.get("browser_id")
            if not user_browser_id:
                return BAD_RESPONSE(f"Browser ID is required , please send id with using key : 'browser_id'. ")
            
            # function to track vistor count 
            current_date = datetime.now().date()
            track_visitor_obj, created = VistorTrackCountModel.objects.get_or_create(
                visit_date=current_date,
                user_browser_id=user_browser_id,
                defaults={
                    "visit_count": 1  # start from 1 if new record
                }
            )
            if not created:  # if it already exists, increment
                today = date.today() 
                """ Added a condition to check, if the user requesting the browser in the same day it'll not increase the count for the browser to the user."""
                if track_visitor_obj.visit_date != today:
                    track_visitor_obj.visit_count += 1
                    track_visitor_obj.save()
                

            return Response({
                "message": "success",
                "status": status.HTTP_200_OK
            },status=status.HTTP_200_OK)

        except Exception as e:
            exc_type , exc_obj , exc_tb = sys.exc_info()
            error_message = f"[ERROR] Failed to create track  visitor count  Record, error ocuur : {str(e)} in line no : {exc_tb.tb_lineno}"
            return Internal_server_response(error_message)

# API FOR GET TRACK VISTOR BASED ON DATE
class GetVistorView(APIView):
    def get(self,format=None):
        try:
            queryset = VistorTrackCountModel.objects.all().values()
            # If data not found
            if not queryset:
                return Response({
                    "message": "Data Not Found",
                    "status": 400
                }, status=400)

            # Create dataframe
            df = pd.DataFrame(list(queryset))
            total_visits = df["visit_count"].sum()

            # Get Today's Visitors
            today = date.today() 
            df['visit_date'] = pd.to_datetime(df['visit_date']).dt.date
            filtered_df = df[df['visit_date'] == today]
            print(filtered_df)
            todays_visitiors = filtered_df["visit_count"].sum()


            return Response({
                "message":"Data get successfully",
                "status": 200,
                "total_visit_count": total_visits,
                "total_todays_visit_counts": todays_visitiors
            }, status=200)

        except Exception as e:
            exc_type , exc_obj , exc_tb = sys.exc_info()
            error_message = f"[ERROR] Failed to get track  visitor count data, error ocuur : {str(e)} in line no : {exc_tb.tb_lineno}"
            return Internal_server_response(error_message)


"""     ###################################           GLOBAL API'S                    ###############################       """
class GlobalSearchAPIView(APIView):

    # function to filter tax data based on the brand name and year
    def Filter_Tax_Data(self,brand_name: str , year: int)-> list:

        # Tax Embedding DF Path
        tax_embedding_df_path = os.path.join(os.getcwd(),"static" , "media",  "EmbeddingDir", "Tax", "tax_embedding.pkl")
        tax_df = pd.read_pickle(tax_embedding_df_path)

        # Filtered Df
        filtered_tax_df = tax_df.loc[
            (tax_df["Company Name"].str.lower().str.strip().str.contains(brand_name, case=False))&
            (tax_df["Year"].astype(int) == year)
        ]   

        # Drop Unneccessary columns
        filtered_tax_df = filtered_tax_df.drop(columns=['tax_text_embedding','text'],axis=1)

        # Convert into json
        json_output = filtered_tax_df.to_dict(orient="records")
        if json_output:
            json_output= json_output[0]

        return json_output
    
    # function to filter ceo worker data based on the brand name and year
    def Filter_CeoWorker_Data(self,device_type : str ,brand_name: str , year: int)-> list:
        
        # CEO Worker  Embedding DF Path
        Ceo_worker_tablet_csv_path= os.path.join(os.getcwd(),"static" , "media", "CEO Worker Data", "Phone_Tablet.csv")
        Ceo_worker_website_path = os.path.join(os.getcwd(), "static" , "media" ,"CEO Worker Data","Website.csv")

        # Read CSV
        tablet_df = pd.read_csv(Ceo_worker_tablet_csv_path)
        website_df = pd.read_csv(Ceo_worker_website_path)

        # Take Empty dataframe
        df = pd.DataFrame()
        if device_type =="mobile":
            df = tablet_df
        else:
            df = website_df
        
        #Filtered Df
        filtered_ceo_worker_df = df.loc[
            (df["Company Name"].str.lower().str.strip() == brand_name)&
            (df["Year"].astype(int) == year)
        ]   

        # Convert into json
        json_output = filtered_ceo_worker_df.to_dict(orient="records")
        return json_output


    def post(self,request, format=None):
        try:
            # Get Query from User
            required_field =["query", "device_type", "target_year"]

            # get payload
            payload = request.data
            missing_fields = [ field for field in required_field if payload.get(field) is None or not payload.get(field)]
            
            if missing_fields:
                return Response({
                    "message": f'{", ".join(missing_fields)} key is required. ',
                    "status": 400,
                }, status=status.HTTP_400_BAD_REQUEST)
            

            device_type =str(payload.get("device_type")).lower().strip()
            

            if device_type not in ["mobile", "desktop"]:
                return Response({
                    "message": "Invalid device type , Please choose one from them ['mobile' , 'desktop']" ,
                    "status": 400,
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Import Base url from the setting file
            target_year =int(payload.get("target_year"))

            # Payload data
            data = {"query":payload.get("query") , "tab_type": "profit", "device_type":device_type , "target_year":target_year }
            headers = {
                "content_type": "application/json"
            }

            # API URL
            url = f"{BASE_URL}/product-semantic-search"
            print(f"Api is hitting on url: {url}")

            response = requests.post(url , json= data , headers=headers)

            # Check if response status is 200
            if response.status_code ==200:
                
                # Get response data.
                response_text = response.text

                # Handle if output json type is string
                if isinstance(response_text , str):
                    import ast
                    response_text = ast.literal_eval(response_text)
                
                # Get data response
                response_status = response_text["status"]
                if response_status == 200:
                    
                    # Get json data 
                    json_data = response_text["data"]

                    # check if json data length is true
                    if len(json_data) > 0:

                        # get first matched data row
                        matched_row = json_data[0]

                        Brand_name = str(matched_row["Brand"]).lower().strip()# Brand name
                        Year = int(matched_row["Production Year"]) # Year

                        # Filter OUT Tax Data
                        tax_data_json = self.Filter_Tax_Data(Brand_name, Year)
                        ceo_worker_data = self.Filter_CeoWorker_Data(device_type ,Brand_name, Year)

                        return Response({
                            "message": "success",
                            "status": 200,
                            "data": response_text["data"],
                            "tax_data": tax_data_json if tax_data_json else [],
                            "ceo_worker_data": ceo_worker_data if ceo_worker_data else []

                        })

                    # Return bad response if no data found
                    else:
                        return Response({
                            "message": "Data not found",
                            "status": 404
                        }, status=status.HTTP_404_NOT_FOUND)
                    
                else:
                    return Response({
                        "message": response_text["message"],
                        "status": response_status
                    }, status=response_status)
                
            else :
                return Response({
                    "message": 'Getting issue in product semantic search api'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            exc_type , exc_obj , exc_tb = sys.exc_info()
            error_message = f"[ERROR] , Failed to Read Gloabl search Query, error occur is : {str(e)} in line no : {exc_tb.tb_lineno}"
            print(error_message)
            return error_message

# APi for send file url
class DataFilesSync(APIView):
    def get(self, request, format=None):
        try:
            base_path = os.path.join(os.getcwd(), "static", "media")

            # Define your files
            files = [
                ("Profit Data", "profit_margin.csv"),
                ("Tax Data", "Tax_Avoidance.csv"),
                ("CEO Worker Data", "Phone_Tablet.csv"),
                ("CEO Worker Data", "Website.csv"),
            ]

            file_list = []

            for folder, filename in files:
                file_path = os.path.join(base_path, folder, filename)
                file_url = file_path.replace(os.getcwd(), BASE_URL)
                
                # check if host live then replace static string
                if HOST =="live":
                    file_url = file_url.replace("/static", "")

                file_list.append({
                    "filename": filename,
                    "file_url": file_url
                })

            return Response({"files": file_list, "status": 200}, status=200)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Failed to load data files, error: {str(e)} in line {exc_tb.tb_lineno}"
            return Internal_server_response(error_message)

# APi for train model
class TrainModelView(APIView):

    # function to return transfer model dir path
    def transfer_model_base_dir_path(self):
        TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
        os.makedirs(TransferModelDir , exist_ok=True)
        return TransferModelDir

    
    # function to return Embedding Dir Path
    def embedding_model_base_dir_path(self, tab_type):
        embedding_base_dir ="Profit Margin"

        if tab_type == "tax":
            embedding_base_dir=  "Tax"

        elif tab_type =="phone" or tab_type =="desktop":
            embedding_base_dir = "CEO-Worker"

        Emedding_dir_path = os.path.join(os.getcwd() ,"static", "media", "EmbeddingDir", embedding_base_dir)
        os.makedirs(Emedding_dir_path , exist_ok=True)
        return Emedding_dir_path
    
    def get_existing_file_path(self, tab_type):
        # Saved dir 
        default_filename = os.path.join("Profit Data" , "profit_margin.csv")
        if tab_type =="tax":
            default_filename =os.path.join("Tax Data" , "Tax_Avoidance.csv")

        elif tab_type =="phone":
            default_filename =os.path.join("CEO Worker Data" , "Phone_Tablet.csv")

        elif tab_type =="desktop":
            default_filename =os.path.join("CEO Worker Data" , "Website.csv")

        existing_file_path = os.path.join(os.getcwd() , "static", "media", default_filename)
        return existing_file_path
    
        
    def Handle_invalid_filename(self,filename, tab_type):
        file_names_array = ["Website.csv", "Phone_Tablet.csv"]

        correct_filename = 'profit_margin.csv'
        if tab_type == 'tax' and filename != 'Tax_Avoidance.csv':
            correct_filename =  'Tax_Avoidance.csv'
        elif tab_type =="phone"  and filename not in file_names_array:
            correct_filename = "Phone_Tablet.csv"
        
        elif tab_type =="desktop" and filename not in file_names_array:
            correct_filename = "Website.csv"
        return correct_filename


    def post(self , request , format=None):
        try:
            file_names_array = ["profit_margin.csv", "Tax_Avoidance.csv", "Website.csv", "Phone_Tablet.csv"]
            tab_types =["profit" , "tax", "phone", "desktop"]

            # Get Payload 
            payload = request.data
            uploaded_file = request.FILES.get("file")

            # Required Fields 
            required_fields = ["tab_type"]
            missing_fields = [field for field in required_fields if field not in payload or payload.get(field) is None]
            if missing_fields:
                return BAD_RESPONSE(f"{','.join(missing_fields)} key is required.")
            
            TAB_TYPE = payload.get("tab_type")
            
            # Check if Upload file is None
            if not uploaded_file:
                return BAD_RESPONSE(f"Please Select file before make request , use 'file' key to upload file")
            
            # Check file Type
            file_type_status = check_suffix(uploaded_file)
            if not file_type_status:
                return FILE_NOT_ACCEPTABLE_RESPONSE( "Invalid File Type , Only CSV file Accepted.")

            # check if filename is not matched
            if uploaded_file.name not in file_names_array:
                corrected_filename = self.Handle_invalid_filename(uploaded_file.name, TAB_TYPE)
                return BAD_RESPONSE(f"Invalid file name. Correct file name is : {corrected_filename}")
            
            # GET TAB TYPE and check it is correct or not 
           
            if TAB_TYPE not in tab_types:
                return BAD_RESPONSE(f"Invalid Tab Type , Valid Tab type is {', '.join(tab_types)}")
            
            # Check dataframe columns 
            new_df = pd.read_csv(uploaded_file)
            column_status , expected_columns = check_columns(TAB_TYPE ,new_df)
            if not column_status:
                return FILE_NOT_ACCEPTABLE_RESPONSE(f"Columns does not matched , Accepted columns list is : {expected_columns}")

            # Get Existing CSV file 
            existing_file_path = self.get_existing_file_path(TAB_TYPE)
            
            df1 = pd.read_csv(existing_file_path)

            # Remove extra spaces from the columns
            df1.columns = df1.columns.str.strip()
            new_df.columns = new_df.columns.str.strip()

            # skip list based on TAB_TYPE
            skip_cols = ["Link to Product Pictures"] if TAB_TYPE == "profit" else []

            # 1) Make copies for comparison only
            df1_cmp = df1.copy()
            new_cmp = new_df.copy()

            # 2) Work only on columns common to both dataframes
            common_cols = df1.columns.intersection(new_df.columns)

            for col in common_cols:
                if col in skip_cols:
                    continue
                # only lowercase string-like columns (safe check)
                if pd.api.types.is_string_dtype(df1[col]) or pd.api.types.is_string_dtype(new_df[col]):
                    # use .where to preserve NaN (don't convert NaN to 'nan')
                    df1_cmp[col] = df1[col].where(df1[col].notna(), None).astype(object).map(
                        lambda x: x.lower().strip() if isinstance(x, str) else x
                    )
                    new_cmp[col] = new_df[col].where(new_df[col].notna(), None).astype(object).map(
                        lambda x: x.lower().strip() if isinstance(x, str) else x
                    )

            # Find rows in new_df that are not present in df1 (comparison on the *_cmp copies)
            mask = ~new_cmp.apply(tuple, axis=1).isin(df1_cmp.apply(tuple, axis=1))
            df2_new = new_df[mask].copy()   # IMPORTANT: select rows from original new_df to preserve original case & skipped cols
            
            #  Merge and save only if new rows exist
            if not df2_new.empty:
                # Optionally drop duplicates inside df2_new itself (if needed)
                df2_new = df2_new.drop_duplicates(ignore_index=True)

                # Merge df
                merged = pd.concat([df1, df2_new], ignore_index=True)

                merged.to_csv(existing_file_path, index=False)
                print(f"File updated with {len(df2_new)} new rows ")
            else:
                df1.to_csv(existing_file_path , index=False)

            print("File saved with new data ....")

            if TAB_TYPE =="profit":
                # Call a function from product stucture file 
                model_response = AllProductDetailMain(self.embedding_model_base_dir_path(TAB_TYPE), self.transfer_model_base_dir_path(), existing_file_path)

                # send response when response return Dataframe
                if isinstance(model_response , pd.DataFrame):
                    return Response({
                    "message": "Model Trained successfully with Profit Margin Data ",
                    "status": status.HTTP_200_OK
                    })
                
                # Handle when empty list comes
                elif isinstance(model_response , list) and not model_response:
                    return DATA_NOT_FOUND("No Data found for Profit Margin Data Tab")

                # Handle when error message come
                elif isinstance(model_response , str):
                    return Internal_server_response(model_response)
            
            elif TAB_TYPE == "tax":
                TaxModelResponse = TaxMainFunc(existing_file_path, self.embedding_model_base_dir_path(TAB_TYPE), self.transfer_model_base_dir_path())

                if TaxModelResponse =="success":
                    TaxModelResponse = "Tax Model Train successfully ..."

                return Response({
                    "message": TaxModelResponse,
                }, status=status.HTTP_200_OK)

            elif TAB_TYPE =="phone" or TAB_TYPE=="desktop":
                #CSV file name
                Tablet_File_path= os.path.join(os.getcwd(), "static", "media", "CEO Worker Data","Phone_Tablet.csv")
                Website_File_path= os.path.join(os.getcwd(), "static", "media", "CEO Worker Data","Website.csv")

                # Transformer model 
                TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
                os.makedirs(TransferModelDir , exist_ok=True)
                
                # Call function to train model with all rows
                CeoworkerModelResponse = CeoWorkerMainFunc(Tablet_File_path,Website_File_path , self.embedding_model_base_dir_path(TAB_TYPE), self.transfer_model_base_dir_path())
                
                if CeoworkerModelResponse =="success":
                    CeoworkerModelResponse = "Model Train successfully ..."
                return Response({
                    "status": status.HTTP_200_OK,
                    "message": CeoworkerModelResponse,
                })

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Failed to upload new data files, error: {str(e)} in line {exc_tb.tb_lineno}"
            return Internal_server_response(error_message)
        
