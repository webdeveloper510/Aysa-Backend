from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response

# Import python packages
import os 
import sys
from pathlib import Path
import pandas as pd
import torch

# import Project files
from .response import BAD_RESPONSE , Success_RESPONSE , DATA_NOT_FOUND, ProductResponse
from .models import *
from .utils import *
from .product_structure import *
from .tax_structure import *
from .ceo_worker import *
from sentence_transformers import SentenceTransformer , util

""" ###############################          Profit Margin Data    ##################################"""
# API FOR PRODUCT DATA TRAIN 
class ProductTrainPipeline(APIView):
    def get(self,format =None):
        try:
            # Vector Database dir path
            Emedding_dir_path = os.path.join(os.getcwd() ,"EmbeddingDir", "Profit_Margin")
            os.makedirs(Emedding_dir_path , exist_ok=True)

            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'profit_margins.csv')
            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Transformer model 
            TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
            os.makedirs(TransferModelDir , exist_ok=True)
            
            # Call function to train model with all rows
            model_response = AllProductDetailMain(input_csv_file_path, Emedding_dir_path, TransferModelDir)
            
            if model_response is None:
                return Response({"message": "Failed","status": status.HTTP_500_INTERNAL_SERVER_ERROR})

            return Response({
                "message": model_response,
            }, status=status.HTTP_200_OK)
        

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to train Model error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            return Response({
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "message": error_message
            })

# API to inference product trained model
class ProductSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_n = 80
    similarity = 0.75

    def drop_unnecessary_cols(self,df):
        drop_cols = ["Type","brand_similarity", "brand", "text", "full_text_similarity", "brand_embedding"]
        return df.drop(columns=drop_cols, errors="ignore")

    # function to change profit margin in dataframe
    def convert_profit_margin(self,df):
        # Remove all characters except digits and dot
        df["Profit Margin"] = (
            df["Profit Margin"]
            .astype(str)
            .str.replace(r"[^0-9.-]", "", regex=True)  # keep digits, minus, dot
            .replace("", "0")  # if empty string, set to 0
            .astype(float)
        )
        return df


    # function to get highest mmargin of unique brand
    def get_highest_margin_per_brand(self,df):
        result = []
        for _, group in df.groupby("Brand"):
            highest_margin = group.loc[group["Profit Margin"].idxmax()]
            result.append(highest_margin)
        return pd.DataFrame(result).reset_index(drop=True)
    
    @staticmethod
    def compare_products(df ,BrandName,Product_type ,matched_year):
        compare_rows =[]
        for row_dict in df.to_dict(orient="records"):
            Brand = str(row_dict.get("Brand", "")).lower().strip()
            Year = int(row_dict.get("Production Year"))
            ProductType = str(row_dict.get("Type Mapped", "")).lower().strip()

            # Compare Rows according to data 
            if BrandName  != Brand and Product_type == ProductType and matched_year ==Year:
                compare_rows.append(row_dict)
        return compare_rows

    # function to get product search
    def ProductSearch(self ,user_query, full_embedding_df_path , model):
        try:
 
            # Reaf full model and save mode
            df = pd.read_pickle(full_embedding_df_path)

            original_df = df.copy()

            # convert_user_query in embedding 
            query_embedding = model.encode(user_query, convert_to_tensor=True).to(self.device)

            # Convert all full Text  embeddings to tensor
            full_embeddings = [torch.tensor(e).to(self.device) for e in df['full_text_embedding']]
            full_text_embedding_tensor = torch.stack(full_embeddings)

            # Convert all sub Text  embeddings to tensor
            Brand_embeddings = [torch.tensor(e).to(self.device) for e in df['brand_embedding']]
            brand_embedding_tensor = torch.stack(Brand_embeddings)

            # Cosine similarity on full Text
            fullText_similarities = util.cos_sim(query_embedding, full_text_embedding_tensor)[0].cpu().numpy()
            df['full_text_similarity'] = fullText_similarities

            # Cosine similarity on Sub Text
            Brand_similarities = util.cos_sim(query_embedding, brand_embedding_tensor)[0].cpu().numpy()
            df["brand_similarity"] = Brand_similarities

            # remove unnessary columns from the Embeddingf df
            embedding_df = (
                df.drop(columns=["full_text_embedding", "brand_embedding", 'brand_similarity'])
                .sort_values('full_text_similarity', ascending=False)
                .head(self.top_n)
            )

            # remove unneccsary columns from original dataframe
            original_df = original_df.drop(columns=["full_text_embedding", "brand_embedding"], errors="ignore")

            return embedding_df , original_df
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed tp get full and sub embed dataframe [ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return None
   
   # Main function 
    def post(self, request, format=None):
        try:
            user_query = request.data.get("query")
            if not user_query:
                return BAD_RESPONSE("User input is required. Please provide it with the key name: 'query'")

            split_query = user_query.split(" ")

            # Define paths
            full_embedding_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "Profit_Margin", "profit_embedding.pkl")
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')

            # Load model
            model = SentenceTransformer(transfer_model_path)

            # Get embeddings and original data
            embedding_df, original_df = self.ProductSearch(user_query, full_embedding_df_path, model)

            # Normalize Profit margin column value in same format
            embedding_df["Profit Margin"] = embedding_df["Profit Margin"].apply(format_profit_margin)
            original_df["Profit Margin"] = original_df["Profit Margin"].apply(format_profit_margin)

            
            # Handle if embedding df is empty 
            if embedding_df.empty:
                return ProductResponse("failed", [])

            # Most similar row
            matched_row = embedding_df.loc[embedding_df["full_text_similarity"].idxmax()]
            matched_row_data = matched_row.to_dict()

            # get values from the matched dataframe
            BrandName = str(matched_row_data.get("Brand", "")).lower().strip()
            Product_type = str(matched_row_data.get("Type Mapped", "")).lower().strip()
            matched_year = int(matched_row_data.get("Production Year"))

            print({
                "BrandName": BrandName,
                "Product_type": Product_type,
                "matched_year": matched_year,
    
            })

            # Handle if user ask only about brand
            if len(split_query) == 1:
                print("Only ask about single brand")
                cleaned_split_query = str(split_query[0]).lower().strip()

                filtered_df = embedding_df.loc[embedding_df["Brand"].str.lower().str.strip() == cleaned_split_query]

                # Check if user input exists in Brand column
                if not filtered_df.empty:
                    # convert profit margin if needed
                    filtered_df = self.convert_profit_margin(filtered_df)

                    # sort by year (latest first) and margin (highest first)
                    filtered_df = filtered_df.sort_values(
                        ['Production Year', 'Profit Margin'], ascending=[False, False]
                    )

                    # drop embedding col if present
                    if "brand_embedding" in filtered_df.columns:
                        filtered_df = filtered_df.drop(columns=["brand_embedding"], axis=1)

                    print("filtered df")
                    print(filtered_df)

                    # Case 1: only one year → keep top 3 rows from that year
                    if filtered_df['Production Year'].astype(int).nunique() == 1:
                        filtered_df = filtered_df.drop_duplicates(subset=["Type Mapped"], keep="first")
                        filtered_df = filtered_df.head(3)
                    else:
                        # Case 2: multiple years → keep top 1 per year (latest 3 years)
                        filtered_df = filtered_df.drop_duplicates(subset=["Production Year"], keep="first")
                        filtered_df = filtered_df.head(3)

                    #  Drop unnecessary cols ONLY at the end, not before trimming
                    filtered_df = self.drop_unnecessary_cols(filtered_df)

                    return ProductResponse("success", filtered_df.to_dict(orient="records"))

                
            print("Hit for compare products ....")
            #compare rows based on the "Brand , Production Year , Category , Type"

  
            # filter data based on the Brand Name , Type and Production Year  from the embedding dataframe
            filtered_df = embedding_df.loc[
                (embedding_df["Brand"].str.lower().str.strip() != BrandName) &
                (embedding_df["Type Mapped"].str.lower().str.strip() == Product_type) &
                (embedding_df["Production Year"].astype(int) == matched_year) 
                ].copy()
            
            print("Embedding filtered dataframe 1 ")
            print(filtered_df)
            print()

            # if embdding dataframe does not contain values then 
            # Filter out data from original dataframe 
            if filtered_df.empty or len(filtered_df) <2:
                filtered_df = original_df.loc[
                    (original_df["Brand"].str.lower().str.strip() != BrandName) &
                    (original_df["Type Mapped"].str.lower().str.strip() == Product_type) &
                    (original_df["Production Year"].astype(int) == matched_year) 
                
                ].copy()

            print("original dataframe  filtered dataframe 2 ")
            print(filtered_df)
            print()

            # get highest and minimum profit margin
            if not filtered_df.empty:
                filtered_df = self.convert_profit_margin(filtered_df)
                # Get row with highest margin
                highest = filtered_df.loc[filtered_df["Profit Margin"].idxmax()]

                # Get row with lowest margin
                lowest = filtered_df.loc[filtered_df["Profit Margin"].idxmin()]

                # Combine into a single dataframe
                filtered_df = pd.DataFrame([highest, lowest])

                filtered_df = filtered_df.sort_values('Profit Margin', ascending=False)

            # Add percentage sign after the value
            filtered_df["Profit Margin"] = filtered_df["Profit Margin"].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
            )

            # Convert series into dataframe
            matched_df = pd.DataFrame([matched_row])

            # Merge Both Dataframe
            merge_df = pd.concat([matched_df, filtered_df]).reset_index(drop=True)
            merge_df = self.drop_unnecessary_cols(merge_df)
            
            # Rename column
            merge_df = merge_df.rename(columns={'Type Mapped': 'Type'})

            if len(merge_df) > 3:
                merge_df = merge_df.iloc[0:3]

            return ProductResponse("success", merge_df.to_dict(orient="records"))

             
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)

            return Response({
                "message": error_message,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "data": [],
            })


# API For get all profit margin data
class GetProfitMarginData(APIView):# #
    def get(self,format=None):
        try:
            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'profit_margins.csv')
            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Read csv 
            df = pd.read_csv(input_csv_file_path)
            
            # Drop Unneccsary columns
            if "Unnamed: 8" in df.columns:
                df = df.drop("Unnamed: 8", axis=1)

            # Replace NaN/inf values with None so JSON can handle them
            df = df.replace([np.inf, -np.inf], np.nan)   # convert inf to NaN
            df = df.where(pd.notnull(df), None)          # convert NaN to None
            
            # Return Response
            return Response({
                "message": "success" if not df.empty else "failed",
                "status": status.HTTP_200_OK if not df.empty  else 404, 
                "data": df.to_dict(orient="records") if not df.empty else []
            })
            

            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            return Response({
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "message": error_message
            })


""" ###############################          Tax Avenue Data    ##################################"""


# API FOR Tax  DATA TRAIN 
class TaxDataTrainPipeline(APIView):
    def get(self,format =None):
        try:
            # Vector Database dir path
            Emedding_dir_path = os.path.join(os.getcwd() ,"EmbeddingDir", "Tax")
            os.makedirs(Emedding_dir_path , exist_ok=True)

            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'Tax_Avoidance.csv')
            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Transformer model 
            TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
            os.makedirs(TransferModelDir , exist_ok=True)
            
            # Call function to train model with all rows
            TaxModelResponse = TaxMainFunc(input_csv_file_path, Emedding_dir_path, TransferModelDir)
            

            return Response({
                "message": TaxModelResponse,
            }, status=status.HTTP_200_OK)
        

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to train Model error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            return Response({
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "message": error_message
            })

# API to inference Tax trained model
class TaxSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_n = 80
    similarity = 0.75

    def post(self , request , format=None):
        try:
            
            user_query = request.data.get("query")
            if not user_query:
                return BAD_RESPONSE("user query is required , Please provide with key name : 'query'")
            
            split_query = user_query.split(" ")

            # Define paths
            tax_embedding_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "Tax", "tax_embedding.pkl")
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            
            # Load model
            model = SentenceTransformer(transfer_model_path)

            # Reaf full model and save mode
            df = pd.read_pickle(tax_embedding_df_path)
         
            # make a copy of original dataframe
            original_df = df.copy()

            # convert_user_query in embedding 
            query_embedding = model.encode(user_query, convert_to_tensor=True).to(self.device)

            # Convert all full Text  embeddings to tensor
            tax_embeddings = [torch.tensor(e).to(self.device) for e in df['tax_text_embedding']]
            tax_embedding_tensor = torch.stack(tax_embeddings)

            # Cosine similarity on full Text
            fullText_similarities = util.cos_sim(query_embedding, tax_embedding_tensor)[0].cpu().numpy()
            df['tax_similarity'] = fullText_similarities


            embedding_df = (
                df.drop(columns=["tax_text_embedding" , "text"])
                .sort_values('tax_similarity', ascending=False)
                .head(self.top_n)
            )

            # Most similar row
            matched_row = embedding_df.loc[embedding_df["tax_similarity"].idxmax()]
            matched_row_data = matched_row.to_dict()
            print("matched row data ", matched_row_data)



            # Get company name from matched row dict
            CompanyName = str(matched_row_data.get("Company Name", "")).lower().strip()
            Year = matched_row_data.get("Year")

            # Take Empty dataframe
            filtered_df= pd.DataFrame()

           # Check if year exists in user query
            if str(Year) in user_query:
                filtered_df = original_df.loc[
                    (original_df["Company Name"].astype(str).str.lower().str.strip() == CompanyName)
                    & (original_df["Year"].astype(int) == int(Year))
                ]
            else:
                filtered_df = original_df.loc[
                    original_df["Company Name"].astype(str).str.lower().str.strip() == CompanyName
                ]

            # Handle if filetred dataframe is empty 
            if filtered_df.empty:
                return ProductResponse("failed", [])
            
            # Drop unncessary columns
            filtered_df = filtered_df.drop(columns=["text", "tax_text_embedding"]).reset_index(drop=True)

            if len(filtered_df) > 4:
                filtered_df = filtered_df.iloc[0:4]

            sorted_df = filtered_df.sort_values(by="Year" , ascending=False)

            if sorted_df.empty:
                return ProductResponse("failed", [])
            
            return ProductResponse("success", sorted_df.to_dict(orient="records"))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return Response({
                "message": error_message,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "data": [],
            })

# API For get all Tax Avenue data
class TaxAvenueView(APIView):
    def get(self, request):
        try:
            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'Tax_Avoidance.csv')

            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Read csv 
            df = pd.read_csv(input_csv_file_path)

            # Clean NaN and infinity values
            if not df.empty:
                df.dropna(inplace=True)
            
            # Return Response
            return Response({
                "message": "success" if not df.empty else "failed",
                "status": status.HTTP_200_OK if not df.empty  else 404, 
                "data": df.to_dict(orient="records") if not df.empty else []
            })
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            return Response({
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "message": error_message
            })


""" ###############################          CEO Worker Frontline Data    ##################################"""

# API FOR  CEO WORKER DATA TRAIN 
class CEOWorkerTrainPipeline(APIView):
    def get(self, format =None):
        try:
            # Vector Database dir path
            Emedding_dir_path = os.path.join(os.getcwd() ,"EmbeddingDir", "CEO-Worker")
            os.makedirs(Emedding_dir_path , exist_ok=True)

            #CSV file name
            File_path = os.path.join(os.getcwd() , "Data", 'Worker_Pay_Gap.csv')
            if not os.path.exists(File_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {File_path}")

            # Transformer model 
            TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
            os.makedirs(TransferModelDir , exist_ok=True)
            
            # Call function to train model with all rows
            CeoworkerModelResponse = CeoWorkerMainFunc(File_path, Emedding_dir_path, TransferModelDir)
            
            return Response({
                "message": CeoworkerModelResponse,
            }, status=status.HTTP_200_OK)
        

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to train Model error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            return Response({
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "message": error_message
            })

# API to inference CEO Worker  data
class CEOWorkerSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_n = 80
    similarity = 0.75

    def post(self , request , format=None):
        try:
            
            user_query = request.data.get("query")
            if not user_query:
                return BAD_RESPONSE("user query is required , Please provide with key name : 'query' ")

            split_query = user_query.split(" ")

            # Define paths
            ceo_worker_embedding_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "CEO-Worker", "ceo_worker_embedding.pkl")
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            
            # Load model
            model = SentenceTransformer(transfer_model_path)

            # Reaf full model and save mode
            df = pd.read_pickle(ceo_worker_embedding_df_path)

            # make a copy of original dataframe
            original_df = df.copy()

            # convert_user_query in embedding 
            query_embedding = model.encode(user_query, convert_to_tensor=True).to(self.device)

            # Convert all full Text  embeddings to tensor
            ceo_worker_embeddings = [torch.tensor(e).to(self.device) for e in df['frontline_text_embedding']]
            frontline_embedding_tensor = torch.stack(ceo_worker_embeddings)

            # Cosine similarity on full Text
            fullText_similarities = util.cos_sim(query_embedding, frontline_embedding_tensor)[0].cpu().numpy()
            df['ceo_worker_similarity'] = fullText_similarities

            embedding_df = (
                df.drop(columns=["frontline_text_embedding" , "text"])
                .sort_values('ceo_worker_similarity', ascending=False)
                .head(self.top_n)
            )

             # Most similar row
            matched_row = embedding_df.loc[embedding_df["ceo_worker_similarity"].idxmax()]
            matched_row_data = matched_row.to_dict()

            # Get company name from matched row dict
            CompanyName = str(matched_row_data.get("Company Name", "")).lower().strip()
            CEOName = str(matched_row_data.get("CEO Name", "")).lower().strip()
            Year = matched_row_data.get("Year")


            filtered_df = pd.DataFrame()
            # Check if year exists in user query
            if str(Year) in user_query:
                filtered_df = original_df.loc[
                    (original_df["Company Name"].astype(str).str.lower().str.strip() == CompanyName) &
                    (original_df["CEO Name"].astype(str).str.lower().str.strip() == CEOName) &
                    (original_df["Year"].astype(int) == int(Year))
                ]
            else:
                filtered_df = original_df.loc[
                    (original_df["Company Name"].astype(str).str.lower().str.strip() == CompanyName) &
                    (original_df["CEO Name"].astype(str).str.lower().str.strip() == CEOName)
                ]

            # Handle if dataframe is empty
            if filtered_df.empty:
                return ProductResponse("failed", [])
            
            # Drop unncessary columns
            filtered_df = filtered_df.drop(columns=["text", "frontline_text_embedding"]).reset_index(drop=True)

            if len(filtered_df)>4:
                filtered_df = filtered_df.iloc[0:4]

            sorted_df = filtered_df.sort_values(by="Year" , ascending=False)

            if sorted_df.empty:
                return ProductResponse("failed", [])
            
            return ProductResponse("success", sorted_df.to_dict(orient="records"))


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return []


# API For get CEO Worker data
class CeoWorkerView(APIView):
    def get(self, request):
        try:
            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'Worker_Pay_Gap.csv')

            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Read csv 
            df = pd.read_csv(input_csv_file_path)

            # Clean NaN and infinity values
            if not df.empty:
                df.dropna(inplace=True)
            
            # Return Response
            return Response({
                "message": "success" if not df.empty else "failed",
                "status": status.HTTP_200_OK if not df.empty  else 404, 
                "data": df.to_dict(orient="records") if not df.empty else []
            })
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            return Response({
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR, 
                "message": error_message
            })