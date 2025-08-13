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
        drop_cols = ["brand_similarity", "brand", "text", "full_text_similarity", "brand_embedding"]
        return df.drop(columns=drop_cols, errors="ignore")

    # function to change profit margin in dataframe
    def convert_profit_margin(self,df):
        df["Profit Margin"] = (
            df["Profit Margin"].astype(str)
            .str.replace("%", "")
            .str.replace(",", ".")
            .str.replace("$", ".")
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

    # Function to compare rows 
    def filter_compare_rows(self,df, matched):
        compare_rows = []
        for row_dict in df.to_dict(orient="records"):
            Brand = str(row_dict.get("Brand", "")).lower().strip()
            Year = int(row_dict.get("Production Year"))
            ProductCategory = str(row_dict.get("Category", "")).lower().strip()
            ProductType = str(row_dict.get("Type", "")).lower().strip()

            if (matched["Product_type"] in ProductType
                    and matched["BrandName"] != Brand
                    and ProductCategory == matched["product_category"]
                    and Year == matched["matched_year"]):
                compare_rows.append(row_dict)
        return pd.DataFrame(compare_rows)

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

            embedding_df = (
                df.drop(columns=['Type_encoded', "full_text_embedding"])
                .sort_values('full_text_similarity', ascending=False)
                .head(self.top_n)
            )

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

            if embedding_df.empty:
                return ProductResponse("failed", [])

            # Most similar row
            matched_row = embedding_df.loc[embedding_df["full_text_similarity"].idxmax()]
            matched_row_data = matched_row.to_dict()

            BrandName = str(matched_row_data.get("Brand", "")).lower().strip()
            Product_type = str(matched_row_data.get("Type", "")).lower().strip()
            product_category = str(matched_row_data.get("Category", "")).lower().strip()
            matched_year = int(matched_row_data.get("Production Year"))

            print({
                "BrandName": BrandName,
                "Product_type": Product_type,
                "product_category": product_category,
                "matched_year": matched_year,
            })

            # Handle if user ask only about brand

            if len(split_query) == 1:
                print("Only ask about single brand")
                filtered_df = embedding_df.loc[embedding_df["brand_similarity"] > 0.65]


                # Check if user input exists in Brand column
                if not filtered_df.empty:
                    
                    # convert profit margin into string
                    filtered_df = self.convert_profit_margin(filtered_df)

                    filtered_df = filtered_df.sort_values(
                        ['Production Year', 'Profit Margin'], ascending=[False, False]
                    )
                    filtered_df = filtered_df.drop(columns =["brand_embedding"], axis=1)

                    print(filtered_df[["Brand", "Product Name", "Type", "Production Year", 'Profit Margin']])
                    filtered_df = filtered_df.drop_duplicates(subset=["Production Year"], keep="first")

                    filtered_df = self.drop_unnecessary_cols(filtered_df)

                    # Show latest + 4 previous years only
                    if len(filtered_df) > 5:
                        filtered_df = filtered_df.iloc[0:5]

                    return ProductResponse("success", filtered_df.to_dict(orient="records"))
                
            print("Hit for compare products ....")
            # Drop unnecessary columns from original_df
            original_df = original_df.drop(
                columns=['Type_encoded', "full_text_embedding", "brand_embedding"],
                errors='ignore'
            )

            filtered_df = self.filter_compare_rows(original_df, {
                "Product_type": Product_type,
                "BrandName": BrandName,
                "product_category": product_category,
                "matched_year": matched_year
            })


            if not filtered_df.empty:
                filtered_df = filtered_df.sort_values('Profit Margin', ascending=False)
                filtered_df = self.get_highest_margin_per_brand(filtered_df)

            matched_df = pd.DataFrame([matched_row])

            merge_df = pd.concat([matched_df, filtered_df]).reset_index(drop=True)
            merge_df = self.drop_unnecessary_cols(merge_df)

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
class GetProfitMarginData(APIView):
    def get(self,request,format=None):
        try:
            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'profit_margins.csv')
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

            # Get company name from matched row dict
            CompanyName = str(matched_row_data.get("Company Name", "")).lower().strip()
            
            filtered_df = original_df.loc[original_df["Company Name"].astype(str).str.lower().str.strip() ==CompanyName]

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
            
            filtered_df = original_df.loc[
                (original_df["Company Name"].astype(str).str.lower().str.strip() ==CompanyName)&
                (original_df["CEO Name"].astype(str).str.lower().str.strip() ==CEOName)
                ]

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