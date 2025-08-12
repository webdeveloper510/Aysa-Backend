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
from .model_pipeline import UserInference , DataTrainPipeline
from .response import BAD_RESPONSE , Success_RESPONSE , DATA_NOT_FOUND, ProductResponse
from .models import *
from .utils import *
from .product_structure import *
from .tax_paid import *
from sentence_transformers import SentenceTransformer , util


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
                df.drop(columns=['Type_encoded', "full_text_embedding" , "brand_embedding"])
                .sort_values('full_text_similarity', ascending=False)
                .head(self.top_n)
            )

            return embedding_df , original_df
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed tp get full and sub embed dataframe [ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return None
   
    def post(self, request, format=None):
        try:

            user_query = request.data.get("query")
            if not user_query:
                return BAD_RESPONSE("User input is required. Please provide it with the key name: 'query'")

            # Split user query
            split_query = user_query.split(" ")

            # Define paths of all rows embedding model
            full_embedding_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "Profit_Margin", "profit_embedding.pkl")

            # Make a path of senetence tranfer model
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            model = SentenceTransformer(transfer_model_path)

            # call function to get highest similar data
            embedding_df , original_df = self.ProductSearch(user_query, full_embedding_df_path, model)
            
            # Hamdle
            if embedding_df.empty:
                return ProductResponse("failed", [])

            # Get most highest similar row
            matched_row = embedding_df.loc[embedding_df["full_text_similarity"].idxmax()]

            # convert into dictionary
            matched_row_data = matched_row.to_dict()

            #Get variables
            BrandName = matched_row_data.get("Brand")
            Product_type=matched_row_data.get("Type")
            product_category = matched_row_data.get("Category")
            matched_year = matched_row_data.get("Production Year")

            #Filtere dataframe based on the query
            filtered_df = original_df.loc[
                (original_df["Brand"] != BrandName) & 
                (original_df["Type"].str.contains(Product_type, case=False, na=False))&
                (original_df["Production Year"] ==matched_year) & 
                (original_df["Category"]==product_category)]
            

            filtered_df= filtered_df.drop(columns=['Type_encoded', "full_text_embedding" , "brand_embedding"])
            if not filtered_df.empty:
                filtered_df = filtered_df.sort_values('Profit Margin', ascending=False)


            # convert Series columbn to dataframe
            matched_df = pd.DataFrame([matched_row])

            # Merge dataframe 
            merge_df = pd.concat([matched_df ,filtered_df]).reset_index(drop=True)

            # Drop unnecessary columns
            merge_df = merge_df.drop(columns=["brand_similarity","brand" , "text" , "full_text_similarity", "text"])

            if len(merge_df)>3:
                merge_df = merge_df.iloc[0:3]

            json_output = merge_df.to_dict(orient="records")

            return ProductResponse("success", json_output)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)

            return Response({
                "message": error_message,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "matched_data": [],
                "compare_data": []
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

# API FOR Tax  DATA TRAIN 
class TaxDataTrainPipeline(APIView):
    def get(self,format =None):
        try:
            # Vector Database dir path
            Emedding_dir_path = os.path.join(os.getcwd() ,"EmbeddingDir", "Tax")
            os.makedirs(Emedding_dir_path , exist_ok=True)

            # Vector Database dir path
            ModelDirPath = os.path.join(os.getcwd() ,"Model", "Tax")
            os.makedirs(ModelDirPath , exist_ok=True)

            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'Tax_Avoidance.csv')
            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Transformer model 
            TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
            os.makedirs(TransferModelDir , exist_ok=True)
            
            # Call function to train model with all rows
            TaxModelResponse = Tax_main(input_csv_file_path, Emedding_dir_path, ModelDirPath, TransferModelDir)
            
            # train model only brand ,Product type data 
            # model_response = BrandProductMain(input_csv_file_path, Emedding_dir_path, ModelDirPath, TransferModelDir)

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

    def post(self , request , format=None):
        try:
            
            user_input = request.data.get("query")
            if not user_input:
                return BAD_RESPONSE("user input is required , Please provide with key name : 'query'")
            
            user_query =SpellCorrector(user_input)
            
            VectoDB_Path = os.path.join(os.getcwd() , "VectorDBS", "Tax_DB", "faiss_index")

            # Call Product Inference Model
            # Function to inference model
            inference_obj = UserInference(VectoDB_Path , TAX_DATA_KEYS)

            # STEP 1 
            retriever = inference_obj.LoadVectorDB()

            result_dict = inference_obj.ModelInference(retriever, user_query)

            return Success_RESPONSE(user_query , result_dict)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return []

# API FOR  CEO WORKER DATA TRAIN 
class CEOWorkerTrainPipeline(APIView):
    def get(self, format =None):
        try:
            # Vector Database dir path
            vector_db_dir = os.path.join(os.getcwd() , "VectorDBS", "Ceo_Worker")
            os.makedirs(vector_db_dir , exist_ok=True)

            #CSV file name
            File_path = os.path.join(os.getcwd() , "Data", 'ceo_vs_worker_pay.csv')

            if not os.path.exists(File_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {File_path}")

            # function to train model
            class_obj = DataTrainPipeline(File_path)

            # STEP 1 : DATA INGESTION
            documents = class_obj.DataIngestion(File_path)
            if not documents:
                return BAD_RESPONSE("Failed to ingest data")

            # STEP 2: DATA CHUNKING
            chunks = class_obj.DataChunking(documents)
            if not chunks:
                return BAD_RESPONSE("Failed to chunk data.")
            

            # STEP 3: VECTORIZATION AND SAVE AT LOCAL
            result_message = class_obj.TextEmbeddingAndVectorDb(vector_db_dir,chunks)
            if result_message is None:
                return BAD_RESPONSE("Failed to Vectorization data")


            return Response({
                "message": result_message,
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

    def post(self , request , format=None):
        try:
            
            user_input = request.data.get("query")
            if not user_input:
                return BAD_RESPONSE("user query is required , Please provide with key name : 'query' ")
            

            user_query =SpellCorrector(user_input)
            
            VectoDB_Path = os.path.join(os.getcwd() , "VectorDBS", "Ceo_Worker", "faiss_index")

            # Function to inference model
            inference_obj = UserInference(VectoDB_Path, CEO_WORKER_DATA_KEYS)

            retriever = inference_obj.LoadVectorDB()

            result_dict = inference_obj.ModelInference(retriever, user_query)

            return Success_RESPONSE(user_query , result_dict)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return []



