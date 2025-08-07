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
from sentence_transformers import SentenceTransformer , util


# API FOR PRODUCT DATA TRAIN 
class ProductTrainPipeline(APIView):
    def get(self,format =None):
        try:
            # Vector Database dir path
            Emedding_dir_path = os.path.join(os.getcwd() ,"EmbeddingDir")
            os.makedirs(Emedding_dir_path , exist_ok=True)

            # Vector Database dir path
            ModelDirPath = os.path.join(os.getcwd() ,"Model")
            os.makedirs(ModelDirPath , exist_ok=True)

            #CSV file name
            input_csv_file_path = os.path.join(os.getcwd() , "Data", 'profit_margins.csv')
            if not os.path.exists(input_csv_file_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {input_csv_file_path}")
            
            # Transformer model 
            TransferModelDir = os.path.join(os.getcwd() ,"transfer_model")
            os.makedirs(TransferModelDir , exist_ok=True)
            
            model_response = product_main(input_csv_file_path, Emedding_dir_path, ModelDirPath, TransferModelDir)

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
    top_n = 10
    similarity = 0.40
    # function to get product search
    def ProductSearch(self ,user_query , embedding_df_path, kmeans_model_path):
        try:
            # Make a path of senetence tranfer model
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            model = SentenceTransformer(transfer_model_path)

            # Read saved dataframe
            df = pd.read_pickle(embedding_df_path)

            # Load Model
            kmeans = joblib.load(kmeans_model_path)

            # Encode user query
            query_embedding = model.encode(user_query, convert_to_tensor=True).to(self.device)

            # Convert all saved embeddings to tensor
            all_embeddings = [torch.tensor(e).to(self.device) for e in df['embedding']]
            all_embeddings_tensor = torch.stack(all_embeddings)

            # Cosine similarity
            similarities = util.cos_sim(query_embedding, all_embeddings_tensor)[0].cpu().numpy()
            df['similarity'] = similarities

            # Predict cluster for the query
            query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
            predicted_cluster = int(kmeans.predict(query_embedding_np)[0])

            # Add predicted cluster to result
            df["predicted_cluster"] = predicted_cluster

            # Sort by similarity and get top N
            results = df.sort_values('similarity', ascending=False).head(self.top_n)

            Mapped_DF = results[[
                "Brand", "Product Name", "Type", "Production Year",
                "Profit Margin", "similarity", "cluster", 'Link to Product Pictures',"Release Price","Profit Made"
            ]]

            return Mapped_DF
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return None
   
    def post(self, request, format=None):
        try:
            user_query = request.data.get("query")

            if not user_query:
                return BAD_RESPONSE("User input is required. Please provide it with the key name: 'query'")

            # Define paths
            embedding_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "product.pkl")
            kmeans_model_path = os.path.join(os.getcwd(), "Model", "product_model.pkl")

            # Load model and data
            df = self.ProductSearch(user_query, embedding_df_path, kmeans_model_path)

            if df.empty:
                return ProductResponse("failed", [], [])

            # Implement logic when someone search on single brand
            df["Brand"] = df["Brand"].str.strip().str.lower()
            number_of_brands = df["Brand"].nunique()

            # Again change in title case
            df["Brand"] = df["Brand"].str.strip().str.title()

            # if user can search only one brand 
            if number_of_brands == 1:
                # Make sure the year is numeric
                sorted_df = df.sort_values("Production Year", ascending=False)
                max_sim_row = sorted_df.loc[sorted_df['similarity'].idxmax()]
                # Convert the single row to dict inside a list
                matched_row = [max_sim_row.to_dict()]

                return ProductResponse("sucess", matched_row, [])


            # Filter by similarity if similarity score is greater than 0.45
            filtered_df = df[df["similarity"] >= self.similarity]
            
            if filtered_df.empty:

                return ProductResponse("failed", [], [])

            # make a copy of filtered df
            df = filtered_df.copy()

            # Ensure Production Year is numeric
            df["Production Year"] = pd.to_numeric(df["Production Year"], errors="coerce")
            df = df[df["Production Year"].notnull()]
            df["Production Year"] = df["Production Year"].astype(int)

            # Get matched row (highest similarity)
            matched_row = df.loc[df["similarity"].idxmax()]
            
            # Convert matched series data into dictionary
            matched_data = matched_row.to_dict()               

            # Remove unwanted keys
            matched_data.pop("similarity", None)
            matched_data.pop("cluster", None)

            # Get values for exclusion/filtering
            matched_brand = matched_data.get("Brand")
            matched_year = matched_data.get("Production Year")


            # Filter out same brand and items with higher production year
            exclude_df = df[
                     df["Brand"].str.strip().str.lower() != str(matched_brand).strip().lower()
                ]

            compare_df = pd.DataFrame()
            compare_rows = []

            if not exclude_df.empty:
                for brand , group in exclude_df.groupby("Brand"):
                    
                    group = group[group["Production Year"] <= matched_year]
                    if group.empty:
                        continue

                    years = sorted(group["Production Year"].unique(), reverse=True) 

                    selected_row = None
                    for yr in years:
                        # Filter group by current year
                        year_group = group[group["Production Year"] == yr]

                        if not year_group.empty:
                            # Select the row with the highest similarity for this year
                            selected_row = year_group.loc[year_group["similarity"].idxmax()]
                            break  # we found the most recent year available, so stop

                    if selected_row is not None:
                        compare_rows.append(selected_row)

                # Convert to DataFrame and clean up
                if compare_rows:
                    compare_df = pd.DataFrame(compare_rows)
                    compare_df = compare_df.drop(columns=["similarity", "cluster"], errors="ignore")
                    compare_df = compare_df.to_dict(orient="records")
                else:
                    compare_df = []
            
            return ProductResponse("success", matched_data, compare_df)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"

            return Response({
                "message": error_message,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "matched_data": [],
                "compare_data": []
            })

        
# API FOR Tax  DATA TRAIN 
class TaxTrainPipeline(APIView):

    def get(self, format =None):
        try:
            # Vector Database dir path
            vector_db_dir = os.path.join(os.getcwd() , "VectorDBS", "Tax_DB")
            os.makedirs(vector_db_dir , exist_ok=True)

            #CSV file name
            File_path = os.path.join(os.getcwd() , "Data", 'corporate_tax_data.csv')

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



