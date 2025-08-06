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
import json

# import Project files
from .model_pipeline import UserInference , DataTrainPipeline
from .response import BAD_RESPONSE , Success_RESPONSE , DATA_NOT_FOUND
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
            print("==========> ", results[["Brand", "Product Name", "similarity"]])

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


    def filter_highest_and_lowest_margin_rows(self,compare_df: pd.DataFrame , latest_year : str) -> pd.DataFrame:

        # Clean up the data
        compare_df = compare_df.copy()


        latest_year_df = compare_df[compare_df["Production Year"] == latest_year]

        if not latest_year_df.empty:

            latest_year_df.loc[:, "Profit Margin"] = latest_year_df["Profit Margin"].str.replace('%', '').astype(float)

            filtered_rows = []

            for brand, group in latest_year_df.groupby("Brand"):

                highest = group.loc[group["similarity"].idxmax()]
                filtered_rows.extend([highest])

            # Create the final filtered DataFrame
            filtered_df = pd.DataFrame(filtered_rows)

            # Convert columns back to original format
            filtered_df["Profit Margin"] = filtered_df["Profit Margin"].astype(str) + '%'
            
            # Optional: sort by Brand and Profit Margin
            filtered_df = filtered_df.sort_values(["Brand", "Profit Margin"], ascending=[True, False])

            # Remove unneccessary columns from the dataframe
            filtered_df = filtered_df.drop(['cluster'], axis=1)
            return filtered_df
        
        return []

    def post(self, request, format=None):
        try:
            latest_row =None
            compare_df =None

            user_query = request.data.get("query")

            if not user_query:
                return BAD_RESPONSE("user input is required, Please provide with key name: 'query'")

            # Define paths
            embedding_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "product.pkl")
            kmeans_model_path = os.path.join(os.getcwd(), "Model", "product_model.pkl")

            # Load model and data
            df  = self.ProductSearch(user_query , embedding_df_path, kmeans_model_path)

            if df.empty:
                return Response({
                    "message": "Dataframe is empty",
                    "status": status.HTTP_404_NOT_FOUND
                    })
            
            # Get total brand comes in dataframe 
            Total_brand = df["Brand"].value_counts()

            # convert into dictionary 
            brand_dict = dict(Total_brand)

            # convert values int64 to int type 
            values_list = list(map(lambda x: int(x), brand_dict.values()))

            # create new dict with updated values with same keys
            updated_data_dict = dict(zip(list(brand_dict.keys()) , values_list))

           
            # if there is only one brand return all data 
            if len(updated_data_dict) ==1:
                df = df.drop(['cluster'], axis=1)

                matched_df = df         
                latest_row = matched_df.loc[matched_df['similarity'].idxmax()]
                compare_df =pd.DataFrame()   
        
            
            # if there is multiple df get highest and lowest profit margin based on the year and brand
            elif len(updated_data_dict) >1 :

                # Get Values list of Updated Data Dict 
                Sorted_Values_list = list(updated_data_dict.values())

                # Sort list 
                Sorted_Values_list.sort()
            
                # Get largest value from list 
                max_value_count  = max(Sorted_Values_list)

                # Get matched df based on multiple rows data matched
                top_brands = [key for key, value in brand_dict.items() if value == max_value_count]

                matched_brand = top_brands[0]

                filtered_matched_df  = df[df['Brand'] == matched_brand]

                matched_df = filtered_matched_df.drop(['cluster'], axis=1)

                # Get single Row with latest year
                latest_row = matched_df.loc[matched_df['similarity'].idxmax()]

                # get latest year
                latest_year =  latest_row['Production Year']

                CompareDf =pd.DataFrame()

                if len(top_brands) > 1:
                    top_brands.remove(matched_brand)
                    CompareDf = df[df['Brand'].isin(top_brands)] 
                else:
                    CompareDf = df[~df['Brand'].isin(top_brands)] 

                # call function to get another brands highest and lower margin difference
                compare_df = self.filter_highest_and_lowest_margin_rows(CompareDf , latest_year)


            return Response({
                "message": "success",
                "status": status.HTTP_200_OK,
                "matched_data": [latest_row.to_dict()],
                "compare_data": compare_df.to_dict(orient="records") if not isinstance(compare_df , list) else []
            })
                

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"

            return Response({
                "message": error_message,
                "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "matched_data":[],
                "compare_data":[]
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



