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

# import Project files
from .product_pipeline import ProductDataTrainPipeline , inference
from .response import BAD_RESPONSE , Success_RESPONSE , DATA_NOT_FOUND

class ProductTrainPipeline(APIView):
    def get(self, request, format =None):
        try:
            # Vector Database dir path
            vector_db_dir = os.path.join(os.getcwd() , "VectorDBS", "Product_DB")
            os.makedirs(vector_db_dir , exist_ok=True)

            #CSV file name
            File_path = os.path.join(os.getcwd() , "Data", 'profit_margins.csv')

            if not os.path.exists(File_path):
                return DATA_NOT_FOUND(f"File Not Found with Name : {File_path}")

            # function to train model
            class_obj = ProductDataTrainPipeline(File_path)

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


from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# API to inference product trained model
class ProductSemanticSearchView(APIView):

    def post(self , request , format=None):
        try:
            
            user_query = request.data.get("query")
            if not user_query:
                return BAD_RESPONSE("user query is required ")
            
            VectoDB_Path = os.path.join(os.getcwd() , "VectorDBS", "Product_DB", "faiss_index")

            # Call Product Inference Model
            result_dict =inference(VectoDB_Path, user_query)

            return Success_RESPONSE(user_query , result_dict)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return []



