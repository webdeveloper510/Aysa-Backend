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
from .product_pipeline import product_data_train_pipeline , inference
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


            # call main fuinction 
            response_data = product_data_train_pipeline(vector_db_dir, File_path)

            http_status = status.HTTP_200_OK if response_data["status"] == "success" else status.HTTP_500_INTERNAL_SERVER_ERROR
            return Response(response_data, status=http_status)


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



