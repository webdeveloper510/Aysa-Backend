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
from .utils import main_func
from .response import BAD_RESPONSE , Success_RESPONSE , DATA_NOT_FOUND

class SemanticSearchView(APIView):

    def post(self , request , format=None):
        try:
            
            user_query = request.data.get("query")

            if not user_query:
                return BAD_RESPONSE("user query is required ")
            
            # GET DIR PATH
            DirPath = os.path.join(Path.cwd(), "Data")

            result_dict = main_func(DirPath , user_query)


            if not result_dict:
                return DATA_NOT_FOUND("No any match found related your query ...")

            return Success_RESPONSE(user_query , result_dict)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return []



