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

from dotenv import load_dotenv
load_dotenv()


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
            return Internal_server_response(error_message)


# API to inference product trained model
class ProductSemanticSearchView(APIView):
   # Main function 
    def post(self, request, format=None):
        try:
            
            # Get threshold value from environemnt file
            threshold_value = os.getenv("THRESHOLD_VALUE")
            if isinstance(threshold_value, str):
                threshold_value = round(float(threshold_value),2)

            # Required Fields
            required_fields= ['query','tab_type']

            # Get Payload data
            payload = request.data

            # Handle missing field
            missing_fields = [field for field in required_fields if payload.get(field) is None  or not payload.get(field)]
            if missing_fields:
                return Response({
                    'message':f"{', '.join(missing_fields)}: key is required .",
                    'status':status.HTTP_400_BAD_REQUEST
                })
            
            # get payload value in parameter
            user_query = payload.get("query")

            # Define paths
            pickle_df_path = os.path.join(os.getcwd(), "EmbeddingDir", "Profit_Margin", "profit_embedding.pkl")
            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')

            # Load model
            model = SentenceTransformer(transfer_model_path)
            pickle_df = pd.read_pickle(pickle_df_path)

            # CALL A CLASS TO PREDICT PROFIT MARGIN DATA 
            Profit_Obj  = ProfitMarginPreidction(pickle_df,model,user_query)

            # Handle when user has asked about only brand name
            split_query = user_query.split()
            
            # Implement logic when user asked about only product
            if len(split_query) == 1:
                print('Single brand query is hitting .....')
                filtered_df = Profit_Obj.BrandDF(user_query , pickle_df)
                print("filtered_df : \n " , filtered_df)
                
                if not filtered_df.empty:
                    return ProductResponse('success',filtered_df.to_dict(orient="records") )

                else:
                    return DATA_NOT_FOUND(f"No Product Matched with : {user_query}")
                
            # Function -1
            Embedding_df  = Profit_Obj.apply_embedding()            # call function to get embedding df
            print("Embedding_df : \n ", Embedding_df[["Brand", "Product Name", "Product Type", "Production Year", "Gender", "Category", "Type Mapped", "similarity_score"]].iloc[0:50])
            print()
            Embedding_df = Embedding_df.loc[Embedding_df["similarity_score"] > threshold_value]    # Filter out dataframe if similarity score greater than 40
            
            if Embedding_df.empty:
                return ProductResponse("No Data Matched", [])

            # Function -2
            paramter_dict , matched_row_data_dict = Profit_Obj.GetMatchedRow_AndParameter(Embedding_df)     # Get matched row parameter dict

            # create a dataframe from matched row data dict
            searched_df = pd.DataFrame([matched_row_data_dict])

            # if searched dataframe is empty  return empty json 
            if searched_df.empty:
                return ProductResponse("failed",[])

            # Remove unneccary columns from searched dataframe
            searched_df = searched_df.drop(columns=["Gender", "text", 'similarity_score','brand_embedding', 'brand'], errors="ignore", axis=1)
            matched_row_json = searched_df.to_dict(orient="records")            # convert json into dict
            searched_product_name = matched_row_json[0]["Product Name"]
            searched_product_type = matched_row_json[0]["Product Type"]
            ProductName = searched_product_name + searched_product_type
            # call function to update product track coubnt 
            vistor_track_res = ProductSearch_Object_create_func(ProductName , payload.get("tab_type"))
            
            # Function -3
            Product_Category_df = Profit_Obj.Get_Category_based_df(paramter_dict)  

            # print("Product_Category_df : \n", Product_Category_df)
            # print()

            # Return Response if only matched row dataframe is true
            if Product_Category_df.empty:
                return ProductResponse("success",matched_row_json)

            # Function -4
            Product_Yearly_df = Profit_Obj.Get_year_based_df(paramter_dict , Product_Category_df) 
            #print("Product_Yearly_df : \n ", Product_Yearly_df[["Brand", "Product Name", "Product Type", "Production Year", "Gender", "Category", "Type Mapped"]])
            #print()

            # Return Response if only matched row dataframe is true
            if Product_Yearly_df.empty:
                return ProductResponse("success",matched_row_json)

            # Function -5
            Product_Gender_df = Profit_Obj.Get_gender_based_df(paramter_dict , Product_Yearly_df) 
            #print("Product_Gender_df : \n ", Product_Gender_df[["Brand", "Product Name", "Product Type", "Production Year", "Gender", "Category", "Type Mapped"]])

            if Product_Gender_df.empty:
                return ProductResponse("success",matched_row_json)

            # Function -6
            brand_product_type_list= Profit_Obj.Filter_rows_list(paramter_dict , Product_Gender_df) 

            # Function -7 
            filtered_df = Profit_Obj.Filtered_Dataframe(brand_product_type_list)
    
            # Handle if filtered datframe return empty list
            if isinstance(filtered_df , list):
                return ProductResponse('success', matched_row_json)
            
            # Add percentage sign
            filtered_df["Profit Margin"] = filtered_df["Profit Margin"].astype(float).map(lambda x: f"{x:.2f} %")
            

            # Drop Unneccessary columns if it filtered_df is dataframe
            if isinstance(filtered_df , pd.DataFrame) and not filtered_df.empty:
                filtered_df = filtered_df.drop(columns=["text", "Gender","similarity_score", "text_embedding", "brand_embedding", "brand"],  errors="ignore")      # remove unneccessary dataframe

            # Merge bot dataframe
            merge_df = pd.concat([searched_df , filtered_df], ignore_index=True)    # concat both dataframe  

            # Only return three product in API
            if len(merge_df) > 3:
                merge_df = merge_df.iloc[0:3]

            return ProductResponse('success', merge_df.to_dict(orient="records"))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occurred: {str(e)} (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)


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

            # Return Response
            return Response({
                "message": "success" if not df.empty else "failed",
                "status": status.HTTP_200_OK if not df.empty  else 404, 
                "data": df.to_dict(orient="records") if not df.empty else []
            })
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)


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
            print(error_message)
            return Internal_server_response(error_message)
          

# API to inference Tax trained model
class TaxSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_n = 80
    similarity = 0.75

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
            user_query = request.data.get("query")

            # Call function to get year status from  the user query ...
            Filter_year_from_user_query= get_year(str(user_query))
            YEAR_STATUS = True if  Filter_year_from_user_query != 'None' else False

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
            Year = matched_row_data.get("Year")

            # Take Empty dataframe
            filtered_df= pd.DataFrame()

            
           # Check if year exists in user query
            if YEAR_STATUS:
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

            first_row = sorted_df.iloc[0]
            first_row_dict = first_row.to_dict()


            soreted_filetered_product_name = first_row_dict.get("Company Name") + str(first_row_dict.get("Year"))
                       
            PRODUCT_NAME = user_query if  YEAR_STATUS else soreted_filetered_product_name

            # Call function to update track count of Tax data:
            ProductSearch_Object_create_func(PRODUCT_NAME , payload.get("tab_type"))

            if sorted_df.empty:
                return ProductResponse("failed", [])
            
            return ProductResponse("success", sorted_df.to_dict(orient="records"))

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            return Internal_server_response(error_message)

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
            print(error_message)
            return Internal_server_response(error_message)

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
                "status": status.HTTP_200_OK,
                "message": CeoworkerModelResponse,
            })

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to train Model error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)

# API to inference CEO Worker  data
class CEOWorkerSemanticSearchView(APIView):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_n = 80
    similarity = 0.75

    def post(self , request , format=None):
        try:
            
            # GET USER QUERY FROM POST REQUEST 
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
            user_query = request.data.get("query")

            # Call function to get year status from  the user query ...
            Filter_year_from_user_query= get_year(user_query)
            YEAR_STATUS = True if  Filter_year_from_user_query != 'None' else False

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
            if YEAR_STATUS:
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

            first_row = sorted_df.iloc[0]
            first_row_dict = first_row.to_dict()


            soreted_filetered_product_name = first_row_dict.get("Company Name") +" " + str(first_row_dict.get("Year"))
                       
            PRODUCT_NAME = user_query if  YEAR_STATUS else soreted_filetered_product_name

            # Call function to update track count of Tax data:
            ProductSearch_Object_create_func(PRODUCT_NAME , payload.get("tab_type"))

            if sorted_df.empty:
                return ProductResponse("failed", [])
            
            return ProductResponse("success", sorted_df.to_dict(orient="records"))


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Occur Reason: {str(e)} (line {exc_tb.tb_lineno})"
            return Internal_server_response(error_message)


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
class GetProductVisitorCount(APIView):
    def get(self,request):
        try:

            date_str = request.GET.get("date")

            Product_Data_obj = ProductSearchTrack.objects.all().values()

            if not Product_Data_obj:
                return DATA_NOT_FOUND("No data found .")

            df = pd.DataFrame(list(Product_Data_obj))

            # Convert to datetime
            df["created_at"] = pd.to_datetime(df["created_at"])

            # Extract only date
            df["date_only"] = df["created_at"].dt.date

            df = df.loc[df["date_only"].astype(str) == str(date_str)]
            

            if df.empty:

                return DATA_NOT_FOUND(f"No Data exist for Date : {date_str}")
            
            return Response({
                "message": f"No Data Found for Date : {date_str}" if df.empty else f"Data get successfully of date : {date_str}" ,
                "status": status.HTTP_200_OK,
                "data": df.to_dict(orient="records")
            })
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"Failed to get profit margin data,  error occur: {str(e)} in (line {exc_tb.tb_lineno})"
            print(error_message)
            return Internal_server_response(error_message)
        