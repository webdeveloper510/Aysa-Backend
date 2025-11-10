from .response import *
import pandas as pd
import numpy as np
import redis
import os 
import sys
import tracemalloc
from SearchMind.settings import REDIS_HOST , REDIS_PORT
import json

from dotenv import load_dotenv
load_dotenv()


# Create Redis Instance
redis_instance = redis.StrictRedis(REDIS_HOST, REDIS_PORT , 1)   # 1 Eepresent to DB Name
cache_expire_time = os.getenv("CACHE_EXPIRE_TIME")

# function to get profit margin data 
def get_product_data_use_redis():
    try:
        
        message = "Data Come From Database"
        redis_key = os.getenv("PRODUCT_DATA_REDIS_KEY_NAME")            # Get key Name from the env file
        
        # Get cache Data
        cached_data = redis_instance.get(redis_key) 
        if cached_data is not None:
            message = "Data Come From Cache"
            return json.loads(cached_data) , message

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

        json_data = df.to_dict(orient="records") if not df.empty else []

        redis_instance.set(redis_key, json.dumps(json_data) ,ex=cache_expire_time)

        print("Product Data Saved in Redis cache with name {}".format(redis_key))
        return json_data , message
    

    except Exception as e:
        exc_type , exc_obj , exc_tb = sys.exc_info()
        error_message =f"[ERROR] Failed to get redis cache product data error occur , error is : {str(e)} in line no : {exc_tb.tb_lineno}"
        return error_message
    

# function to get  tax data 
def get_tax_data_use_redis():
    try:

        redis_key = os.getenv("TAX_DATA_REDIS_KEY_NAME")            # Get key Name from the env file
        
        message = "Data Come From Database"
        # Get cache Data
        cached_data = redis_instance.get(redis_key) 
        if cached_data is not None:
            message= "Data Come From Cache"
            return json.loads(cached_data) , message

        # Get Data from Tax model
        tax_csv_file_path = os.path.join(os.getcwd(), "static", "media" , "Tax Data", 'Tax_Avoidance.csv')

        # Read CSV
        df = pd.read_csv(tax_csv_file_path)

        # Clean NaN and infinity values
        if not df.empty:
            df.dropna(inplace=True)
        
        json_data = df.to_dict(orient="records") if not df.empty else []

        redis_instance.set(redis_key, json.dumps(json_data) ,ex=cache_expire_time)
        print("Tax Data Saved in Redis cache with name {}".format(redis_key))
        return json_data , message
    

    except Exception as e:
        exc_type , exc_obj , exc_tb = sys.exc_info()
        error_message =f"[ERROR] Failed to get redis cache Tax data error occur , error is : {str(e)} in line no : {exc_tb.tb_lineno}"
        return error_message
    

# function to get CEO WORKER data 
def get_ceo_worker_data_use_redis():
    try:

        redis_key = os.getenv("PAYGAP_DATA_REDIS_KEY_NAME")            # Get key Name from the env file
        
        message = "Data Come From Database"

        # Get cache Data
        cached_data = redis_instance.get(redis_key) 
        if cached_data is not None:
            message = "Data Come From Cache"
            return json.loads(cached_data) , message

    
        # Get Data from Tax model
        ceo_worker_file_path = os.path.join(os.getcwd(), "static", "media" ,"CEO Worker Data", "Website.csv")

        # Read CSV
        df = pd.read_csv(ceo_worker_file_path)

        # Clean NaN and infinity values
        if not df.empty:
            df.dropna(inplace=True)
        
        json_data = df.to_dict(orient="records") if not df.empty else []

        redis_instance.set(redis_key, json.dumps(json_data) ,ex=cache_expire_time)
        print("CEO WORKER Data Saved in Redis cache with name {}".format(redis_key))
        return json_data , message
    

    except Exception as e:
        exc_type , exc_obj , exc_tb = sys.exc_info()
        error_message =f"[ERROR] Failed to get redis cache CEO WORKER data error occur , error is : {str(e)} in line no : {exc_tb.tb_lineno}"
        return error_message
    


from .redis_helper import redis_instance


# List of keys to delete
def delete_cache_data(redis_key_list: list) -> list:
    deleted_keys = []
    for key in redis_key_list:
        if redis_instance.exists(key):
            redis_instance.delete(key)
            deleted_keys.append(key)
            print(f"Deleted Redis key: {key}")
        else:
            print(f"Key not found in Redis: {key}")

    return deleted_keys