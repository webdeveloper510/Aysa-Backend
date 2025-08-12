import pandas as pd
import numpy as np
import sys
import os
import torch
import joblib
from sentence_transformers import SentenceTransformer , util
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from .utils import preprocess_text
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def GetBestK_score(max_range: int, embeddings) -> int:
    best_score = -1
    best_k = 2

    for k in range(2, max_range):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        print(f"k={k}, silhouette_score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


class TAXModelStructure:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.max_range = 20
        self.random_state_value = 42
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def DownloadUpdateModel(self , TransferModelDir):
        import os
        os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'huggingface_cache')

        model_path = os.path.join(TransferModelDir, "all-MiniLM-L6-v2")

        if not os.path.exists(model_path):
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            model.save(model_path)

        else:
            model = SentenceTransformer(model_path)

        return model


    def read_csv(self):
        print("Step 1: Reading CSV file...")
        try:
            df = pd.read_csv(self.csv_path)
            print("columns name ", df.columns.tolist())
            return df
        except Exception as e:
            raise Exception(f"Failed to read CSV: {e}")

    def preprocess_text_data(self, df):
        print("Step 2: Preprocessing text...")
        ['Company Name', 'Year', 'Taxes Paid', 'Taxes Avoided']

        try:
            df = df.dropna().drop_duplicates()
            df.columns = df.columns.str.strip()
            df = df.copy()

            df["Production Year"] = df["Production Year"].fillna("").apply(
                lambda x: str(int(x)) if x != "" else ""
            )

            df["Text"] = (
                #df["Brand"].astype(str) + " " +
                df["Product Name"].astype(str) + " " +
                df["Type"].astype(str) + " " +
                df["Production Year"].astype(str) + " " +
                df["Profit Margin"].astype(str)
            )

            df["Text"] = df["Text"].apply(preprocess_text)
            return df
        except Exception as e:
            raise Exception(f"Text preprocessing failed: {e}")

    def encode_text(self, df):
        print("Step 3: Generating embeddings...")
        try:

            transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
            model = SentenceTransformer(transfer_model_path)

            embeddings = model.encode(df['Text'].tolist(), show_progress_bar=True)

            if len(embeddings) == 0:
                raise ValueError("No embeddings generated. Check input data.")
            
            df["embedding"] = embeddings.tolist()

            return df, embeddings
        except Exception as e:
            raise Exception(f"Text embedding failed: {e}")

    def apply_kmeans(self, df, embeddings, best_k ,  embedding_dir_path ,ModelDirPath):
        print(f"Step 4: Fitting KMeans with k={best_k}...")
        try:
            kmeans = KMeans(n_clusters=best_k, random_state=self.random_state_value)
            df["cluster"] = kmeans.fit_predict(embeddings)

            # save embedding df
            embedding_path = os.path.join(embedding_dir_path,"product.pkl")

            df.to_pickle(embedding_path)

            emebdding_response = f"Full Embedding DataFrame saved to {embedding_path}"
            print(emebdding_response)
            
            # Save Kmeans model
            model_save_path = os.path.join(ModelDirPath,"product_model.pkl")

            joblib.dump(kmeans, model_save_path)

            model_response = f"Kmeans model saved to {model_save_path}"
            print(model_response)
            
            return "success"
        
        except Exception as e:
            raise Exception(f"KMeans clustering failed: {e}")

# class BrandModelPredictorSemantic:

#     def __init__(self, csv_path):
#         self.csv_path = csv_path
#         self.max_range = 20
#         self.random_state_value = 42
#         self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def DownloadUpdateModel(self , TransferModelDir):
#         import os
#         os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'huggingface_cache')

#         model_path = os.path.join(TransferModelDir, "all-MiniLM-L6-v2")

#         if not os.path.exists(model_path):
#             model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#             model.save(model_path)

#         else:
#             print("model path ", model_path)
#             model = SentenceTransformer(model_path)

#         return model


#     def read_csv(self):
#         print("Step 1: Reading CSV file...")
#         try:
#             df = pd.read_csv(self.csv_path)
#             rename_columns = {"Production Year ": "Production Year", "Profit Margin": "Profit Margin"}
#             df.rename(rename_columns, inplace=True)
#             return df
#         except Exception as e:
#             raise Exception(f"Failed to read CSV: {e}")

#     def preprocess_text_data(self, df):
#         print("Step 2: Preprocessing text...")
#         try:
#             df = df.dropna().drop_duplicates()
#             df.columns = df.columns.str.strip()
#             df = df.copy()

#             df["Production Year"] = df["Production Year"].fillna("").apply(
#                 lambda x: str(int(x)) if x != "" else ""
#             )

#             df["Text"] = (
#                 #df["Brand"].astype(str) + " " +
#                 df["Product Name"].astype(str) + " " +
#                 df["Type"].astype(str) + " "
     
#             )

#             df["Text"] = df["Text"].apply(preprocess_text)
#             return df
#         except Exception as e:
#             raise Exception(f"Text preprocessing failed: {e}")

#     def encode_text(self, df):
#         print("Step 3: Generating embeddings...")
#         try:

#             transfer_model_path = os.path.join(os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2')
#             model = SentenceTransformer(transfer_model_path)

#             embeddings = model.encode(df['Text'].tolist(), show_progress_bar=True)

#             if len(embeddings) == 0:
#                 raise ValueError("No embeddings generated. Check input data.")
            
#             df["embedding"] = embeddings.tolist()

#             return df, embeddings
#         except Exception as e:
#             raise Exception(f"Text embedding failed: {e}")

#     def apply_kmeans(self, df, embeddings, best_k ,  embedding_dir_path ,ModelDirPath):
#         print(f"Step 4: Fitting KMeans with k={best_k}...")
#         try:
#             kmeans = KMeans(n_clusters=best_k, random_state=self.random_state_value)
#             df["cluster"] = kmeans.fit_predict(embeddings)

#             # save embedding df
#             embedding_path = os.path.join(embedding_dir_path,"brand_product.pkl")

#             df.to_pickle(embedding_path)

#             emebdding_response = f"Full Embedding DataFrame saved to {embedding_path}"
#             print(emebdding_response)
            
#             # Save Kmeans model
#             model_save_path = os.path.join(ModelDirPath,"brand_product_model.pkl")

#             joblib.dump(kmeans, model_save_path)

#             model_response = f"Kmeans model saved to {model_save_path}"
#             print(model_response)
            
#             return "success"
        
#         except Exception as e:
#             raise Exception(f"KMeans clustering failed: {e}")



def Tax_main(file_path, embedding_dir_path ,ModelDirPath , TransferModelDir):
    try:
        product_structure = TAXModelStructure(file_path)
        model = product_structure.DownloadUpdateModel(TransferModelDir)
        df = product_structure.read_csv()
        # cleaned_df = product_structure.preprocess_text_data(df)
        # encoded_df, embeddings = product_structure.encode_text(cleaned_df)
        # best_k = GetBestK_score(product_structure.max_range, embeddings)
        # response = product_structure.apply_kmeans(encoded_df, embeddings, best_k, embedding_dir_path, ModelDirPath)

        # return response     

    except Exception as e:
        error_message = f" Failed to process product model structure: {e}"
        print(error_message)
        return None








