import os
import sys
from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from .response import DATA_NOT_FOUND
from .tests import *

# function to convert data into list 
def ListToDict(input_list):
    data_dict ={}
    for item in input_list :
        split_item=  item.split(":")
        data_dict[split_item[0]] =split_item[1]
    return data_dict

# Product data train Pipeline
class ProductDataTrainPipeline:
    def __init__(self, ProductCSV):
        self.fileName = ProductCSV

    @staticmethod
    def DataIngestion(FILEPATH: list) -> list:
        print("Step 1: Starting data ingestion...")
        try:
            loader = CSVLoader(file_path=FILEPATH)
            docs = loader.load()
            return docs

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Data ingestion failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []

    @staticmethod
    def DataChunking(documents: list) -> list:
        print("Step 2: Chunking data...")
        try:
            # Use record-level chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
            chunks = splitter.split_documents(documents)
            print(f"Generated {len(chunks)} chunks.")
            return chunks

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Chunking failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []

    @staticmethod
    def TextEmbeddingAndVectorDb(VECTORDB_DIR_PATH, chunks: list):
        print("Step 4: Embedding chunks and creating vector store...")
        try:
            embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")
            vectorstoreDB = FAISS.from_documents(chunks, embedding)

            # Save the FAISS index locally
            db_path = os.path.join(VECTORDB_DIR_PATH,"faiss_index")

            vectorstoreDB.save_local(db_path)
            print(f"[INFO] FAISS index saved to '{db_path}'")

            return f"Model Trained and update successfully ..."

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Embedding failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return None


# Product data inference 
class InferenceProduct:
    def __init__(self , model_path):
        self.model_path = model_path
        
    # function to load vector DB
    def LoadVectorDB(self):
        embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")
        vector_db = FAISS.load_local(self.model_path,embedding,allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever()
        return retriever 

    @staticmethod
    def ModelInference(retriever, user_query):
        print("Step 5: Running model inference...")
        try:
            final_data_list =[]

            results_with_scores = retriever.vectorstore.similarity_search_with_score(user_query)
            sorted_results = sorted(results_with_scores, key=lambda x: x[1])

            for doc, score in sorted_results:
                for line in doc:
                    
                    if (isinstance(line, tuple) and line[0] == "page_content" and 0.3 < float(score) <= 0.9):
                        content_list = line[1].split("\n")
                        print("score ", score  ,"content_list  ", content_list)
                        data_dict = ListToDict(content_list)
                        final_data_list.append(data_dict)
                    else:
                        print(f"Skipping Line because it has no needed info : {line}",float(score))

            return final_data_list

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Model inference failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []



# function to train model
def product_data_train_pipeline(VECTORDB_DIR_PATH , FilePath):
    class_obj = ProductDataTrainPipeline(FilePath)

    # STEP 1
    documents = class_obj.DataIngestion(FilePath)
    if not documents:
        return DATA_NOT_FOUND("Failed to ingest data.")

    # STEP 2
    chunks = class_obj.DataChunking(documents)
    if not chunks:
        return DATA_NOT_FOUND("Failed to chunk data.")

    # STEP 3
    Response = class_obj.TextEmbeddingAndVectorDb(VECTORDB_DIR_PATH,chunks)
    if Response is None:
        return DATA_NOT_FOUND("Failed Train DATA.")
    return Response



# Function to inference model
def inference(local_db_path , user_query):
    inference_obj = InferenceProduct(local_db_path)

    # STEP 1 
    retriever = inference_obj.LoadVectorDB()

    response = inference_obj.ModelInference(retriever, user_query)
    return response


