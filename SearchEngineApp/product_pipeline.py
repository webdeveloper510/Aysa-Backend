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
from .utils import *



# Product data train Pipeline
class DataTrainPipeline:
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
        print("Step 3: Embedding chunks and creating vector store...")
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
            error_message = f"[ERROR] Embedding failed. Reason: {str(e)} (line {exc_tb.tb_lineno})"
            return error_message


# Product data inference 
class UserInference:

    def __init__(self , model_path , required_keys):
        self.model_path = model_path
        self.required_keys  = required_keys
        
    # function to load vector DB
    def LoadVectorDB(self):
        print("Step 1: Starting to load local saved database ....")
        embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")
        vector_db = FAISS.load_local(self.model_path,embedding,allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever()
        return retriever 

    
    def ModelInference(self,retriever, user_query):
        print("Step 2: Running model inference...")
        try:
            final_data_list =[]

            results_with_scores = retriever.vectorstore.similarity_search_with_score(user_query)

            sorted_results = sorted(results_with_scores, key=lambda x: x[1])

            for doc, score in sorted_results:
                for line in doc:

                    if isinstance(line, tuple) and line[0] == "page_content" and 0.3 < float(score) <= 1.0:
                        content_list = line[1].split("\n")

                        # Skip if the content list is empty or doesn't contain key-value format
                        if not content_list or ":" not in content_list[0]:
                            print("Invalid content format, skipping this line.")
                            continue

                        try:
                            data_dict = ListToDict(content_list)

                            final_data_list.append(data_dict)
                        except Exception as e:
                            print(f"ListToDict failed: {e}, skipping line.")
                            continue
                    else:
                        print(f"Skipping Line because it has no needed info : {line}", float(score))

            filtered_result = [item for item in final_data_list if is_valid_product(item, self.required_keys)]
            if filtered_result:
                return filtered_result

            return []

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            error_message = f"[ERROR] Model inference failed. Reason: {str(e)} (line {exc_tb.tb_lineno})"
            return error_message





