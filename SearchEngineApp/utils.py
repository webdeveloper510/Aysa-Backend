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


class ReadDataDir:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def ReadDir(self):
        print(f"Step 1: Reading CSV files from directory: {self.dir_path}")
        try:
            if not os.path.exists(self.dir_path):
                print(f"[ERROR] Directory does not exist: {self.dir_path}")
                return []

            dataFilesList = []
            for root, dirs, files in os.walk(self.dir_path):
                csv_files = [os.path.join(root, file) for file in files if file.endswith(".csv")]
                dataFilesList.extend(csv_files)

            if not dataFilesList:
                print("No CSV files found.")
            else:
                print(f"Found {len(dataFilesList)} CSV files.")

            return dataFilesList

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Failed to read CSV files. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []

    @staticmethod
    def DataIngestion(fileList: list) -> list:
        print("Step 2: Starting data ingestion...")
        try:
            documents = []
            for file in fileList:
                loader = CSVLoader(file_path=file)
                docs = loader.load()
                documents.extend(docs)
            return documents

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Data ingestion failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []

    @staticmethod
    def DataChunking(documents: list) -> list:
        print("Step 3: Chunking data...")
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
    def TextEmbeddingAndVectorDb(chunks: list):
        print("Step 4: Embedding chunks and creating vector store...")
        try:
            embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")
            vectorstoreDB = FAISS.from_documents(chunks, embedding)
            retriever = vectorstoreDB.as_retriever()
            return retriever

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Embedding failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []

    @staticmethod
    def ModelInference(retriever, user_query):
        print("Step 5: Running model inference...")
        try:
            final_data_list =[]

            results_with_scores = retriever.vectorstore.similarity_search_with_score(user_query)

            for doc, score in results_with_scores:
                for line in doc:

                    if isinstance(line , tuple) and line[0] =="page_content":
                        content_list = line[1].split("\n")
                        data_dict = ListToDict(content_list)
                        final_data_list.append(data_dict)
                    else:
                        print(f"Skipping Line because it has no needed info : {line}")

            return final_data_list

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f"[ERROR] Model inference failed. Reason: {str(e)} (line {exc_tb.tb_lineno})")
            return []




def main_func(dir_path, user_query):
    class_obj = ReadDataDir(dir_path)

    file_list = class_obj.ReadDir()
    if not file_list:
        return DATA_NOT_FOUND(f"No CSV files found in directory: {dir_path}")

    documents = class_obj.DataIngestion(file_list)
    if not documents:
        return DATA_NOT_FOUND("Failed to ingest data.")

    chunks = class_obj.DataChunking(documents)

    if not chunks:
        return DATA_NOT_FOUND("Failed to chunk data.")

    retriever = class_obj.TextEmbeddingAndVectorDb(chunks)
    if not retriever:
        return DATA_NOT_FOUND("Failed to create vector store.")

    response = class_obj.ModelInference(retriever, user_query)
    return response
