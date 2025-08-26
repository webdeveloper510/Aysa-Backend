from django.test import TestCase

"""
gender_map = {
"Women": "Women",
"Female": "Women",
"Men": "Men",
"male": "Men",
"Unisex": "Unisex",
"Unixes": "Unisex",
"Kids": "Kids",
"Boys": "Kids",
"Girls": "Kids",
"Baby": "Kids",
"Babies": "Kids",
}
# function to remove preprocessing
def preprocess_text(text):
import re
text = str(text).lower() # Lowercase
text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
return text

# ['Brand', 'Product Name', 'Product Type', 'Category', 'Gender', 'Production Year', 'Link to Product Pictures', 'Release Price', 'Profit Margin']

import pandas as pd
from sentence_transformers import SentenceTransformer , util
import os

csv_path = "/home/mandeep/Desktop/mandeep_personal/InsightHub/SearchMind/Data/profit_margins.csv"

df = pd.read_csv(csv_path)

df.columns = df.columns.str.strip() # Remove extra space form columns

# Make correction of gender column 
df['Gender'] = df['Gender'].fillna('Unisex') # Fill nan values with Unisex which is used for both
df['Gender'] = df['Gender'].astype(str).str.strip().map(gender_map) # Map gender columns with gender map dictionary 

df = df.dropna(subset=['Product Name']) # remove only rows which have no product Name
df.drop_duplicates(inplace=True) # Remove duplicacy from dataframe

df["text"] =( # Add Searchable column from filter out data
df["Brand"] +" " + df["Product Name"] + " " +df["Product Type"]
) 

df['text'] = df['text'].apply(preprocess_text) # Apply text preprocessing function on text data

transfer_model_path = os.path.join( # Upload Transfer model 
os.getcwd(), "transfer_model", 'all-MiniLM-L6-v2'
) 

model = SentenceTransformer(transfer_model_path) # Load Model

embeddings_full_text = model.encode(df['text'].tolist(), show_progress_bar=True) # Implement embedding on text column

df["text_embedding"] = list(embeddings_full_text) # Add new column with name text embedding 

embedding_dir_path = os.path.join(os.getcwd(), "EmbeddingDir", "Profit_Margin") # Made a Embedding dir path

embedding_df_path = os.path.join(embedding_dir_path,"profit_embedding.pkl") # Name of embedding model 

df.to_pickle(embedding_df_path)

print(f"Successfully Embedding df saved in {embedding_df_path}")
"""

