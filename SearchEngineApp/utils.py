from textblob import TextBlob
import re

PRODUCT_DATA_KEYS = [
    'Brand', 'Product Name', 'Type', 'Production Year',
    'Link to Product Pictures', 'Release Price',
    'Profit Made', 'Profit Margin'
]

TAX_DATA_KEYS = ['Company Name', 'Year', 'Tax Paid', 'Tax Avoided']

CEO_WORKER_DATA_KEYS =['Company Name', 'Year', 'CEO Name', 'CEO Total Compensation', 'Frontline Worker Salary']

# function to convert data into list 
def ListToDict(input_list):
    data_dict ={}
    for item in input_list :
        split_item=  item.split(":", 1)
        data_dict[split_item[0]] =split_item[1]
    return data_dict

# Function to check response dict has all rows data
def is_valid_product(data, required_keys):
    return all(key in data and data[key].strip() != '' for key in required_keys)

# function to check grammer corrector
def SpellCorrector(input_str:str) -> str:
    correct_string = TextBlob(input_str)
    return str(correct_string.correct()).lower()


# function to remove preprocessing
def preprocess_text(text):
    text = str(text).lower()                          # Lowercase
    text = re.sub(r'[^\w\s]', '', text)               # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()          # Remove extra spaces
    return text



def select_most_similar_by_brand_year(df):
    import pandas as pd
    compare_rows = []

    for brand, group in df.groupby("Brand"):
        # Get available years in descending order
        highest_margin = group.loc[group["Profit Margin"].idxmax()]
        lowest_margin = group.loc[group["Profit Margin"].idxmin()]

        # extend in compare rows
        compare_rows.extend([highest_margin, lowest_margin])

    
    # Create DataFrame from results
    compare_df = pd.DataFrame(compare_rows).reset_index(drop=True)
    return compare_df
        

def filter_plus_rows(df, cleaned_product_type):
    # Return a boolean Series, not a filtered DataFrame
    return df["Type_clean"].str.contains(cleaned_product_type, case=False, na=False)

    


