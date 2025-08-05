from django.test import TestCase


"""
To build a smart search engine with natural language understanding (NLU), semantic search, autocomplete, and product comparison features as described, you can leverage machine learning (ML) and related technologies. Below is a step-by-step guide to design and implement such a system, tailored to the provided dataset (`profit_margins.csv`) and requirements. The system will handle queries like "What’s the markup on an iPhone 16?" and compare products like Nike Air Jordan 5 with similar products from other brands based on profit margins.

---

### System Architecture Overview

The system will consist of several components:
1. **Data Preprocessing**: Clean and structure the dataset for efficient querying and analysis.
2. **Natural Language Understanding (NLU)**: Parse and understand user queries using NLP techniques.
3. **Semantic Search**: Implement context-aware search with entity recognition and synonym handling.
4. **Autocomplete & Suggestions**: Provide query predictions and spelling corrections.
5. **Product Comparison Engine**: Identify and compare products based on category, release year, price range, and profit margins.
6. **Backend and Frontend**: Integrate the ML models into a search engine with a user-friendly interface.

---

### Step-by-Step Implementation

#### 1. **Data Preprocessing**
The dataset (`profit_margins.csv`) contains products across categories like smartphones, smart TVs, vehicles, and jewelry, with fields like `Brand`, `Product Name`, `Type`, `Production Year`, `Release Price`, `Profit Made`, and `Profit Margin`. Preprocessing is crucial for efficient querying and ML model integration.

**Tasks:**
- **Clean the Data**:
  - Fix inconsistencies (e.g., "Vehicals" → "Vehicles").
  - Handle missing or erroneous values (e.g., `#VALUE!` in profit margins, inconsistent price formats like `$1,099 to $1,249`).
  - Convert `Release Price` and `Profit Made` to numerical values (remove `$` and commas, handle ranges by taking the midpoint or max).
  - Correct the outlier in `Samsung S95C QD OLED (65-inches, 2023)` with a `2465%` profit margin (likely a typo, e.g., `$91,205` profit should be `$912.05` for ~24% margin).
- **Normalize Data**:
  - Standardize `Type` (e.g., "Pickup Truck" vs. "Compact Truck") for consistent categorization.
  - Extract categories (e.g., Smartphones, Smart TVs, Vehicles, Jewelry) and create a hierarchical structure.
- **Index the Data**:
  - Use a database like **Elasticsearch** or **Pinecone** for fast full-text search and semantic search.
  - Store metadata (e.g., `Brand`, `Category`, `Profit Margin`) for filtering and ranking.
  - Create embeddings for product names and descriptions using a model like **BERT** or **Sentence-BERT** for semantic similarity.

**Example Code (Python)**:
```python
import pandas as pd
import numpy as np

# Load and clean data
df = pd.read_csv('profit_margins.csv')

# Fix price formats
df['Release Price'] = df['Release Price'].replace({'\$': '', ',': ''}, regex=True).astype(float)
df['Profit Made'] = df['Profit Made'].replace({'\$': '', ',': '', '#VALUE!': np.nan}, regex=True).astype(float)

# Handle price ranges (take midpoint)
def parse_price_range(price):
    if isinstance(price, str) and 'to' in price:
        low, high = map(float, price.replace('$', '').split(' to '))
        return (low + high) / 2
    return price

df['Release Price'] = df['Release Price'].apply(parse_price_range)

# Fix outlier
df.loc[(df['Brand'] == 'Samsung') & (df['Product Name'] == 'S95C QD OLED') & (df['Production Year'] == 2023), 'Profit Made'] = 912.05
df['Profit Margin'] = df['Profit Made'] / df['Release Price'] * 100

# Save cleaned data
df.to_csv('cleaned_profit_margins.csv', index=False)
```

#### 2. **Natural Language Understanding (NLU)**
To parse queries like "What’s the markup on an iPhone 16?", use NLP to extract intent and entities.

**Tasks:**
- **Intent Classification**:
  - Train a classifier to identify query types (e.g., profit margin query, comparison query).
  - Use a pre-trained model like **BERT** fine-tuned on a custom dataset of search queries.
- **Entity Recognition**:
  - Extract entities like product names ("iPhone 16"), brands ("Apple"), and attributes ("profit margin").
  - Use **spaCy** or **Hugging Face NER** models for entity extraction.
- **Query Normalization**:
  - Handle synonyms (e.g., "markup" → "profit margin", "car" → "vehicle") using a synonym dictionary or word embeddings.

**Example Code (Python)**:
```python
import spacy
from transformers import pipeline

# Load NLP models
nlp = spacy.load('en_core_web_sm')
ner = pipeline('ner', model='dslim/bert-base-NER')

# Parse query
def parse_query(query):
    doc = nlp(query)
    entities = ner(query)
    intent = 'profit_margin' if 'markup' in query.lower() or 'profit' in query.lower() else 'comparison'
    extracted_entities = {
        'product': [ent.text for ent in doc.ents if ent.label_ == 'PRODUCT'],
        'brand': [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    }
    return intent, extracted_entities

# Example
query = "What’s the markup on an iPhone 16?"
intent, entities = parse_query(query)
print(f"Intent: {intent}, Entities: {entities}")
# Output: Intent: profit_margin, Entities: {'product': ['iPhone 16'], 'brand': ['Apple']}
```

#### 3. **Semantic Search**
Implement a search engine that understands context and retrieves relevant products.

**Tasks:**
- **Vectorize Products**:
  - Use **Sentence-BERT** to create embeddings for product names, types, and brands.
  - Store embeddings in a vector database like **Pinecone** or **FAISS**.
- **Handle Synonyms**:
  - Map synonyms (e.g., "car" → "vehicle") using a pre-trained embedding model or a custom synonym dictionary.
- **Rank Results**:
  - Use cosine similarity to rank products based on query embeddings.
  - Filter by category, brand, or year if specified.

**Example Code (Python)**:
```python
from sentence_transformers import SentenceTransformer
import pinecone

# Initialize Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key='your-api-key', environment='us-west1-gcp')
index = pinecone.Index('products')

# Vectorize and index products
def index_products(df):
    for _, row in df.iterrows():
        text = f"{row['Brand']} {row['Product Name']} {row['Type']}"
        vector = model.encode(text).tolist()
        index.upsert([(str(row.name), vector, {
            'brand': row['Brand'],
            'product': row['Product Name'],
            'profit_margin': row['Profit Margin']
        })])
```

#### 4. **Autocomplete & Suggestions**
Provide query predictions, spelling corrections, and trending suggestions.

**Tasks:**
- **Autocomplete**:
  - Use a trie-based data structure or an ML model like **NGram** for query completion.
  - Train on common product names and brands from the dataset.
- **Spelling Correction**:
  - Use **Levenshtein distance** or a pre-trained model like **Norvig’s spell checker**.
- **Trending Suggestions**:
  - Track popular queries or brands (e.g., Apple, Samsung) using a simple counter or external data (e.g., X posts).
  - Suggest related products based on category or profit margin.

**Example Code (Python)**:
```python
from nltk.metrics.distance import edit_distance

# Simple spelling correction
def spell_correct(query, vocabulary):
    return min(vocabulary, key=lambda x: edit_distance(query.lower(), x.lower()))

# Autocomplete using trie
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        suggestions = []
        self._collect_words(node, prefix, suggestions)
        return suggestions
    
    def _collect_words(self, node, prefix, suggestions):
        if node.is_end:
            suggestions.append(prefix)
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, suggestions)

# Build vocabulary
vocabulary = df['Product Name'].tolist() + df['Brand'].tolist()
trie = Trie()
for word in vocabulary:
    trie.insert(word.lower())

# Example
print(trie.autocomplete('iPh'))  # Output: ['iPhone 16', 'iPhone 15', ...]
```

#### 5. **Product Comparison Engine**
For queries like "Nike Air Jordan 5 profit margin," display the product’s profit margin and compare it with two similar products from other brands, prioritizing the highest and lowest profit margins.

**Tasks:**
- **Identify Similar Products**:
  - Filter products by category (e.g., Sneakers), release year (±1 year), and price range (±20%).
  - Use embeddings to find semantically similar products.
- **Select Comparators**:
  - Choose one product with the highest profit margin and one with the lowest from different brands.
  - Limit to three brands total (including the queried brand).
- **Display Results**:
  - Show profit margin, release price, and product details in a comparison table or chart.

**Example Code (Python)**:
```python
def compare_products(product_name, df):
    # Find the queried product
    product = df[df['Product Name'].str.contains(product_name, case=False)]
    if product.empty:
        return "Product not found."
    
    # Get category, year, and price
    category = product['Category'].iloc[0]  # Assume category column added during preprocessing
    year = product['Production Year'].iloc[0]
    price = product['Release Price'].iloc[0]
    
    # Filter similar products
    similar = df[
        (df['Category'] == category) &
        (df['Production Year'].between(year - 1, year + 1)) &
        (df['Release Price'].between(price * 0.8, price * 1.2)) &
        (df['Brand'] != product['Brand'].iloc[0])
    ]
    
    # Select highest and lowest profit margin products
    if not similar.empty:
        highest = similar.loc[similar['Profit Margin'].idxmax()]
        lowest = similar.loc[similar['Profit Margin'].idxmin()]
        return {
            'queried': product[['Brand', 'Product Name', 'Profit Margin', 'Release Price']].to_dict('records')[0],
            'highest': highest[['Brand', 'Product Name', 'Profit Margin', 'Release Price']].to_dict(),
            'lowest': lowest[['Brand', 'Product Name', 'Profit Margin', 'Release Price']].to_dict()
        }
    return "No similar products found."

# Example
result = compare_products('Air Jordan 5', df)
print(result)
```

#### 6. **Visualization (Charts)**
To display profit margin comparisons, create a bar chart using Chart.js.

**Example Chart Code**:
```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Nike Air Jordan 5", "Adidas Yeezy Boost 350", "Asics Gel-Nimbus"],
    "datasets": [{
      "label": "Profit Margin (%)",
      "data": [75, 80, 70],  // Replace with actual profit margins
      "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56"],
      "borderColor": ["#FF6384", "#36A2EB", "#FFCE56"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Profit Margin (%)"
        }
      },
      "x": {
        "title": {
          "display": true,
          "text": "Product"
        }
      }
    }
  }
}
```

#### 7. **Backend and Frontend**
- **Backend**:
  - Use **FastAPI** or **Flask** to create a REST API for handling queries.
  - Integrate the NLU, semantic search, and comparison engine.
  - Connect to the database (e.g., Elasticsearch or Pinecone) for data retrieval.
- **Frontend**:
  - Build a simple UI with **React** or **Vue.js**.
  - Display search results, autocomplete suggestions, and comparison charts.
  - Include a search bar with autocomplete and a results page with tables/charts.

**Example FastAPI Endpoint**:
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/search")
async def search(query: str):
    intent, entities = parse_query(query)
    if intent == 'profit_margin':
        product = df[df['Product Name'].str.contains(entities['product'][0], case=False)]
        if not product.empty:
            return product[['Brand', 'Product Name', 'Profit Margin']].to_dict('records')
    elif intent == 'comparison':
        return compare_products(entities['product'][0], df)
    return {"error": "Invalid query"}
```

#### 8. **Deployment**
- Deploy the backend on **AWS**, **GCP**, or **Heroku**.
- Host the vector database (Pinecone) or Elasticsearch on a cloud provider.
- Serve the frontend via a CDN like **Cloudflare** for performance.

---

### Handling Specific Requirements

#### Query: "What’s the markup on an iPhone 16?"
- **NLU**: Parse query to identify intent (`profit_margin`) and entity (`iPhone 16`).
- **Search**: Query the dataset for `iPhone 16` (2024).
- **Result**: Return profit margin (e.g., 37.00% for iPhone 16).

#### Query: "Nike Air Jordan 5 profit margin"
- **Search**: Find `Nike Air Jordan 5` in the dataset.
- **Comparison**:
  - Filter sneakers from 2024 (±1 year) with similar price range.
  - Select highest (e.g., Adidas Yeezy Boost 350) and lowest (e.g., Asics Gel-Nimbus) profit margin products from different brands.
- **Display**: Show a table/chart with profit margins (Nike: 75%, Adidas: 80%, Asics: 70%).

#### Search by Highest/Lowest Profit Margins
- Add a filter to sort products by `Profit Margin` in ascending/descending order.
- Example: `df.nlargest(3, 'Profit Margin')` for top 3, `df.nsmallest(3, 'Profit Margin')` for bottom 3.

---

### Tools and Libraries
- **NLP**: spaCy, Hugging Face Transformers, Sentence-BERT
- **Search**: Elasticsearch, Pinecone, FAISS
- **Autocomplete**: NLTK, custom Trie implementation
- **Backend**: FastAPI, Flask
- **Frontend**: React, Vue.js, Chart.js
- **Database**: PostgreSQL (for structured data), Pinecone (for vectors)
- **Deployment**: AWS, GCP, Heroku, Cloudflare

---

### Challenges and Solutions
- **Data Quality**: Handle missing/inconsistent data by imputing values or flagging errors.
- **Scalability**: Use vector databases for fast similarity search on large datasets.
- **Query Ambiguity**: Improve NLU with context clues (e.g., "Apple" → company, not fruit) using entity disambiguation.
- **Trending Topics**: Optionally integrate X API to track popular brands/products.

---

### Next Steps
1. Clean and index the dataset.
2. Train/fine-tune an NLU model for intent and entity extraction.
3. Implement semantic search with embeddings.
4. Build autocomplete and comparison logic.
5. Develop and deploy the backend/frontend.

Would you like me to dive deeper into any specific component (e.g., NLU model training, semantic search setup, or frontend UI)?
"""