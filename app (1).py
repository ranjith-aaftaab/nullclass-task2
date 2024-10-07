import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import threading
import time
from datetime import datetime

# ------------------------------
# Configuration
# ------------------------------

# List of RSS feed URLs
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",  # BBC News
    "http://rss.cnn.com/rss/edition.rss",    # CNN
    "https://feeds.npr.org/1001/rss.xml",    # NPR News
    "https://www.aljazeera.com/xml/rss/all.xml",  # Al Jazeera
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml"  # Wall Street Journal
]

# Embedding model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# FAISS configuration
EMBEDDING_DIMENSION = 384  # Dimension of 'all-MiniLM-L6-v2' embeddings

# Persistence files
INDEX_FILE = "faiss_index.bin"
ARTICLES_FILE = "articles.pkl"

# ------------------------------
# Initialize Embedding Model
# ------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedding_model = load_embedding_model()

# ------------------------------
# Initialize or Load FAISS Index
# ------------------------------

@st.cache_resource
def load_faiss_index():
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        return index
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        return index

index = load_faiss_index()

# ------------------------------
# Load or Initialize Articles
# ------------------------------

@st.cache_data
def load_articles():
    if os.path.exists(ARTICLES_FILE):
        with open(ARTICLES_FILE, "rb") as f:
            articles = pickle.load(f)
    else:
        articles = []
    return articles

@st.cache_data
def save_articles(articles):
    with open(ARTICLES_FILE, "wb") as f:
        pickle.dump(articles, f)

# Initialize session state for articles
if 'articles' not in st.session_state:
    st.session_state['articles'] = load_articles()

# ------------------------------
# Function to Fetch Latest News from RSS Feeds
# ------------------------------

def fetch_latest_news(rss_feeds):
    articles = []
    for feed_url in rss_feeds:
        parsed_feed = feedparser.parse(feed_url)
        for entry in parsed_feed.entries:
            title = entry.get('title', '')
            description = entry.get('description', '')
            link = entry.get('link', '')
            published = entry.get('published', '')
            # Combine title, description, link, and published date
            full_text = f"{title}. {description}. Published on: {published}. Source: {link}"
            articles.append(full_text)
    return articles

# ------------------------------
# Function to Generate Embeddings
# ------------------------------

def generate_embeddings(text_list):
    return embedding_model.encode(text_list, convert_to_numpy=True, show_progress_bar=True)

# ------------------------------
# Function to Update FAISS Index
# ------------------------------

def update_faiss_index(new_embeddings):
    faiss.normalize_L2(new_embeddings)  # Normalize for cosine similarity
    index.add(new_embeddings)
    faiss.write_index(index, INDEX_FILE)  # Persist the index

# ------------------------------
# Function to Update Knowledge Base
# ------------------------------

def update_knowledge_base(rss_feeds):
    st.write("Fetching latest news articles...")
    new_articles = fetch_latest_news(rss_feeds)
    
    # Remove duplicates
    unique_new_articles = [article for article in new_articles if article not in st.session_state['articles']]
    
    if not unique_new_articles:
        st.warning("No new unique articles fetched.")
        return
    
    st.write(f"Fetched {len(unique_new_articles)} new unique articles.")
    
    # Generate embeddings
    st.write("Generating embeddings for new articles...")
    new_embeddings = generate_embeddings(unique_new_articles)
    
    # Update FAISS index
    st.write("Updating FAISS index with new embeddings...")
    update_faiss_index(new_embeddings)
    
    # Update articles list
    st.session_state['articles'].extend(unique_new_articles)
    save_articles(st.session_state['articles'])  # Persist articles
    
    st.success(f"Knowledge base updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------
# Scheduler Function
# ------------------------------

def run_scheduler(rss_feeds, interval_hours=24):
    while True:
        update_knowledge_base(rss_feeds)
        time.sleep(interval_hours * 3600)  # Sleep for the specified interval

# ------------------------------
# Start Scheduler in a Separate Thread
# ------------------------------

if 'scheduler_started' not in st.session_state:
    scheduler_thread = threading.Thread(target=run_scheduler, args=(RSS_FEEDS,), daemon=True)
    scheduler_thread.start()
    st.session_state['scheduler_started'] = True

# ------------------------------
# Initial Knowledge Base Update
# ------------------------------

if not st.session_state['articles']:
    update_knowledge_base(RSS_FEEDS)

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Dynamic Knowledge Chatbot")

st.sidebar.header("Settings")

# Manual Update Button
if st.sidebar.button("Update Knowledge Base Now"):
    update_knowledge_base(RSS_FEEDS)

# User Query
st.subheader("Ask the Chatbot")
user_query = st.text_area("Enter your question here:", height=100)
submit = st.button("Get Answer")

if submit and user_query:
    if len(st.session_state['articles']) == 0:
        st.error("Knowledge base is empty. Please update it first.")
    else:
        # Generate embedding for the query
        query_embedding = embedding_model.encode([user_query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        k = 5  # Number of nearest neighbors
        distances, indices = index.search(query_embedding, k)
        
        # Retrieve relevant articles
        relevant_articles = [st.session_state['articles'][i] for i in indices[0] if i < len(st.session_state['articles'])]
        
        # Display the results
        st.subheader("Chatbot's Response:")
        for idx, article in enumerate(relevant_articles, 1):
            # Extract source URL for "Read More" link
            if "Source: " in article:
                source = article.split("Source: ")[-1]
            else:
                source = "#"
            snippet = article[:200] + "..." if len(article) > 200 else article
            st.markdown(f"**Article {idx}:** {snippet} [Read More]({source})")

# ------------------------------
# Display Last Updated Time
# ------------------------------

st.markdown("---")
st.markdown("**Last Updated:**")
if st.session_state['articles']:
    st.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.write("Never")
