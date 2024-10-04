import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from knowledgeflow.core import KnowledgeFlow
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Assuming you're using OpenAI GPT-4

# Initialize the sentence transformer model for generating embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to fetch up to 100 latest news articles based on a keyword
def get_current_news_texts(query, api_key, page_size=100):
    url = f"https://newsapi.org/v2/everything?q={query}&from=2024-08-28&sortBy=publishedAt&language=en&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        # Extract the titles and descriptions to use as current texts
        return [article['title'] + ". " + article['description'] for article in articles]
    else:
        print(f"Failed to fetch news: {response.status_code}, {response.text}")
        return []

# Replace with your actual NewsAPI key set in environment variables or hard-code it
news_api_key = os.getenv("NEWS_API_KEY")

# Get up to 100 latest news texts for the query "technology" (you can change the topic as needed)
texts = get_current_news_texts("technology", news_api_key, page_size=100)


# Generate embeddings for the fetched news texts
if len(texts) > 0:
    vectors = model.encode(texts)
else:
    print("No news articles fetched; using default texts.")
    texts = [
        "Machine learning is fascinating.",
        "Artificial intelligence will change the world.",
        "I love natural language processing.",
        "FAISS is great for vector search.",
        "Text embeddings are useful for many applications.",
        "I do not like you at all.",
    ]
    vectors = model.encode(texts)

# Function to calculate the total number of words in the texts
def calculate_word_count(texts):
    total_word_count = sum([len(text.split()) for text in texts])
    return total_word_count

# Example usage with your `texts` list
total_word_count = calculate_word_count(texts)

print(f"Total number of words in texts: {total_word_count}")

# Function to calculate the size of the texts
def calculate_text_size(texts):
    # Calculate the total size in bytes
    total_size_in_bytes = sum([sys.getsizeof(text) for text in texts])
    
    # Convert size to kilobytes (KB) and megabytes (MB)
    total_size_in_kb = total_size_in_bytes / 1024
    total_size_in_mb = total_size_in_kb / 1024

    return total_size_in_bytes, total_size_in_kb, total_size_in_mb

# Example usage with your `texts` list
total_size_in_bytes, total_size_in_kb, total_size_in_mb = calculate_text_size(texts)

print(f"Total size of texts:\n- {total_size_in_bytes} bytes\n- {total_size_in_kb:.2f} KB\n- {total_size_in_mb:.2f} MB")

# Initialize KnowledgeFlow with the news vectors and texts
knowledge_flow = KnowledgeFlow(
    index_type='ivf',
    dimension=vectors.shape[1],
    vectors=vectors,
    texts=texts,
    URI="mongodb://192.168.1.151:27017/",
    db_name='test_db',
    collection_name='test_collection'
)

def chatbot(query):
    # Generate embedding for the user's query
    query_vector = model.encode([query])[0]

    # Search for the top 3 similar texts from the news articles
    k = 3
    similar_texts = knowledge_flow.search(query_vector, k)

    # Construct a prompt for the model with context from the similar news articles
    prompt = "You are a helpful assistant. Based on the following relevant information:\n"
    for i, text in enumerate(similar_texts):
        prompt += f"{i + 1}. {text}\n"

    prompt += f"\nUser query: {query}\n"
    prompt += "Please generate a helpful response."

    # Use the OpenAI API to generate the response
    try:
        response_context_aware = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        response_context_aware = response_context_aware.choices[0].message.content.strip()

        prompt_unaware = "You are a helpful assistant. Based on the following relevant information:\n"
        prompt_unaware += f"\nUser query: {query}\n"
        prompt_unaware += "Please generate a helpful response."
        response_context_unaware = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_unaware}
            ],
            max_tokens=150
        )
        response_context_unaware = response_context_unaware.choices[0].message.content.strip()

        # Return the response
        return response_context_aware, response_context_unaware

    except openai.error.OpenAIError as e:
        # Handle any API errors
        print(f"Error: {e}")
        return "Sorry, I encountered an error while processing your request."

# Example chatbot interaction loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response_context_aware, response_context_unaware = chatbot(user_input)
    print(f"Chatbot Unaware: {response_context_unaware}")
    print(f"Chatbot Aware: {response_context_aware}")
