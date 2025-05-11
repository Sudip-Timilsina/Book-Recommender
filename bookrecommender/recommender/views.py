import pandas as pd
from django.shortcuts import render
from random import shuffle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Constants
DATA_PATH = r"C:\Users\user\OneDrive\Desktop\book-recommender\recommender_llm\books_with_emotions.csv"
TAGGED_DESC_PATH = "tagged_description.txt"


# Load the dataset once (with caching)
def load_data():
    df = pd.read_csv(DATA_PATH).fillna("")

    # Pre-compute genres and emotions
    GENRES = sorted(df['simple_categories'].dropna().unique())
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

    return df, GENRES, EMOTIONS


# Initialize semantic search
def init_semantic_search(df):
    if not os.path.exists(TAGGED_DESC_PATH):
        df["tagged_description"].to_csv(TAGGED_DESC_PATH, sep="\n", index=False, header=False)

    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(TextLoader(TAGGED_DESC_PATH, encoding="utf-8").load())
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents, embedding_model)


# Load data and initialize components
df, GENRES, EMOTIONS = load_data()
db_books = init_semantic_search(df)


def semantic_retrieve(query: str, top_k: int = 10) -> pd.DataFrame:
    results = db_books.similarity_search(query, k=16)
    isbn_matches = [int(doc.page_content.strip('"').split()[0]) for doc in results]
    return df[df["isbn13"].isin(isbn_matches)].drop_duplicates(subset=["title"]).head(top_k)


def home(request):
    # Get query parameters
    query = request.GET.get('q', '').strip()
    genre = request.GET.get('genre', '').strip()
    emotion = request.GET.get('emotion', '').strip()

    # Apply filters
    filtered_df = df.copy()

    if genre:
        filtered_df = filtered_df[filtered_df['simple_categories'].str.contains(genre, case=False, na=False)]

    if emotion:
        filtered_df = filtered_df[filtered_df[emotion].astype(float) > 0.7]

    # Handle search query
    if query:
        text_filtered = filtered_df[
            filtered_df['description'].str.contains(query, case=False)
            | filtered_df['title'].str.contains(query, case=False)
            | filtered_df['tagged_description'].str.contains(query, case=False)
            ]

        if text_filtered.empty:
            filtered_df = semantic_retrieve(query)
            # Reapply filters to semantic results
            if genre:
                filtered_df = filtered_df[filtered_df['simple_categories'].str.contains(genre, case=False, na=False)]
            if emotion:
                filtered_df = filtered_df[filtered_df[emotion].astype(float) > 0.7]
        else:
            filtered_df = text_filtered

    # Prepare results
    filtered_df = filtered_df.drop_duplicates(subset=['title'])
    filtered_df['short_description'] = filtered_df['description'].str.split().str[:30].str.join(' ') + '...'

    books = filtered_df[['title', 'authors', 'average_rating', 'thumbnail',
                         'short_description', 'description']].head(10).to_dict(orient='records')

    # Shuffle only for new users (no search query)
    if not query:
        shuffle(books)

    return render(request, 'recommender/index.html', {
        'books': books,
        'genres': GENRES,
        'emotions': EMOTIONS,
        'selected_genre': genre,
        'selected_emotion': emotion,
        'query': query,
    })