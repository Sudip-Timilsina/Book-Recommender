
# Smart Book Recommender with LLM Intelligence

This project is an AI-powered book recommendation system that leverages modern large language models (LLMs) and natural language processing to enhance discoverability and personalization of book suggestions.

## 🔍 Key Features

- **LLM-Based Semantic Search**: Uses embeddings from Sentence Transformers (`all-MiniLM-L6-v2`) to enable natural language search over book descriptions.
- **Zero-Shot Text Classification**: Book genres and tags are inferred using zero-shot classification techniques based on textual content.
- **Emotion Detection**: Supports emotion-based filtering using pre-tagged sentiment scores (e.g., joy, sadness, anger).
- **Tag-Aware Recommendations**: Books are preprocessed into "tagged descriptions" using advanced LLM pipelines to improve relevance.

---

## 🧠 LLM and NLP Components

### 1. **Semantic Vector Search**
- **Model Used**: `sentence-transformers/all-MiniLM-L6-v2`
- **Tool**: `LangChain + Chroma`
- **How it Works**:
  - Each tagged book description is split using `CharacterTextSplitter`
  - The embeddings are generated using HuggingFace
  - Stored and searched via `Chroma` vector DB
  - Similarity search returns the most relevant book entries

### 2. **Zero-Shot Classification (Conceptual)**
- LLMs like `BART` or `T5` can classify texts into multiple labels (e.g., genre or theme) without training.
- This concept inspired the structure of `tagged_description` to support rich queries.

### 3. **Sentiment and Emotion Scoring**
- Each book is scored across common emotion axes (e.g., anger, fear, joy) with pre-trained LLM pipelines via `transformers`.
- Emotion filtering lets users search based on how they want a book to make them feel.

---

## 📦 Dataset

- File: `books_with_emotions.csv`
- Fields:
  - `isbn13`, `title`, `authors`, `description`, `thumbnail`, `average_rating`
  - `tagged_description` (preprocessed for vector embedding)
  - Emotion scores: `joy`, `sadness`, `anger`, `surprise`, etc.
  - `simple_categories` (cleaned genre tags)

---

## 💻 Interface (Minimal Django UI)

- Simple search bar, genre and emotion dropdown filters
- Modal popup for detailed book preview
- Responsive, clean, single-page layout

---

## 🚀 How to Run

```bash
# Set up Python env
pip install -r requirements.txt

# Run Django app
python manage.py runserver
```

Ensure the `books_with_emotions.csv` and `tagged_description.txt` exist in your configured path.

---

## 🔮 Future Ideas

- Add user personalization using embeddings of user preferences
- Real-time zero-shot tagging with OpenAI or Cohere APIs
- Generate book summaries and mood previews using LLMs

---

## 📚 Powered By

- 🤗 HuggingFace Transformers
- 🦜 LangChain + Chroma DB
- 🐍 Python + Django
- 📘 Sentence-Transformers (MiniLM)
