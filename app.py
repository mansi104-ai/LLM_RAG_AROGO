import streamlit as st
import PyPDF2
import nltk
from gensim import corpora, models
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import groq
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')

# Initialize Groq client with API Key
try:
    api_key = os.environ.get("GROQ_API_KEY", "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")
    client = groq.Groq(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")

# Load SentenceTransformer model
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Failed to load SentenceTransformer model: {e}")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to create a simple hierarchical index and embed sentences
# Function to create a simple hierarchical index and embed sentences
def create_hierarchical_index(text):
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text provided for hierarchical indexing.")

        # Tokenize text into sentences
        sentences = nltk.sent_tokenize(text)

        # Ensure all entries in sentences are non-empty strings
        sentences = [str(sentence).strip() for sentence in sentences if sentence.strip()]
        if not sentences:
            raise ValueError("No valid sentences found in the text.")

        # Create embeddings for valid sentences
        embeddings = sentence_model.encode(sentences, convert_to_tensor=True)

        # Group sentences into paragraphs and chapters
        paragraphs = [sentences[i:i+5] for i in range(0, len(sentences), 5)]
        chapters = [paragraphs[i:i+10] for i in range(0, len(paragraphs), 10)]

        return {"chapters": chapters, "sentences": sentences, "embeddings": embeddings}

    except Exception as e:
        st.error(f"Error creating hierarchical index: {e}")
        return {}

# Function for semantic retrieval using embeddings
def retrieve_relevant_text(query, index):
    try:
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, index["embeddings"])[0]
        top_index = scores.argmax().item()
        return index["sentences"][top_index]
    except Exception as e:
        st.error(f"Error retrieving relevant text: {e}")
        return ""

# Function to answer questions using Groq
def answer_question(question, context):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
            ],
            model="mixtral-8x7b-32768",
            max_tokens=200
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error answering the question: {e}")
        return "Unable to answer the question."

# Streamlit UI
st.title("Textbook Q&A System")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    if text:
        st.success("PDF uploaded and text extracted successfully!")
        st.write("Extracted Text Preview:", text[:500])  # Show first 500 characters as a preview

        # Create hierarchical index with embeddings
        index = create_hierarchical_index(text)
        if index:
            st.success("Hierarchical index created!")

            # User query input
            query = st.text_input("Enter your question:")

            if query:
                # Retrieve relevant text using embeddings
                relevant_text = retrieve_relevant_text(query, index)
                if relevant_text:
                    st.write("Relevant text:", relevant_text)

                    # Answer question using Groq
                    answer = answer_question(query, relevant_text)
                    st.write("Answer:", answer)
                else:
                    st.error("Could not retrieve relevant text.")
        else:
            st.error("Failed to create hierarchical index.")
    else:
        st.error("No text extracted from the PDF.")
