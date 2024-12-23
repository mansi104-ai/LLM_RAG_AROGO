import streamlit as st
from text_extraction import extract_text_from_pdf
from hierarchical_indexing import create_hierarchical_index, index_to_dict
from retrieval import create_faiss_index, retrieve_relevant_text
from question_answering import answer_question

# Initialize session state
if 'textbooks' not in st.session_state:
    st.session_state.textbooks = {}

st.title("Advanced Textbook Q&A System")

# File upload
uploaded_files = st.file_uploader("Choose up to 3 PDF textbooks", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files[:3]:  # Limit to 3 textbooks
        if file.name not in st.session_state.textbooks:
            with st.spinner(f"Processing {file.name}..."):
                # Extract text
                text = extract_text_from_pdf(file)
                
                # Create hierarchical index
                index = create_hierarchical_index(text)
                
                # Create FAISS index for efficient retrieval
                faiss_index, indexed_texts = create_faiss_index([node.content for node in index.children])
                
                st.session_state.textbooks[file.name] = {
                    "text": text,
                    "index": index_to_dict(index),
                    "faiss_index": faiss_index,
                    "indexed_texts": indexed_texts
                }
            
            st.success(f"{file.name} processed successfully!")

# Textbook selection
if st.session_state.textbooks:
    selected_textbook = st.selectbox("Select a textbook to query:", list(st.session_state.textbooks.keys()))
    
    # User query input
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Searching for relevant information and generating answer..."):
            # Retrieve relevant text
            relevant_texts = retrieve_relevant_text(
                query, 
                st.session_state.textbooks[selected_textbook]["faiss_index"],
                st.session_state.textbooks[selected_textbook]["indexed_texts"]
            )
            
            # Combine relevant texts
            context = " ".join(relevant_texts)
            
            # Answer question
            answer = answer_question(query, context)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Relevant Context:")
            for i, text in enumerate(relevant_texts, 1):
                st.write(f"{i}. {text[:200]}...")  # Show first 200 characters of each relevant text
else:
    st.info("Please upload at least one textbook to start querying.")

