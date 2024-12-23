import streamlit as st
from text_extraction import extract_text_from_pdf
from hierarchical_indexing import create_hierarchical_index, index_to_dict
from retrieval import create_faiss_index, retrieve_relevant_text
from question_answering import answer_question
import base64

# Set page configuration
st.set_page_config(page_title="SOWA", page_icon="📚", layout="wide")

# Custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# App title and description
# Sidebar
with st.sidebar:
    st.title("SOWA")
    st.markdown("Unlock the knowledge within your textbooks! 🚀")
    
    # File upload
    uploaded_files = st.file_uploader("Choose up to 3 PDF textbooks", type="pdf", accept_multiple_files=True)

# Initialize session state
if 'textbooks' not in st.session_state:
    st.session_state.textbooks = {}

# Main content
st.title("📚 SOWA")

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

# Textbook selection and querying
if st.session_state.textbooks:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_textbook = st.selectbox("Select a textbook:", list(st.session_state.textbooks.keys()))
    
    with col2:
        query = st.text_input("Enter your question:", placeholder="What would you like to know?")
    
    if query:
        with st.spinner("🔍 Searching for answers..."):
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
            
            # Display answer
            st.markdown("## 💡 Answer")
            st.info(answer)
            
            # Display relevant context
            with st.expander("📖 Relevant Context", expanded=False):
                for i, text in enumerate(relevant_texts, 1):
                    st.markdown(f"**Excerpt {i}:**")
                    st.write(text[:300] + "...")  # Show first 300 characters of each relevant text
                    st.markdown("---")

else:
    st.info("👆 Please upload at least one textbook to start querying.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Mansi")
