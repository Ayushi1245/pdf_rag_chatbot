import streamlit as st
from rag_pipeline import load_pdf, split_text, create_vector_store, get_answer
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot")
st.caption("PDF upload karo aur kuch bhi poochho!")
# Session state initialize karo
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
# Sidebar — PDF Upload
with st.sidebar:
    st.header("📂 PDF Upload")
    uploaded_file = st.file_uploader("PDF choose karo", type="pdf")
    
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("PDF padh raha hoon... ⏳"):
            text = load_pdf(uploaded_file)
            chunks = split_text(text)
            st.session_state.vector_store = create_vector_store(chunks)
            st.success(f"✅ PDF ready! {len(chunks)} chunks bane")
# Chat history dikhao
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
# User input
if question := st.chat_input("PDF ke baare mein kuch bhi poochho..."):
    if st.session_state.vector_store is None:
        st.warning("⚠️ Pehle PDF upload karo!")
    else:
        # User message
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)
        
        # Bot answer
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
                answer = get_answer(
                    st.session_state.vector_store, question
                )
            st.write(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )