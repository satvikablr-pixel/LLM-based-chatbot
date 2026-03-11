import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

st.title("AI Chatbot")
st.write("Type something and get a detailed explanation. Type 'exit' to stop.")

# Chat input
user_input = st.text_input("You:")

if user_input:
    if user_input.lower() == "exit":
        st.write("Goodbye! 👋")
    else:
        prompt = "Explain in detail: " + user_input
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_new_tokens=200)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("Bot:", value=answer, height=150)