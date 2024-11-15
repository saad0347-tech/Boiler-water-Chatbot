
# app.py

import pandas as pd
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load data
df = pd.read_csv('/content/gdrive/MyDrive/12fnX6XSxPALlD4gEF_kiy_xw40AV6aCQ/boiler chemistry.csv')

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define chatbot function
def chatbot(question):
    inputs = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    output = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Create Streamlit app
st.title("Boiler Chemistry Chatbot")
st.write("Ask me a question!")

question = st.text_input("Question")
if st.button("Ask"):
    answer = chatbot(question)
    st.write("Answer:", answer)
