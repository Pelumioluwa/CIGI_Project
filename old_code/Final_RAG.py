import json
import pandas as pd
import requests
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery   
)
import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration,BertModel, BertTokenizer 
from dotenv import dotenv_values
from pprint import pprint


#import Ai search credentials 
config = dotenv_values('credential.env')
ai_search_location = config['ai_search_location']
ai_search_key = config['ai_search_key']
ai_search_url = config['ai_search_url']
ai_search_index = 'oewg-speech-meeeting-index'
ai_search_name = 'aicpcigi'
embedding_length = 768

search_client = SearchClient(ai_search_url, ai_search_index, AzureKeyCredential(ai_search_key)) 


# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased' 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def generate_contextual_embedding(text):
    # Tokenize input text and enforce maximum sequence length
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    # Generate contextual embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        # Extract contextual embeddings from the last layer
        last_hidden_states = outputs.last_hidden_state
    # Average pooling over tokens to get a single vector representation
    contextual_embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
    return contextual_embedding


def retrieve_hybrid_top_chunks(k, question, question_embedding):
    """Retrieve the top K entries from Azure AI Search using hybrid search with speaker embedding.""" 
    vector_query = VectorizedQuery(vector=question_embedding, 
                                k_nearest_neighbors=k, 
                                fields="TextEmbeddings")

    results = search_client.search(  
        search_text=question,  
        vector_queries=[vector_query],
        select=["Speaker", "Text","Meeting", "Session"],
        top=k
    )    

    output = [[f'Session: {result["Session"]}',f'Meeting: {result["Meeting"]}',f'{result["Speaker"]}: {result["Text"]}'] for result in results]  

    return output

#using hybrid context because it's search is more related
def get_hybrid_context(user_question, retrieved_k = 5):
    # Generate embeddings for the question
    question_embedding = generate_contextual_embedding(user_question)

    # Retrieve the top K entries
    output = retrieve_hybrid_top_chunks(retrieved_k, user_question, question_embedding)

    # concatenate the content of the retrieved documents
    context = '. '.join([item for sublist in output for item in sublist])

    return context

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)


def format_query(query, context):
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query

def generate_answer(query):
    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt").to(device)
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer

def main():
    st.title('Space Chat')

    # Create a text input field for the PDF file path
    input_pdf_path = st.text_input("Enter your question")

    # If the input field is not empty, call the summarize_pdf function and display the result
    if st.button('send'):
        if input_pdf_path:
            query = input_pdf_path
            context = get_hybrid_context(query, retrieved_k = 5)
            query = format_query(query, context)
            output_answer = generate_answer(query)
            st.write(output_answer)

if __name__ == "__main__":
    main()

