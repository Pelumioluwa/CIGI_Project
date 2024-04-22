## This py file develops a retrival augmented generation model 
#that uses a hybrid, KNN and vector search to retrieve the top K entries from Azure AI Search using text embedding.
#Users are able to interact with the RAG through a user interface that allows them to ask questions and get answers.


# Import necessary libraries
import streamlit as st
import os
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import pandas as pd
from openai import OpenAI
from dotenv import dotenv_values
import json 
import requests
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery   
)


#import Ai search credentials 

config = dotenv_values('/Users/pelumioluwaabiola/Desktop/Transcriptions/credential.env')
ai_search_location = config['ai_search_location']
ai_search_key = config['ai_search_key']
ai_search_url = config['ai_search_url']
ai_search_index = 'oewg-speech-meeeting-index'
ai_search_name = 'aicpcigi'
embedding_length = 768

openai_key = config['openai_api_key']
openai_deployment_name = "gpt-4"
search_client = SearchClient(ai_search_url, ai_search_index, AzureKeyCredential(ai_search_key)) 

#convert data to vector embeddings
def generate_embeddings(text):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = embedding_model.embed_query(text)
    return embeddings

#retrieve the top k entries from Azure AI Search using Vector search
def retrieve_vector_top_chunks(k, question_embedding):
    """Retrieve the top K entries from Azure AI Search using vector search with speaker embedding.""" 
    vector_query = VectorizedQuery(vector=question_embedding, 
                                k_nearest_neighbors=k, 
                                fields="TextEmbeddings")

    results = search_client.search(  
        search_text=None,  
        vector_queries=[vector_query],
        select=["Speaker", "Text","Meeting", "Session","ClusterLabel"],
        top=k
    )  
    output = [[f'Session: {result["Session"]}',f'Meeting: {result["Meeting"]}',f'Sentiment cluster: {result["ClusterLabel"]}',f'{result["Speaker"]}: {result["Text"]}'] for result in results]

    return output

#retrieve the top k entries from Azure AI Search using Hybrid search
def retrieve_hybrid_top_chunks(k, question, question_embedding):
    """Retrieve the top K entries from Azure AI Search using hybrid search with speaker embedding.""" 
    vector_query = VectorizedQuery(vector=question_embedding, 
                                k_nearest_neighbors=k, 
                                fields="TextEmbeddings")

    results = search_client.search(  
        search_text=question,  
        vector_queries=[vector_query],
        select=["Speaker", "Text","Meeting", "Session","ClusterLabel"],
        top=k
    )    

    output = [[f'Session: {result["Session"]}',f'Meeting: {result["Meeting"]}',f'Sentiment cluster: {result["ClusterLabel"]}',f'{result["Speaker"]}: {result["Text"]}'] for result in results]  

    return output

#convert the results from vector search to content 
def get_vector_context(user_question, retrieved_k = 5):
    # Generate embeddings for the question
    question_embedding = generate_embeddings(user_question)

    # Retrieve the top K entries
    output = retrieve_vector_top_chunks(retrieved_k, question_embedding)

    # concatenate the content of the retrieved documents
    context = '. '.join([item for sublist in output for item in sublist])

    return context

#convert the results from hybrid search to content 
def get_hybrid_context(user_question, retrieved_k = 5):
    # Generate embeddings for the question
    question_embedding = generate_embeddings(user_question)

    # Retrieve the top K entries
    output = retrieve_hybrid_top_chunks(retrieved_k, user_question, question_embedding)

    # concatenate the content of the retrieved documents
    context = '. '.join([item for sublist in output for item in sublist])

    return context

#define chatbot 
def prompt_engineering(VectorContext,HybridContext):
    # Define the chat context prompt
    VectorContext,HybridContext = VectorContext, HybridContext

    chat_context_prompt = f"""

    You are an assistant to answer questions about the Meetings by the UN Open Ended Working Group on Space Threats. 
    Do not hallucinate. 
    Use all the information below to answer your questions
    
    first information: {VectorContext}

    Second information : {HybridContext}

    If the answer to the question is not in information above,respond 'I am unable to provide a response on that'

    """

    return chat_context_prompt

client = OpenAI(api_key=openai_key)

def PR_Assistant(text,chat_context_prompt):
    MESSAGES = [
    {"role": "system", "content": chat_context_prompt},
    {"role": "user", "content": text},
    ]
    MESSAGES.append({"role": "user", "content": text})

    completion = client.chat.completions.create(model="gpt-4", messages=MESSAGES,temperature=0.9)
    return completion.choices[0].message.content



# # Define the main function
def main():
    st.title('Space Chat')

    # Create a text input field for the PDF file path
    input_pdf_path = st.text_input("Enter your question")

    # If the input field is not empty, call the summarize_pdf function and display the result
    if st.button('send'):
        if input_pdf_path:
            question = input_pdf_path
            VectorContext = get_vector_context(question, retrieved_k = 5)
            HybridContext = get_hybrid_context(question, retrieved_k = 5)
            chat_context_prompt = prompt_engineering(VectorContext,HybridContext)
            output_answer = PR_Assistant(question, chat_context_prompt)
            st.write(output_answer)

if __name__ == "__main__":
    main()

