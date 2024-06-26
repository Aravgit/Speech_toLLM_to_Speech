import os
import openai
from llama_index import set_global_service_context
import base64
import json
import requests
from datetime import datetime, timedelta
from llama_index.response_synthesizers import get_response_synthesizer
import pandas as pd
import numpy as np
import shutil
from sklearn.preprocessing import MinMaxScaler
from llama_index import StorageContext, load_index_from_storage,LLMPredictor,ServiceContext,VectorStoreIndex
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import LangchainEmbedding,LLMPredictor
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI 
from langchain.embeddings import OpenAIEmbeddings
from llama_index import SummaryPrompt
from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.prompts.base import Prompt
import requests
import pyaudio
import sys
import ipywidgets as widgets
from IPython.display import display
import wave
import threading
import time 
import librosa
import soundfile as sf
from IPython.display import Audio
from gtts import gTTS
import IPython.display as ipd
import asyncio

asyncio.set_event_loop(asyncio.new_event_loop())

## Add your openai api keys here
os.environ["OPENAI_API_VERSION"] = '2023-05-15'
os.environ["OPENAI_API_BASE"] ="add_api_base"
os.environ["OPENAI_API_KEY"] = 'add_api_key'
os.environ["OPENAI_API_TYPE"] = "azure"
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = os.getenv("OPENAI_API_VERSION")


SYSTEM_MESSAGE=Prompt("""
      You are an expert Q&A engine that always answers questions with the most relevant information using the context at your disposal.
      The context have information regarding the question that the user has asked.
      Here are some guidelines that you must follow:
      * For business questions, you must use the context to find the answer and then write a response.
      * Even if it seems like your context won't be able to answer the question, you must still use it to find the most relevant information and insights. Not using them will appear as if you are not doing your job.
      * For any user message that isn't related to business analysis,return any useful information from the context provided.
      * If context does not contain an answer, you should say that you haven't found an answer but still relay any useful information from the context provided.
      * Never directly reference the given context in your answer.
      * Avoid statements like \'Based on the context, ...\' or \'The context information ...\' or anything along those lines.
     
      Context: {context_str}
      Question: {query_str}
    """)

def get_llm_response(question):
    count = 3
    mode = 'compact'
    async_mode=False
    llm_predictor = LLMPredictor(AzureChatOpenAI (deployment_name='gpt-35-turbo',model='gpt-3.5-turbo', 
                                             temperature=0,max_tokens=256,
                                             openai_api_key=openai.api_key,openai_api_base=openai.api_base,
                                             openai_api_type=openai.api_type,openai_api_version='2023-05-15',
                                             ))
    embeddings = LangchainEmbedding(OpenAIEmbeddings( deployment="text-embedding-ada-002",model="text-embedding-ada-002",
                                                    openai_api_base=openai.api_base,openai_api_key=openai.api_key,
                                                    openai_api_type=openai.api_type,openai_api_version=openai.api_version),
                                    embed_batch_size=1,)    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,embed_model=embeddings)

    client = qdrant_client.QdrantClient(url=f"{hostname}:6333")
    vector_store = QdrantVectorStore(client=client, collection_name=f"{collection_name}")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,service_context=service_context)
    response_synthesizer = get_response_synthesizer(response_mode=mode, use_async=async_mode,service_context=service_context, text_qa_template = SYSTEM_MESSAGE)
    query_engine = index.as_query_engine(response_synthesizer=response_synthesizer,similarity_top_k=count) )
    response = query_engine.query(question)
    return response.response

def run_knowpro(translation):
    start_time = time.time()
    answer = get_llm_response(translation)
    with open('output.txt', 'w') as f:
            f.write(answer)
    print(f"{answer}")
    return answer


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    result = run_knowpro(translation)
    print(result)