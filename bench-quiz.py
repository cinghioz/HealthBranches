from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import JSONLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from tqdm.notebook import tqdm
import faiss
import os
import glob
import pickle
import random
import pandas as pd
import re
import json
import ast
import argparse

from typing import Dict, List
import torch
from prompt import *

parser = argparse.ArgumentParser(description="LLM inference with optional baseline mode.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
args = parser.parse_args()

# Set BASELINE based on the argument
BASELINE = args.base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("##### BASELINE MODE #####\n" if BASELINE else "##### BENCHMARK MODE #####\n")

class VectorStore:
    def __init__(self, index_path: str, embedder_name: str = "mxbai-embed-large"):
        self.index_path = index_path
        self.embedder_name = embedder_name
        self.embedder = OllamaEmbeddings(model=embedder_name)
        self._load_vector_store()

    def _load_vector_store(self):
        if os.path.exists(self.index_path):
            print("### LOAD VECTOR DB ###")

            self.index = faiss.read_index(self.index_path+'index.faiss')
            
            with open(self.index_path+'doc_to_id.pkl', "rb") as f:
                self.index_to_doc_id = pickle.load(f)

            with open(self.index_path+'docstore.pkl', "rb") as f:
                self.docstore = pickle.load(f)

            self.vector_store = FAISS(
                embedding_function=self.embedder,
                index=self.index,
                docstore=self.docstore,
                index_to_docstore_id=self.index_to_doc_id
            )   
        else:
            print("### CREATE VECTOR DB ###")

            self.index = faiss.IndexFlatL2(len(self.embedder.embed_query('hello world')))
            self.index_to_doc_id = {}
            self.docstore = InMemoryDocstore()

            self.vector_store = FAISS(
                embedding_function=self.embedder,
                index=self.index,
                docstore=self.docstore,
                index_to_docstore_id=self.index_to_doc_id
            )

            if not os.path.exists(self.index_path):
                os.makedirs(self.index_path)

    def _load_documents(self, doc_path: str, doc_type: str = "*.txt") -> list[Document]:
        loader = DirectoryLoader(doc_path, glob=doc_type)
        documents = loader.load()
        return documents

    def _split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        return chunks

    def add_documents(self, doc_path: str, doc_type: str = "*.txt"):
        documents = self._load_documents(doc_path, doc_type)
        chunks = self._split_text(documents)
        self.vector_store.add_documents(documents=chunks)
        self._update_vector_db()
    
    def search(self, query: str, k: int = 3):
        return self.vector_store.similarity_search(query=query, k=k)
        # return self.vector_store.similarity_search(query=self._transform_query(query), k=3)

    def _update_class(self):
        self.index = self.vector_store.index
        self.index_to_doc_id = self.vector_store.index_to_docstore_id
        self.docstore = self.vector_store.docstore
    
    def _update_vector_db(self):
        faiss.write_index(self.vector_store.index, self.index_path+'index.faiss')

        with open(self.index_path+'doc_to_id.pkl', "wb") as f:
            pickle.dump(self.vector_store.index_to_docstore_id, f)

        with open(self.index_path+'docstore.pkl', "wb") as f:
            pickle.dump(self.vector_store.docstore, f)

        self._update_class()

        print("### UPDATE VECTOR DB ###")

class LLMinference:
    def __init__(self, llm_name, temperature=0.5):
        self.llm_name = llm_name
        self.model = OllamaLLM(model=llm_name, temperature=temperature) 

    def _transform_query(self, query: str) -> str:
        return f'Represent this sentence for searching relevant passages: {query}'

    def single_inference(self, query: str, template: str, path: str, text: str,  choices: List[str], cond: str,  context) -> str | List[str]:
        context_text = "\n\n---\n\n".join([doc.page_content for doc in context])
        prompt_template = ChatPromptTemplate.from_template(template)
        
        if path != "" and text != "":
            prompt = prompt_template.format(context=context_text, question=query, path=path, text=text, condition=cond, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
        else:
            prompt = prompt_template.format(context=context_text, question=query, condition=cond, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])

        response_text = self.model.invoke(prompt)
        response_text = response_text.strip().replace("\n", "").replace("  ", "")

        sources = [doc.metadata.get("source", None) for doc in context]
        
        return response_text, sources

    def qea_evaluation(self, query: str, template: str, path: str, txt: str, choices: List[str], cond: str,  vector_store):

        results = vector_store.search(query=query, k=3)

        response, sources = self.single_inference(query, template, path, txt, choices, cond, results)

        return response

def check_results(search_directory: str, string_to_check: str, search_strings: List[str]) -> List[str]:
    # Get all matching files
    matching_files = glob.glob(os.path.join(search_directory, string_to_check))

    # Find matching files and remove matched strings
    found_files = [file for file in matching_files if any(name in os.path.basename(file) for name in search_strings)]
    remaining_strings = [name for name in search_strings if not any(name in os.path.basename(file) for file in matching_files)]

    print("Remaining models to run: ", remaining_strings)

    return remaining_strings

# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore('/home/cc/PHD/ragkg/indexes/kgbase-new/')

# Add documents in vector store (comment this line after the first add)
# vector_store.add_documents('/home/cc/PHD/ragkg/data/kgbase')

folder_path = "/home/cc/PHD/ragkg/questions_pro/ultimate_questions_v2.csv"
questions = pd.read_csv(folder_path)

# models = ["mistral", "llama3.1:8b", "llama2:7b", "medllama2:7b", "gemma:7b", "gemma2:9b", "phi4:14b", "qwen2.5:7b", "mixtral:8x7b", "deepseek-r1:7b"]
models = ["mistral", "llama3.1:8b", "llama2:7b", "gemma:7b", "gemma2:9b", "qwen2.5:7b", "phi4:14b", "medllama2:7b"]
models = check_results('/home/cc/PHD/ragkg/', "results_quiz_baseline_v2_*.csv" if BASELINE else "results_quiz_v2_*.csv", models)

templates = [PROMPT_QUIZ, PROMPT_QUIZ_RAG]

if BASELINE:
    templates = [PROMPT_QUIZ_BASELINE]

cnt_rag = 0
cnt = 0

rows = []
questions = pd.read_csv(folder_path)

for model_name in models:
    llm = LLMinference(llm_name=model_name)

    cnt = 0
    rows = []

    for index, row in questions.iterrows():
        res = []
        try:
            opts = ast.literal_eval(row['options'].replace("['", '["').replace("']", '"]').replace("', '", '", "'))
            
            if not isinstance(opts, list) or len(opts) != 5:
                print(f"Skipping row {index} due to invalid options")
                continue  # Skip this iteration if the condition is not met

        except (ValueError, SyntaxError):
            print(f"Skipping row {index} due to value/syntax error")
            continue  # Skip if there's an issue with evaluation

        txt_name = row['condition'].upper()+".txt"
        txt_folder_name = "/home/cc/PHD/ragkg/data/kgbase-new/"

        try:
            with open(os.path.join(txt_folder_name, txt_name), 'r') as file:
                text = file.readlines()
        except Exception:
            print(os.path.join(txt_folder_name, txt_name))
            print(f"{txt_name} text is EMPTY!")
            continue    
        
        for template in templates:
            if BASELINE:
                res.append(llm.qea_evaluation(row['question'], template, row['path'], text, opts, row['condition'].lower(), vector_store)) # Baseline
            else:
                res.append(llm.qea_evaluation(row['question'], template, "", "", opts, row['condition'].lower(), vector_store))

        res.append(row["correct_option"])
        res.append(row['question'])
        res.append(row['path'])
        res.insert(0, row['condition'].lower())

        rows.append(res)

    if BASELINE:
        df = pd.DataFrame(rows, columns=["name", "zero_shot", "real", "question", "path"]) # Baseline
        df.to_csv(f"/home/cc/PHD/ragkg/results_quiz_baseline_v2_{model_name}.csv", index=False) # Baseline
    else:
        df = pd.DataFrame(rows, columns=["name", "zero_shot", "zero_shot_rag", "real", "question", "path"])
        df.to_csv(f"/home/cc/PHD/ragkg/results_quiz_v2_{model_name}.csv", index=False)

    print(f"Model {model_name} done!\n")