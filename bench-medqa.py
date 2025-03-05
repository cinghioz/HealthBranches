# %%
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
import pickle
import random
import pandas as pd
import re
import json
import ast
import glob

from typing import Dict, List
import torch
from prompt import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
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
            chunk_size=1000,
            chunk_overlap=200,
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

# %%
# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore('/home/cc/PHD/ragkg/indexes/kgbase-new/')

# Add documents in vector store (comment this line after the first add)
# vector_store.add_documents('/home/cc/PHD/ragkg/data/kgbase-new')

# %%
class LLMinference:
    def __init__(self, llm_name):
        self.llm_name = llm_name
        self.model = OllamaLLM(model=llm_name) 

    def _transform_query(self, query: str) -> str:
        return f'Represent this sentence for searching relevant passages: {query}'

    def single_inference(self, query: str, template: str, path: str,  choices: List[str], cond: str,  context) -> str | List[str]:
        context_text = "\n\n---\n\n".join([doc.page_content for doc in context])

        prompt_template = ChatPromptTemplate.from_template(template)
        if path != "":
            prompt = prompt_template.format(context=context_text, question=query, path=path, condition=cond, 
                                            o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
        else:
            prompt = prompt_template.format(context=context_text, question=query, condition=cond, o1=choices[0], 
                                            o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])

        response_text = self.model.invoke(prompt)
        response_text = response_text.strip().replace("\n", "").replace("  ", "")

        sources = [doc.metadata.get("source", None) for doc in context]
        
        return response_text, sources

    def qea_evaluation(self, query: str, template: str, path: str, choices: List[str], cond: str,  vector_store):

        results = vector_store.search(query=query, k=5)

        response, sources = self.single_inference(query, template, path, choices, cond, results)

        return response

# %%

folder_path = "/home/cc/PHD/ragkg/MedQA"
# models = ["mistral", "llama3.1:8b", "llama2:7b", "medllama2:7b", "gemma:7b", "gemma2:9b", "phi4:14b", "qwen2.5:7b", "mixtral:8x7b", "deepseek-r1:7b"]
models = ["mistral"]
# templates = [PROMPT_TEMPLATE, PROMPT_TEMPLATE_ONE, PROMPT_TEMPLATE_RAG]
templates = [PROMPT_MED, PROMPT_MED_RAG]

med_files = glob.glob(f"{folder_path}/*/top*", recursive=True)

# %%
cnt_rag = 0
cnt = 0

rows = []

for model_name in models:
    llm = LLMinference(llm_name=model_name)

    cnt = 0
    rows = []

    # for jso in tqdm(med_files):
    for jso in med_files:
        questions = pd.read_json(jso, lines=True)

        for index, row in questions.iterrows():
            res = []

            cond = jso.split('/')[-2].lower()

            for template in templates:
                res.append(llm.qea_evaluation(row['question'], template, "", list(row['options'].values()), cond, vector_store))
                # res.append(llm.qea_evaluation(row['question'], template, row['reasoning_trace'], ast.literal_eval(row['choices']), row['name'].lower(), vector_store)) # Baseline

            res.append(row["answer_idx"])
            res.append(row['question'])
            res.insert(0, cond)

            rows.append(res)
        
        cnt += 1
        print(f"process: {cnt}/{len(med_files)}")

    df = pd.DataFrame(rows, columns=["name", "zero_shot", "zero_shot_rag", "real", "question"]) # medqa
    df.to_csv(f"/home/cc/PHD/ragkg/results_medqax5_{model_name}.csv", index=False)

    print(f"{model_name} model processed!")

def extract_option(answer):
    """Extract the chosen option (A, B, C, D or E) from the LLM's answer."""
    match = re.search(r'\b([A-Ea-e])\b', str(answer))
    return match.group(1).upper() if match else None

def evaluate_answers(file_path):
    df = pd.read_csv(file_path)

    # option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    # df['correct_answer'] = df['real'].map(option_map)
    
    # result_columns = ['question', 'correct_answer']
    result_columns = ['question', 'real']
    
    # Evaluate all columns that start with "zero_shot" or "one_shot"
    for col in [c for c in df.columns if c.startswith(("zero_shot", "one_shot"))]:
        df[f'{col}_choice'] = df[col].apply(extract_option)
        # df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['correct_answer']
        df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['real']
        accuracy = df[f'{col}_is_correct'].mean()
        print(f'Accuracy for {col}: {accuracy:.2%}')
        result_columns.extend([f'{col}_choice', f'{col}_is_correct'])
    
    return df[result_columns]

evaluated_df = evaluate_answers("/home/cc/PHD/ragkg/results_medqax5_mistral.csv")
evaluated_df.head(5)


