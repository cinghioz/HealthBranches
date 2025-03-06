from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_ollama import OllamaEmbeddings
import faiss
import os
import pickle

from typing import List

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

    def _load_documents(self, doc_path: str, doc_type: str = "*.txt") -> List[Document]:
        loader = DirectoryLoader(doc_path, glob=doc_type)
        documents = loader.load()
        return documents

    def _split_text(self, documents: List[Document]):
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