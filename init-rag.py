from os.path import normpath, basename
import argparse

from classes.vector_store import VectorStore

parser = argparse.ArgumentParser(description="Init RAG")
parser.add_argument("-kgbase", type=ascii, default="data/kgbase", help="Path contain the data to index in the kg base")
parser.add_argument("-embedder", type=ascii, default="mxbai-embed-large", help="Model to embed chunk")
parser.add_argument("-chunk_size", type=int, default=500, help="Chunk dimension")
parser.add_argument("-overlap", type=int, default=150, help="Overlap tokens between chunks")
args = parser.parse_args()

# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore(f"indexes/{basename(normpath(args.kgbase))}", args.embedder, args.chunk_size, args.overlap)

vector_store.add_documents(args.kgbase)