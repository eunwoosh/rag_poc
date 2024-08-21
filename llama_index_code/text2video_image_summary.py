import os
from pathlib import Path

import qdrant_client
import ollama
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

from llama_index.core import VectorStoreIndex


llm = Ollama(model="llava", request_timeout=60.0)

def img_parse(img_path: Path, output_path: Path):
    res = ollama.chat(
        model="llava",
        messages=[
            {
                'role': 'user',
                'content': 'Describe this image:',
                'images': [str(img_path)]
            }
        ]
    )

    with (output_path / img_path.stem).open("a") as write_file:
        write_file.write("---"*10 + "\n\n")
        write_file.write(os.path.basename(img_path) + "\n\n")
        write_file.write(res['message']['content'])
        write_file.flush()
    print("Proceeding ", img_path)

    
def get_nodes():
    nodes = []
    summary_dir = Path("llava_summary")
    num = 0
    for file_path in summary_dir.iterdir():
        with file_path.open("r") as f:
            lines = f.readlines()

        nodes.append(TextNode(text=lines[4], metadata={"image_path" : lines[2]}, id=num))
        num += 1

    return nodes


def get_documents():
    nodes = []
    summary_dir = Path("llava_summary")
    num = 0
    for file_path in summary_dir.iterdir():
        with file_path.open("r") as f:
            lines = f.readlines()

        nodes.append(Document(text=lines[4], metadata={"image_path" : lines[2]}, id=num))
        num += 1

    return nodes


def main():
    embed_model = HuggingFaceEmbedding()
    #     model_name="BAAI/bge-small-en-v1.5"
    # )
    
    # Create a local Qdrant vector store
    client = qdrant_client.QdrantClient(path="qdrant_img_db")
    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    storage_context = StorageContext.from_defaults(vector_store=text_store)
    # index = VectorStoreIndex(get_nodes(), embed_model=embed_model, storage_context=storage_context)
    
    index = VectorStoreIndex.from_documents(get_documents(), embed_model=embed_model, storage_context=storage_context)



    # generate Text retrieval results
    retriever_engine = index.as_retriever(similarity_top_k=20)
    # retrieve more information from the GPT4V response
    retrieval_results = retriever_engine.retrieve("scene including vehicle")
    retrieved_images = []
    frame_dir = Path("op_trailer/frames")
    for res in retrieval_results:
        retrieved_images.append(frame_dir / res.metadata["image_path"].split("/")[0])

    for file_path in retrieved_images:
        print(file_path)


if __name__ == "__main__":
    main()
