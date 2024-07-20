import pickle
import hashlib
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import os


def load_embedding_mode(params):
    # 加载embedding模型
    model_path = params["model_path"]
    encode_kwargs = params["encode_kwargs"]
    model_kwargs = params["model_kwargs"]
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def load_llm_mode(llm_path):
    # 加载大语言模型
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(llm_path, trust_remote_code=True).quantize(4).half().cuda()
    model = model.eval()
    return model, tokenizer


def store_chroma(docs, embeddings, file_path):
    # 存储/加载embeddings
    key = pickle.dumps((file_path,))
    hash_key = hashlib.sha1(key).hexdigest()

    persist_directory = os.path.join("VectorStore", str(hash_key))
    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)
        db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
        db.persist()
    else:
        db = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    return db


def augment_prompt(db, query: str, k=2):
    # 获取提示词
    similar_docs = db.similarity_search(query, k=k)

    info = ""
    for idx, doc in enumerate(similar_docs):
        info += f"{idx + 1}. {doc.page_content}\n"

    return info