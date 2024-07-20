from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


def load_Text(directory):
    # txt文件的加载和切分
    loader = TextLoader(directory, autodetect_encoding=True)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50)
    split_docs = text_spliter.split_documents(documents)
    return split_docs


def load_Pdf(directory):
    # pdf文件的加载和切分
    loader = PyPDFLoader(directory)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(pages)
    return docs


def load_Docx(directory):
    # doc或docx文件的加载和切分
    loader = Docx2txtLoader(directory)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(pages)
    return docs


def load_Docs(directory):
    # 根据路径选择加载和划分方式
    if directory.endswith(".txt"):
        print("txt")
        documents = load_Text(directory)
    elif directory.endswith(".pdf"):
        print("pdf")
        documents = load_Pdf(directory)
    elif directory.endswith(".docx") or directory.endswith(".doc"):
        print("docx")
        documents = load_Docx(directory)
    else:
        documents = []

    return documents

