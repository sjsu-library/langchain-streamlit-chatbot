from langchain_community.document_loaders import CSVLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "Your open AI key"

COLLECTIONNAME = 'XXX'
DBPATH = 'path/to/chromadb'
DATAPATH = 'path/to/yourdata'

###############################################################################################
# load data file
loader = DirectoryLoader(path=DATAPATH, glob='**/*.csv', show_progress=True, loader_cls=CSVLoader, loader_kwargs = {'source_column':'url', 'encoding':'utf-8'})
documents = loader.load()
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100, separators=["|", "\n", "(?<=. )", " ", ""])
docs = splitter.split_documents(documents)

################################################################################################
# save docs into chromaDB
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(collection_name=COLLECTIONNAME, documents=docs, embedding=embedding, persist_directory=DBPATH)

