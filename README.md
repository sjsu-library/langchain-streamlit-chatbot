# Build a chatbot powered by Langchain, Chroma, and Streamlit


## Requirments:
Install python, Langchain, Chroma, and Streamlit.


## Setup:
1. Vector database
2. Chatbot
3. Mysql for response stats (optional)


## Tested:
REDhat 8, python 3.9, Lanchain 0.1, Chroma 0.4.24, Streamlit 1.36


## Steps:
1. Vector database
- Downlaod Chroma, Langchain. 
- Put the data file (here is in csv format, could be other format) at DATAPATH.
- Run saveChromadb.py.
- Get the chroma file from DBPATH.
- Use podman to launch the Chroma server with the chroma file:

  podman run -d --rm --name [chromadbname] -v [/pathto/your/chromafile] -e IS_PERSISTENT=TRUE  chromadb/chroma:latest


2. Chatbot
- Downlaod Lanchain, Streamlit.
- Streamlit run chatbot.py.


3. Mysql for response stats (optional)
   
   If you would like to save users' response or feedback, set up mysql access.

