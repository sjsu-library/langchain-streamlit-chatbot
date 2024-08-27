import os, datetime
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.chat_history import BaseChatMessageHistory
import chromadb
from sqlalchemy.sql import text
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_feedback import streamlit_feedback


class InMemoryHistory(BaseChatMessageHistory):
    messages = []
    max_messages: int = 10  # Set the limit (K) of messages to keep

    def add_messages(self, messages) -> None:
        """Add a list of messages to the store, keeping only the last K messages."""
        self.messages.extend(messages)
        self.messages = self.messages[-self.max_messages:]
    
    def clear(self) -> None:
        self.messages = []


HIDEMENU = """
<style>
.stApp [data-testid="stHeader"] {
    display:none;
}

.stChatInput button{
    display:none;
}

#chat-with-sjsu-library-s-kingbot  a {
    display:none;
}
</style>
"""

@st.cache_data(max_entries=1000)
def getOpenAIKey():
    return st.secrets.openai.key

@st.cache_resource(ttl="1d")
def getDSConnection():
    return st.connection("mysqldb",autocommit=True)

@st.cache_resource(ttl="15d")
def getLLm():
    openai_api_key = getOpenAIKey()    
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0, streaming=True)
    return llm

@st.cache_resource(ttl="15d")
def getRetriever():
    llm = getLLm()
    openai_api_key = getOpenAIKey()
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    client = chromadb.HttpClient(host=st.secrets.vectordb.host, port=st.secrets.vectordb.port)
    dbremote = Chroma(
        client=client,
        collection_name="xxxx",
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )
    retriever=dbremote.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.5})
    myretriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    return myretriever


def getPrompt(custom_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", custom_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt    

@st.cache_resource(ttl="15d")
def getHARetriever():
    llm = getLLm()
    retriever = getRetriever()
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = getPrompt(contextualize_q_system_prompt)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever  

@st.cache_resource(ttl="15d")
def getRAGChain():
    llm = getLLm()    
    haRetriever = getHARetriever()
    ### Answer question ###
    system_prompt = (
        "You are a library AI assistant for question-answering tasks. Your name is Kingbot and you are a library AI assistant for the SJSU MLK Jr., Library. You respond in a supportive, professional, and reassuring manner like a peer-mentor. You do not generate creative content such as stories, poems, tweets, or code. You may generate relevant search terms if prompted by a user. You do not know any celebrities, influential politicians, activists or state heads. "
        "Use the following pieces of retrieved context to answer users' questions. "
        "If you don't know the answer, say that you don't know, don't try to make up an answer."
        "\n\n"
        "{context}"
    )
    qa_prompt = getPrompt(system_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(haRetriever, question_answer_chain)
    return rag_chain

def getBotengine(memory):          
    rag_chain = getRAGChain()    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )   
    return conversational_rag_chain

def streamHandler(response,msgs):    
    answer = response['answer']
    st.write(answer)
    if len(response['context']) > 0:
        if response['context'][0].metadata['source']:
            extra = "For more information, please check: " + response['context'][0].metadata['source']
            answer += '\r\n\r\n'
            answer += extra
            st.write(extra)         
    msgs.add_ai_message(answer)
                   

def printRetrievalHandler(container, response):
    status = container.status("**Context Retrieval**")
    query = response['input']
    history = response['chat_history']
    status.write(f"**Question:** {query}")
    status.write(f"**history:** {history}")    
    if len(response['context']) > 0:
        documents = response['context']
        status.update(label=f"**Context Retrieval:** {query}")
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            status.write(f"**Document {idx} from {source}**")
            status.markdown(doc.page_content)
    status.update(state="complete")

           

def saveFB(feedback, ts):
	try:
		conn = getDSConnection()
		score = ''
		comment = ''
		timest = ts.replace('T', ' ')
		if feedback['score']:
			thumbs = (feedback['score']).strip()
			score = {"👍": "Good", "👎": "Bad"}[thumbs]
		if feedback['text']:
			comment = (feedback['text']).strip()
		with conn.session as s:
			fb = score + "-" + comment
			params = {'fb':fb, 'timest':timest}
			s.execute(text("UPDATE tablename SET field=:fb WHERE condi=''"), params)
			s.commit()
	except Exception as e:
		st.error(e)


if __name__ == "__main__":    

    #Set up streamlit page
    st.set_page_config(page_title="Kingbot - SJSU Library", page_icon="🤖")
    st.markdown(HIDEMENU, unsafe_allow_html=True)
    st.title("Chat with SJSU Library's Kingbot")
    st.text("This experimental version of Kingbot uses Streamlit, LangChain, and ChatGPT.")

    #Get streamlit session 
    if 'session_id' not in st.session_state:
        session_id = get_script_run_ctx().session_id
        st.session_state.session_id = session_id
    session_id = st.session_state.session_id

    #Start real work
    msgs = StreamlitChatMessageHistory()
    
    # lastest 5 messeges kept in memory for bot prompt
    if 'memory' not in st.session_state: 
        memory = InMemoryHistory()
        st.session_state.memory = memory  
    memory = st.session_state.memory

    # get bot
    if 'mybot' not in st.session_state: 
        st.session_state.mybot = getBotengine(msgs)   
    bot = st.session_state.mybot


    feedback_kwargs = {
		"feedback_type": "thumbs",
		"optional_text_label": "Optional. Please provide extra information",
		"on_submit": saveFB,
	}
 
    # display chat history
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
            
    # chat
    if user_query := st.chat_input(placeholder="Ask me anything about the SJSU Library!"):
        msgs.add_user_message(user_query)
        current = datetime.datetime.now()
        st.session_state.moment = current.isoformat()
        answer = ''
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):           
            config = {"configurable": {"session_id": session_id}}
            with st.spinner(text="In progress..."):
                response = bot.invoke({"input": user_query}, config)
                answer = response['answer']
                printRetrievalHandler(st.container(),response)
                streamHandler(response,msgs)
            
            #Save QA to database
            try:
                conn = getDSConnection()
                with conn.session as s:
                    s.execute(
                        text('INSERT INTO tablename VALUES (:field1, :field2, :field3, :fb);'), 
                        params=dict(field1='', field1='', field1='',fb='')) 
                    s.commit()
            except Exception as e:
                st.error("Something went wrong. Please refresh your page.")
                # st.error(e)

    # feedback, works outside user_query section           
    if 'moment' in st.session_state:
        currents = st.session_state.moment
        streamlit_feedback(
            **feedback_kwargs, args=(currents,), key=currents,
        )



            