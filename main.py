from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import os

app = Flask(__name__)

def load_db():
    # Load documents
    loader = PyPDFLoader("PM_Narendra_Modi_US_Congress_speech.pdf")
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    docs = text_splitter.split_documents(documents)

    if not docs:  # Check if docs list is empty
        return None

    # Define embedding
    embeddings = OpenAIEmbeddings()

    # Create vector database from data
    persist_directory = './chroma'

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return db

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="refine",
        retriever=retriever,
        return_source_documents=True,
    )
    return conversation_chain

def handle_userinput(user_question):
    conversation = app.session['conversation']
    chat_history = app.session.get('chat_history')

    if chat_history is None:
        chat_history = []

    inputs = {
        'question': user_question,
        'chat_history': chat_history
    }

    if conversation:
        response = conversation(inputs)
        answer = response.get('answer')
        response.pop('source_documents', None)  # Remove 'source_documents' key

        # Update the chat history
        app.session['chat_history'] = response.get('chat_history')
        
        return answer



@app.route('/api', methods=['POST'])
def api_endpoint():
    load_dotenv()
    
    # Setup the connection with OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None or openai_api_key == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # Load the documents and create the vector database
    db = load_db()

    # Initialize the conversation chain
    conversation_chain = get_conversation_chain(db)

    # Initialization
    app.session = {
        'conversation': conversation_chain,
        'chat_history': None
    }

    data = request.get_json()

    user_question = data['user_question']
    if user_question:
        response = handle_userinput(user_question)

    result = {
        'message': 'Data processed successfully',
        'response': response
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run()
