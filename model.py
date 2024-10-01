import os
import openai
from langchain_community.llms import OpenAI
from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from werkzeug.utils import secure_filename
from flask_cors import CORS
from langchain_openai import ChatOpenAI

# Ensure OpenAI API key is set
openai_api_key1= ''
if not openai_api_key1:
    raise ValueError("OpenAI API key is missing. Please set the 'OPENAI_API_KEY' environment variable.")
EMBEDDINGS_STORE_FOLDER="./EMBEDDINGS_STORE_USING_LANGCHAIN/db_faiss"
# Initialize OpenAI Embeddings with the correct model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key1, model="text-embedding-ada-002")
vector_store = None

# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080"]}})

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx'}



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def save_vector_store():
    global vector_store
    if vector_store:
        if not os.path.exists(EMBEDDINGS_STORE_FOLDER):
            os.makedirs(EMBEDDINGS_STORE_FOLDER)
        vector_store.save_local(EMBEDDINGS_STORE_FOLDER)
        print(f"Vector store saved to {os.path.abspath(EMBEDDINGS_STORE_FOLDER)}")
        # Log saved files in the folder
        saved_files = os.listdir(EMBEDDINGS_STORE_FOLDER)
        print(f"Files saved in {EMBEDDINGS_STORE_FOLDER}: {saved_files}")

@app.route('/')
def index():
    return render_template('model.html') 

@app.route('/upload', methods=['POST'])
def upload_files():
    global vector_store

    if 'files' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    files = request.files.getlist('files')
    documents = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.docx'):
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif filename.endswith('.xlsx'):
                    loader = UnstructuredExcelLoader(file_path)
                else:
                    return jsonify({'error': f'Unsupported file type: {filename}'}), 400

                documents.extend(loader.load())
            except Exception as e:
                return jsonify({'error': f'Error processing file {filename}: {str(e)}'}), 500
        else:
            return jsonify({'error': f'File {file.filename} has an invalid extension'}), 400

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Use OpenAI Embeddings to embed the documents
    try:
        vector_store = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        return jsonify({'error': f'Error creating embeddings: {str(e)}'}), 500
    save_vector_store()

    return jsonify({'message': 'Files uploaded and processed successfully'}), 200


@app.route('/query', methods=['POST'])
def query():
    global vector_store

    if vector_store is None:
        return jsonify({'error': 'No files uploaded yet'}), 400

    query_text = request.form.get('query', None)
    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key1)

        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )

        # Call the invoke method with the correct input structure
        response = qa_chain.invoke({"query": query_text})

        # Inspect the response to determine how to access the answer
        print("Response from QA Chain:", response)

        # Assuming the answer is in response["result"], adjust if necessary
        answer = response.get("result", "I don't know")  # Default if "result" is not present

        return jsonify({'answer': answer}), 200  # Return answer in JSON format
    except Exception as e:
        print(f"Error processing query: {str(e)}")  # Print the error to the terminal
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

if __name__ == '__main__':
    
    app.run(debug=True, port=8080)
