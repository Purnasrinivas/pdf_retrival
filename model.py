
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
# import chainlit as cl
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
# from sympy.physics.units import temperature
os.environ["OPENAI_API_KEY"] = "sk-0G9RJ3QTR4myu0U9uTRx3LYreMhyDg7zkvN5FiN82_T3BlbkFJmIDFfZDmyACDDFTCs1ea_0jLwp8FqV-WoFoGlI9RUA"
db_faiss_path  = 'EMBEDDINGS_STORE_USING_LANGCHAIN/db_faiss'
data_path = 'data/' 
# custom_prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# #Retrieval QA Chain
# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=db.as_retriever(search_kwargs={'k': 2}),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={'prompt': prompt}
#                                        )
#     return qa_chain

# #Loading the model
# #def load_llm():
#     # Load the locally downloaded model here
#     #llm = CTransformers(
#       #  model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
#        # model_type="llama",
#        # max_new_tokens = 512,
#        # temperature = 0.5
#     #)
#   #  return llm
# llm1=ChatOpenAI(temperature=0.5)
# #QA Model Function
# def qa_bot():
#     embeddings = OpenAIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(db_faiss_path, embeddings,allow_dangerous_deserialization=True)
#     llm = llm1
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     return qa

# #output function
# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response

# #chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to  Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message):
#     chain = cl.user_session.get("chain") 
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = chain.invoke(message.content, callbacks=[cb])
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\nSources:" + str(sources)
#     else:
#         answer += "\nNo sources found"

#     await cl.Message(content=answer).send()

# def save_uploaded_file(uploaded_file):
#     file_path = os.path.join(data_path, uploaded_file.filename)
#     uploaded_file.save(file_path)
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost"}})

# @app.route('/upload', methods=['POST'])
# def upload():
#     files = request.files.getlist('files')
#     query = request.form.get('query')

#     # Save uploaded files
#     for file in files:
#         save_uploaded_file(file)

#    # Create vector database function
# def create_vector_db(data_path, db_faiss_path):
#     # Initialize an empty list to collect all documents
#     documents = []

#     # Iterate through files in the directory
#     for file_name in os.listdir(data_path):
#         file_path = os.path.join(data_path, file_name)

#         # Determine the file type and use the appropriate loader
#         if file_path.endswith('.pdf'):
#             documents.extend(PyPDFLoader(file_path).load())
#         elif file_path.endswith('.docx') or file_path.endswith('.doc'):
#             documents.extend(UnstructuredWordDocumentLoader(file_path).load())
#         elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
#             documents.extend(UnstructuredExcelLoader(file_path).load())
#         elif file_path.endswith('.csv'):
#             documents.extend(CSVLoader(file_path).load())
#         else:
#             print(f"Skipping unsupported file type: {file_name}")

#     # Split the documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     # Create embeddings using Hugging Face models
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

#     # Create a FAISS vector store from the text chunks
#     db = FAISS.from_documents(texts, embeddings)

#     # Save the FAISS database locally
#     db.save_local(db_faiss_path)
#     print("Vector database created and saved successfully.")

#     def set_custom_prompt():
#      """Prompt template for QA retrieval for each vectorstore
#     """
#     prompt = PromptTemplate(template=custom_prompt_template,
#                             input_variables=['context', 'question'])
#     return prompt

# #Retrieval QA Chain
# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=db.as_retriever(search_kwargs={'k': 2}),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={'prompt': prompt}
#                                        )
#     return qa_chain

# llm1=ChatOpenAI(temperature=0.5)
# #QA Model Function
# def qa_bot():
#     embeddings = OpenAIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(db_faiss_path, embeddings,allow_dangerous_deserialization=True)
#     llm = llm1
#     qa_prompt = set_custom_prompt()
#     qa = retrieval_qa_chain(llm, qa_prompt, db)

#     return qa

# #output function
# def final_result(query):
#     qa_result = qa_bot()
#     response = qa_result({'query': query})
#     return response
# return jsonify({'answer': answer, 'sources': sources})
    
# if __name__ == '__main__':
#     app.run(debug=True)



def save_uploaded_file(uploaded_file):
    file_path = os.path.join(data_path, uploaded_file.filename)
    try:
        uploaded_file.save(file_path)
        print(f"File saved at {file_path}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost"}})

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    query = request.form.get('query')

    # Save uploaded files
    for file in files:
        save_uploaded_file(file)

    # Create vector database function
    def create_vector_db(data_path, db_faiss_path):
        # Initialize an empty list to collect all documents
        documents = []

        # Iterate through files in the directory
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)

            # Determine the file type and use the appropriate loader
            if file_path.endswith('.pdf'):
                documents.extend(PyPDFLoader(file_path).load())
            elif file_path.endswith('.docx') or file_path.endswith('.doc'):
                documents.extend(UnstructuredWordDocumentLoader(file_path).load())
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                documents.extend(UnstructuredExcelLoader(file_path).load())
            elif file_path.endswith('.csv'):
                documents.extend(CSVLoader(file_path).load())
            else:
                print(f"Skipping unsupported file type: {file_name}")

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Create embeddings using Hugging Face models
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

        # Create a FAISS vector store from the text chunks
        db = FAISS.from_documents(texts, embeddings)

        # Save the FAISS database locally
        db.save_local(db_faiss_path)
        print("Vector database created and saved successfully.")

    def set_custom_prompt():
        """Prompt template for QA retrieval for each vectorstore"""
        prompt = PromptTemplate(template=custom_prompt_template,
                                input_variables=['context', 'question'])
        return prompt

    # Retrieval QA Chain
    def retrieval_qa_chain(llm, prompt, db):
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                               chain_type='stuff',
                                               retriever=db.as_retriever(search_kwargs={'k': 2}),
                                               return_source_documents=True,
                                               chain_type_kwargs={'prompt': prompt}
                                               )
        return qa_chain

    llm1 = ChatOpenAI(temperature=0.5)

    # QA Model Function
    def qa_bot():
        embeddings = OpenAIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
        db = FAISS.load_local(db_faiss_path, embeddings, allow_dangerous_deserialization=True)
        llm = llm1
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(llm, qa_prompt, db)
        return qa

    # Output function
    def final_result(query):
        qa_result = qa_bot()
        response = qa_result({'query': query})
        return response

    return jsonify({'answer': answer, 'sources': sources})

if __name__ == '__main__':
    app.run(debug=True)
