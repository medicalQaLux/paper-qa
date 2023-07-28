from flask import Flask, request
from dotenv import load_dotenv
import os
import pinecone

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
app = Flask(__name__)

from paperqa import DocsPineCone  # must come after the .env loads

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
pinecone.list_indexes()
index = pinecone.Index("paperqa-index")
parquet_file_path = 'docs_index_dets.parquet'
paperqa_engine = DocsPineCone(text_index_name = "paperqa-index", parquet_file = parquet_file_path)
small_paperqa_engine = DocsPineCone(text_index_name = "paperqa-openai-check", parquet_file = parquet_file_path)

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    req = request.get_json()
    chat_log = req['chat_log']
    if req['index_size'] == 'small':
        engine = small_paperqa_engine
    else:
        engine = paperqa_engine
    last_question = chat_log[-1]['user'] if chat_log else ''
    
    a = engine.query(last_question)
    return a.answer, 200

if __name__ == "__main__":
    app.run(debug=True, port=5001)
