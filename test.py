from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
from paperqa import Docs
import csv
import multiprocessing
import argparse

args = argparse.ArgumentParser()
args.add_argument('--metadata_csv', type=int, default=32)

# get a list of paths
def process_documents():

    docs = Docs()
    docs.add("/mnt/c/Users/jeffs/OneDrive/Desktop/2305.03726.pdf")
#     docs.add_url(
#        "https://arxiv.org/pdf/2305.03726.pdf",
#        citation="Otter: A Multi-Modal Model with In-Context Instruction Tuning, 2023, Accessed now",
#        dockey="test2",
#    )
    answer = docs.query("What is a transformer")
    print(answer)
process_documents()
