from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
from paperqa import DocsPineCone, Docs
import csv
import multiprocessing
import argparse

# get a list of paths
def process_documents():
    # this set of parquet files should be in s3 and MUST be loaded
    parquet_files = '/Users/mandytoh/docs_index_dets.parquet'
    # There are two indexes 
    # 1. paperqa-index (big, large, confusing, difficult, sad output)
    # 2. paperqa-openai-check (smaller, easier to search)
    docs = DocsPineCone(text_index_name = "paperqa-openai-check", parquet_file = parquet_files )

    # how to add docs in vanilla paperqa

    # docs.add_url(
    #     "/Users/mandytoh/Projects/lux_google/paper-qa/paperqa/2305.03726.pdf",
    #     citation="Otter: A Multi-Modal Model with In-Context Instruction Tuning, 2023, Accessed now",
    #     dockey="test2",
    # )

    # docs.add(
    #     "/Users/mandytoh/Projects/lux_google/paper-qa/2305.03726.pdf",
    #     citation="Otter: A Multi-Modal Model with In-Context Instruction Tuning, 2023, Accessed now",
    #     dockey="test2",
    # )

    #query sets up the search + retrieval + llm
    answer = docs.query("What manufacturing challenges are unique to bispecific antibodies?")
    print(answer)
    answer2 = docs.query("What is a transformer?")
    print(answer2)
process_documents()


#     docs.add_url(
#         "https://arxiv.org/pdf/2304.08485.pdf",
#         citation=" Visual instruction tuning., 2023, Accessed now",
#         dockey="test",
#     )

# def process_documents(row):
#     # Do something with the row
#     # ...

#     # Return the processed result
#     return processed_result

# # Function to handle each chunk of rows
# def process_chunk(chunk):
#     results = []
#     for row in chunk:
#         result = process_documents(row)
#         results.append(result)
#     return results

# def main(args):
#     # Define the number of processes to use
#     num_processes = multiprocessing.cpu_count()

#     # Open the CSV file and read rows
#     with open(args.metadata_csv, 'r') as file:
#         reader = csv.reader(file)
#         rows = list(reader)

#     # Split rows into chunks for parallel processing
#     chunk_size = len(rows) // num_processes
#     chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

#     # Create a pool of processes
#     pool = multiprocessing.Pool(processes=num_processes)

#     # Process chunks in parallel
#     results = pool.map(process_chunk, chunks)

#     # Close the pool of processes
#     pool.close()
#     pool.join()

#     # Flatten the results into a single list if needed
#     flattened_results = [result for chunk_result in results for result in chunk_result]

# if __name__ == "__main__":
#     main()
