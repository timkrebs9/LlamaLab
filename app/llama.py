# llama.py

import os
import json
from typing import List
from tqdm import tqdm
import PyPDF2
from openai import OpenAI

from app.settings import settings
# Initialize the OpenAI client
client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key=settings.OPENAI_API_KEY
)

# Template for generating JSON
template = {
    "question": " ",
    "answer": " "
}

# ==========================================================
# ETL (Extract, Transform, Load) - PDF Text Extraction
# ==========================================================

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.
    
    :param file_path: Path to the PDF file
    :return: Extracted text as a string
    """
    text = ''
    with open(file_path, 'rb') as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        for page_num in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[page_num]
            text += page_obj.extract_text()
    return text


# ==========================================================
# JSON Handling - Fixing and Validating JSON
# ==========================================================

def fix_json(crptd_json: str) -> dict:
    """
    Attempts to fix corrupted JSON using the OpenAI model.
    
    :param crptd_json: Corrupted JSON string
    :return: Properly formatted JSON
    """
    messages = [
        {'role': 'system', 'content': f'You are an API that converts wrongly formatted JSON into a properly formatted one using this template: {template}. Only respond with the JSON and no additional text.'},
        {'role': 'user', 'content': 'Wrong JSON: ' + crptd_json}
    ]

    response = client.chat.completions.create(
        model=settings.LLM_TYPE,
        messages=messages,
        max_tokens=settings.LLM_MAX_RESPONSE_TOKENS,
        n=1,
        stop=None,
        temperature=settings.LLM_TEMPERATURE,
    )

    response_text = response.choices[0].message.content.strip()

    try:
        json_data = json.loads(response_text)
        print(json_data)
        return json_data
    except json.JSONDecodeError:
        print("The JSON is not valid, reformatting again.")
        return []


# ==========================================================
# LLM Interactions - Generate Questions and Answers
# ==========================================================

def generate_questions_answers(text_chunk: str) -> dict:
    """
    Generates a question and answer pair from a chunk of text using the Llama 3.1 model.
    
    :param text_chunk: A string chunk of text
    :return: JSON object containing a question and answer
    """
    messages = [
        {'role': 'system', 'content': 'You are an API that converts bodies of text into a single question and answer in a JSON format. Each JSON should contain a single question with a single answer. Only respond with the JSON and no additional text.'},
        {'role': 'user', 'content': 'Text: ' + text_chunk}
    ]

    response = client.chat.completions.create(
        model=settings.LLM_TYPE,
        messages=messages,
        max_tokens=settings.LLM_MAX_RESPONSE_TOKENS,
        n=1,
        stop=None,
        temperature=settings.LLM_TEMPERATURE+0.2,
    )

    response_text = response.choices[0].message.content.strip()

    try:
        json_data = json.loads(response_text)
        if settings.DEBUG:
                print(json_data)
        return json_data
    except json.JSONDecodeError:
        if settings.DEBUG:
            print("Error: Response is not valid JSON.... Trying to fix the JSON.")
        print("Error: Response is not valid JSON.... Trying to fix the JSON.")
        return []


# ==========================================================
# Text Chunk Processing
# ==========================================================

def process_text(text: str, chunk_size: int = 4000) -> List[dict]:
    """
    Processes large texts by dividing them into chunks and generating question-answer pairs.
    
    :param text: The full text to process
    :param chunk_size: The size of each text chunk
    :return: A list of dictionaries containing questions and answers
    """
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_responses = []

    for chunk in tqdm(text_chunks, desc="Processing chunks", unit="chunk"):
        response = generate_questions_answers(chunk)
        if 'question' in response and 'answer' in response:
            all_responses.append({'question': response['question'], 'answer': response['answer']})

    return all_responses


# ==========================================================
# Processing Multiple PDFs in a Folder
# ==========================================================

def process_pdfs_in_folder(folder_path: str) -> dict:
    """
    Processes all PDF files in a specified folder, extracts text, and generates question-answer pairs.
    
    :param folder_path: Path to the folder containing PDF files
    :return: Dictionary containing responses for all PDF files
    """
    all_responses = {"responses": []}
    
    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            
            # Step 1: Extract text from the current PDF
            pdf_text = extract_text_from_pdf(file_path)
            
            # Step 2: Process the extracted text into question-answer pairs
            file_responses = process_text(pdf_text)
            
            # Append responses to the overall list
            all_responses["responses"].extend(file_responses)
    
    return all_responses


# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":
    # User inputs the folder containing PDF files
    folder_path = input("Enter the folder path containing PDF files: ")

    # Step 1: Process all PDFs in the specified folder
    responses = process_pdfs_in_folder(folder_path)

    # Step 2: Save all responses to a JSON file
    output_file = 'dataset/responses.json'
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"Processing complete. Responses saved to {output_file}")