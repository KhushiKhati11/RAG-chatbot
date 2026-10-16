# RAG-chatbot

A simple Retrieval Augmented Generation (RAG) chatbot that answers questions using information from PDF documents. It searches a local document database for relevant text and uses a language model to generate an answer based on that text.

## How It Works

1. PDF files are loaded from the data/ folder.
2. Each PDF is divided into smaller text sections.
3. The text sections are converted into embeddings and stored in a Chroma database.
4. When a question is asked, the chatbot searches the database for relevant text.
5. The retrieved text is sent to a language model to generate an answer.

## Project Files

- create_database.py : Loads PDF files, splits their text into sections, creates embeddings, and saves them in the Chroma database.
- query_data.py : Searches the database for relevant text and sends the retrieved context to a language model to answer a question.
- requirements.txt : Lists the Python packages required by the project.
- data/ : Folder where the source PDF documents should be placed.
- chroma/ : Folder where the local Chroma database is stored. This folder is generated when the database is created.

## Installation

1. Install the Python packages
From the project directory, run:
pip install -r requirements.txt
2. Prepare the PDF documents
Create a folder named data in the project directory and place the PDF files you want to search inside it. PDF files in subfolders are also supported.
3. Download the Ollama models
Make sure Ollama is running, then run these commands if the models are not already installed:
ollama pull nomic-embed-text
ollama pull phi3:mini

## Usage

Step 1: Create the database
Run this command from the project directory:
python create_database.py

This command reads the PDFs in data/, splits their text into sections, creates embeddings, and stores the results in chroma/.
Run this step again whenever you want to rebuild the database after changing the PDF documents. Note that the script deletes the existing chroma/ folder before creating a new database.

Step 2: Ask a question
Run the query script and put your question in quotation marks:
python query_data.py "What is the main topic of the document?"

Replace the example question with your own question.

## Answer Behaviour
- The chatbot retrieves up to eight relevant text sections from the database.
- If no results are found, or the top relevance score is below the configured threshold, it reports that it could not find relevant information.
- The prompt instructs the language model to answer using the retrieved context and provide a short supporting quote.
- The final response includes source information from the retrieved documents.

The model is instructed to respond with Not in context. when the answer is not explicitly stated in the supplied text.

## Limitations
- Answers depend on the content and quality of the PDF documents provided.
- The chatbot may not answer questions if the relevant information is not retrieved.
- The language model is instructed to use the retrieved context only, but its output should still be checked against the source text.