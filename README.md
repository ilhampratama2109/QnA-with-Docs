# 🦙 QnA with Doc - LLMA 3.2

This project is a document-based Q&A application that uses a large language model (LLM) from Ollama. The application allows users to upload a PDF file, ask questions related to its content, and receive direct answers. Built with **LangChain** and **Streamlit**, this project provides an interactive experience for extracting information from documents.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

## Overview

This application is designed to help users understand documents by asking questions related to an uploaded PDF's content. It leverages an LLM model from Ollama, combined with text-splitting techniques and vector embeddings powered by LangChain.

## Features

- **PDF Parsing**: Extracts text from uploaded PDF files.
- **Text Splitting**: Divides text into smaller segments for efficient processing.
- **Vector Embedding**: Creates vector representations of text for relevance-based searching.
- **Conversational Memory**: Maintains conversation history for a continuous chat experience.
- **Interactive Q&A Interface**: User-friendly interactive interface with Streamlit.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ilhampratama2109/QnA-with-Docs.git
   cd QnA-with-Docs
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload the PDF file you want to inquire about.
3. Type a question in the input field, and the Ollama LLM model will respond with relevant information.

## Project Structure

- `app.py` - Main script to run the Streamlit application.
- `requirements.txt` - List of dependencies needed for the project.
- `README.md` - Documentation for the project.

## Requirements

This project uses several libraries listed in `requirements.txt`:

- `PyPDF2`: For reading text from PDF files.
- `langchain` and `langchain_community`: For text splitting, embeddings, and LLMs.
- `faiss-cpu`: For vector-based document retrieval.
- `streamlit`: For the user interface.
