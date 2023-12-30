# Project Name: InsightFlow

## Overview

InsightFlow is a web application that leverages the power of open-source language models to create vector embeddings for YouTube videos and PDF files. The project utilizes the Llama2-7b language model and LangChain to process text data, generate embeddings, and enable users to ask questions based on the video transcript or uploaded PDF content.

## Features

- **Embedding Generation:** Utilize the Llama2-7b language model to create vector embeddings for text data extracted from YouTube videos or PDF files.

- **Support for Various Sources:** Accept URLs of YouTube videos or upload PDF files to extract relevant textual content.

- **Streamlit-based Interface:** Provide a user-friendly web interface using Streamlit for easy interaction with the application.

- **Question-Answering Module:** Enable users to ask questions related to the content of the video transcript or the uploaded PDF file.

## Prerequisites

Make sure you have the following dependencies installed:

- Python 3.6+
- Streamlit
- LangChain
- Llama2-7b language model
- Other required libraries (specified in requirements.txt)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AbdullahTabassam/Lama2-7b-Langchain.git
   cd Lama2-7b-Langchain

2. Create a virtual enviornment (Recommended).

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

4. Download and set up the Llama2-7b language model. Refer to the official documentation for instructions.

## Usage

1. Run the Streamlit app:
    
    ```bash
    streamlit run app.py

2. Access the web application in your browser at http://localhost:8501.

3. Enter the URL of a YouTube video or upload a PDF file.

4. Use the question-answering module to ask questions related to the content.

5. Click the "Submit" button to process the text data, create vector embeddings and get an answer form the the model. 

## Contributing

All contributions to enhance the application are welcome.

## Acknowledgments

1. The Llama2-7b language model and LangChain for providing powerful natural language processing capabilities.
2. Streamlit for simplifying the development of interactive web applications.

## Contact

For issues, questions, or feedback, please contact abdullahdar2017@gmail.com
