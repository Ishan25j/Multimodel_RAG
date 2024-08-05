# Multimodal Retrieval-Augmented Generation (RAG) Project

This project implements a multimodal Retrieval-Augmented Generation (RAG) system that leverages multi-vector retrieval to handle various data types including text, images, and more. The system integrates several advanced technologies including Langchain, HuggingFace models, Flask, Docker, and Chroma as the vector database.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

## Overview

The multimodal RAG system is designed to perform seamless question-answering across diverse data types, including images and text. It uses multiple HuggingFace models for different tasks:

- `all-MiniLM-L6-v2` for text embeddings.
- `xtuner/llava-phi-3-mini-hf` for image summarization and image retrieval-based QA.
- `microsoft/Phi-3-mini-128k-instruct` for text summarization.

The system is built using Langchain for orchestrating the various components, Flask for the API, Docker for containerization, and Chroma as the vector database.

## Features

- **Multi-Vector Retrieval**: Efficiently retrieves relevant data using multi-vector similarity search.
- **Multimodal Query Handling**: Supports queries involving both text and images.
- **Advanced Summarization**: Utilizes state-of-the-art models for summarizing text and images.
- **Containerized Deployment**: Easily deployable using Docker.
- **Scalable Vector Database**: Uses Chroma for fast and scalable vector searches.

## Architecture

![image.png](https://blog.langchain.dev/content/images/size/w1000/2023/10/image-16.png)

The architecture of the system is as follows:

1. **Data Ingestion**: Text and images are ingested and stored in the Chroma vector database.
2. **Query Processing**: Queries are processed to determine the type (text or image) and relevant embeddings are generated.
3. **Retrieval**: Relevant data is retrieved from the vector database using multi-vector similarity search.
4. **Response Generation**: The retrieved data is used to generate a response using the appropriate HuggingFace model.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
