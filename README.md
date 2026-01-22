# Project Repository

This repository contains projects that I have developed independently to explore computer vision, deep learning, and natural language processing applications.

## 1. Emotion Recognition

The Emotion Recognition project contains Python code to recognize human facial expressions using a webcam.

Packages Used

OpenCV (cv2)
The cv2 module from OpenCV is used to capture input from the webcam, display the processed output, and write text or visual information directly onto the output frame.

DeepFace
DeepFace is a lightweight Python framework used for facial recognition and facial attribute analysis such as age, gender, emotion, and race estimation. It simplifies complex computer vision tasks by providing a high-level interface over multiple state-of-the-art deep learning models, allowing facial analysis to be implemented with minimal code.

## 2. Waste Recognition System

The Waste Recognition System project contains Python code to recognize waste objects in real time using a webcam.

Packages Used

OpenCV (cv2)
The cv2 module from OpenCV is used to capture input from the webcam, display the processed output, and write text or visual information directly onto the output frame.

Torch
PyTorch is the core deep-learning library used in this project. It handles tensor operations, runs neural networks, performs mathematical computations, and manages CPU/GPU execution. All model inference in this project depends on PyTorch.

Torchvision
Torchvision is PyTorch’s computer vision companion. It provides pretrained vision models, datasets, and common image transformations that support vision-based deep learning workflows.

Transformers
Transformers is the Hugging Face library that provides access to state-of-the-art models such as CLIP, BERT, and GPT. It handles model loading, preprocessing, and tokenization, enabling advanced AI capabilities without building models from scratch.

Pillow
Pillow is an image processing library used to handle image objects. It is mainly used to convert images between formats, especially when transferring frames from OpenCV (NumPy arrays) to models like CLIP that expect PIL images in RGB format.

## 3. RAG-Based Legal Document Analysis

This project analyzes legal documents such as FIRs and generates strategic legal insights to assist client-side lawyers. It uses a Retrieval-Augmented Generation (RAG) approach to ground responses in relevant legal documents.

Packages Used

Streamlit
Streamlit is used to build a simple web-based user interface in Python. It enables interactive display of outputs, user input handling, and readable presentation of results without requiring frontend development.

LangChain Community
LangChain Community provides integrations for chat models, document loaders, and vector stores. In this project, it is used to interact with Ollama-based language models, load text and PDF documents, manage chat history, and store embeddings.

LangChain Core
LangChain Core contains the foundational components such as prompt templates, output parsers, and runnable chains. It controls prompt structure and ensures the model’s responses are converted into usable text.

LangChain Text Splitters
This package is used to split large legal documents into smaller overlapping chunks. Chunking is essential for efficient retrieval while preserving enough context for accurate reasoning.

LangChain Ollama
LangChain Ollama provides support for chat and embedding models that run locally via Ollama. It is used to generate embeddings and connect the application to locally hosted language models.

ChromaDB
ChromaDB is a vector database used to store and retrieve document embeddings. It enables similarity-based search so that only the most relevant document sections are used during legal analysis.

Ollama
Ollama is the local runtime used to serve large language models and embedding models. It allows all inference to run locally without relying on external APIs.

LLaMA 3.3 (latest)
LLaMA 3.3 is a locally hosted large language model served through Ollama. In this project, it acts as the core reasoning engine to understand legal documents, generate summaries, perform analysis under the Bharatiya Nyaya Sanhita (BNS), and produce structured, human-readable legal strategies. The latest tag ensures the most recent optimized version of the model is used.

## Author

Raavinuthala Sai Praneeth Karthikeya
