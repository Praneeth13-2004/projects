# Projects
this repository contains the projects that i did on my own 
### the projects are 
  #### 1. Emotion Recognition
        The emotion recognition project contains the code in python to recognize the
        facial expressions of a human.

        This project uses the packages such as 
        1. opencv
          The cv2 module from OpenCV is used to capture input from the webcam, display the 
          processed output, and write text or visual information directly onto the output frame.
        2. deepface
           DeepFace module from deepface is a lightweight Python framework used for facial recognition and facial
           attribute analysis, such as age, gender, emotion, and race estimation. It simplifies
           complex computer vision tasks by providing a high-level interface over multiple state-of-the-art
           deep learning models, allowing developers to implement facial analysis features with minimal code.
           

  #### 2. Waste Recognition System
        The Waste recognition system project contains the code in python to
        recognize the waste objects through webcam

        This project uses the packages such as
        1. opencv
           The cv2 module from OpenCV is used to capture input from the webcam, display the 
          processed output, and write text or visual information directly onto the output frame.
        2. torch
            PyTorch is the core deep-learning library. It handles tensors (large numerical arrays),
            runs neural networks, performs all mathematical operations, and manages CPU/GPU execution.
            Any model inference or training, including CLIP, fundamentally depends on torch.
        3. torchvision
            Torchvision is PyTorch’s computer-vision companion. It provides image datasets,
            pretrained vision models, and common image transformations. Even when not used directly, 
            many vision pipelines and models rely on it for image handling utilities.
        4. transformers
            Transformers is the Hugging Face library that provides ready-made state-of-the-art models
            like CLIP, BERT, and GPT. It handles model loading, tokenization, and preprocessing so you
            can use powerful AI models without implementing them from scratch.
        5. pillow
            Pillow is an image processing library that works with image objects (PIL images).
            It is mainly used to convert images between formats, especially when moving data 
            from OpenCV (NumPy arrays) to models like CLIP that expect PIL images in RGB format.

            
  #### 3. RAG Based Legal Document Analysis 
         This project is useful to analyse the legal documents that is provided to it such as FIR's
         and gives the strategies that the client side lawyers are supposed to follow

         This project uses the packages such as
        1. streamlit
            Streamlit is used to build simple web-based user interfaces directly in Python.
            It helps display outputs, manage user interaction, and maintain chat history in 
            an interactive app format without writing HTML or JavaScript.
        2. langchain-community
           LangChain Community provides integrations for chat models, document loaders, and 
           vector stores. In this project, it enables interaction with Ollama-based language
           models, loading text and PDF documents, managing chat history, and storing embeddings
           in a vector database.
        3. langchain-core
           LangChain Core contains the fundamental building blocks such as prompt templates, 
           output parsers, and runnable chains. It is responsible for structuring prompts,
           controlling how the model receives instructions, and converting the model’s responses 
           into usable text.
        4. langchain-text-splitters
           This package is used to divide large documents into smaller overlapping chunks.
           Splitting is necessary to process long legal documents efficiently while maintaining
           enough context for accurate retrieval and reasoning.
        5. langchain-ollama
           LangChain Ollama provides support for embedding models and chat models that run locally 
           through Ollama. It is used to generate embeddings and connect the application to locally
           hosted large language models.
        6. chromadb
           ChromaDB is the vector database used to store and retrieve document embeddings.
           It enables similarity-based search so that only the most relevant document sections 
           are used during legal analysis.
        7. ollama
           Ollama is the local model runtime that serves large language models and embedding models.
           It allows the application to run LLaMA and embedding models locally without relying on external 
           APIs.
        8. llama3.3: latest
           LLaMA 3.3 is a locally hosted large language model served through Ollama.
           In this project, it is used as the core reasoning engine to understand documents,
           generate summaries, perform legal analysis under the Bharatiya Nyaya Sanhita (BNS), 
           and produce structured, human-readable responses. The latest tag ensures the most 
           recent optimized version of the model is used for better accuracy, reasoning quality, 
           and performance during inference.

           
#### author: Raavinuthala Sai Praneeth Karthikeya
