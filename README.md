# projects
this repository contains the projects that i did on my own 
### the projects are 
  #### 1. emotion recognition
        The emotion recognition project contains the code in python to 
        recognize the facial expressions of a human.

        This project uses the packages such as 
        1. cv2 from opencv
           This can be used to take the input, display the output and 
           to write the output on output frame
        2. DeepFace from deepface
           The DeepFace package is a lightweight Python framework used
           for facial recognition and facial attribute analysis 
           (such as age, gender, emotion, and race estimation). 
           It simplifies complex computer vision tasks by wrapping several
           state-of-the-art deep learning models, allowing developers to
           implement these functionalities with minimal code. 
           

  #### 2. Waste Recognition system
        The Waste recognition system project contains the code in python to
        recognize the waste objects through webcam

        This project uses the packages such as
        1. cv2 from opencv
           This can be used to take the input, display the output and 
           to write the output on output frame
        2. Torch
            PyTorch is the core deep-learning library. It handles tensors (large numerical arrays),
            runs neural networks, performs all mathematical operations, and manages CPU/GPU execution.
            Any model inference or training, including CLIP, fundamentally depends on torch.
        3. Torchvision
            Torchvision is PyTorch’s computer-vision companion. It provides image datasets,
            pretrained vision models, and common image transformations. Even when not used directly, 
            many vision pipelines and models rely on it for image handling utilities.
        4. Transformers
            Transformers is the Hugging Face library that provides ready-made state-of-the-art models
            like CLIP, BERT, and GPT. It handles model loading, tokenization, and preprocessing so you
            can use powerful AI models without implementing them from scratch.
        5. Pillow
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
        2. langchain_community.chat_models (ChatOllama)
            ChatOllama is a LangChain wrapper that allows the application to interact with 
            locally running large language models through Ollama. It is responsible for generating
            responses, reasoning over prompts, and producing legal analysis using models like LLaMA 3.
        3. langchain_community.chat_message_histories (StreamlitChatMessageHistory)
           This module manages chat history inside Streamlit applications.
           It stores previous user–assistant messages so the model can maintain
           conversational context across multiple interactions.
        4. Transformers
            Transformers is the Hugging Face library that provides ready-made state-of-the-art models
            like CLIP, BERT, and GPT. It handles model loading, tokenization, and preprocessing so you
            can use powerful AI models without implementing them from scratch.
        5. Pillow
            Pillow is an image processing library that works with image objects (PIL images).
            It is mainly used to convert images between formats, especially when moving data 
            from OpenCV (NumPy arrays) to models like CLIP that expect PIL images in RGB format.       
#### author: Raavinuthala Sai Praneeth Karthikeya
