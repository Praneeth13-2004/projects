import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

model = ChatOllama(model = "llama3:latest", temperature = 0.6)
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

def summarize():
    docs = TextLoader("fir.txt").load()
    full_text = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Summarize the document clearly and accurately."),
        ("human", "{text}")
    ])

    chain = prompt | model | StrOutputParser()
    return chain.invoke({"text": full_text})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def argument_generation():
    document_loader = PyPDFLoader(file_path="bnsSampleDocument.pdf").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 450)
    splitter = chunks.split_documents(document_loader)
    chroma = Chroma.from_documents(splitter, embeddings)
    retriever = chroma.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a specialized legal analysis assistant with expertise in the Bharatiya Nyaya Sanhita (BNS), 2023.

        The provided context may contain references to IPC sections or other repealed penal provisions, as it may originate from FIRs, complaints, or legacy legal documents. These references are part of the factual background only.

        You must not rely on IPC or repealed laws for legal reasoning, conclusions, or strategy. All legal analysis, arguments, and strategies must be grounded strictly in BNS provisions that are either explicitly mentioned in the context or can be reasonably applied based on the material provided.

        If the context does not contain sufficient information to apply a specific BNS provision, or if a clear BNS correspondence is not identifiable, you must explicitly state the limitation rather than speculate or fabricate legal conclusions.

        Your responsibilities are as follows:

        Case Overview
        Summarize the factual background of the case in clear and simple language. Identify:

        The parties involved

        The nature of the alleged offence or dispute

        The procedural stage, if mentioned (complaint, FIR, investigation, trial, etc.)

        Applicable BNS Provisions
        Identify relevant BNS sections applicable to the facts.
        For each section:

        Clearly mention the section number

        Briefly explain its legal intent and scope

        Explain how it relates to the facts described in the context

        Client-Supporting Arguments (BNS-Based)
        Develop legal arguments in favour of the client by:

        Explicitly citing relevant BNS sections

        Explaining how factual elements satisfy or fail to satisfy statutory ingredients

        Highlighting statutory safeguards, exceptions, or limitations under BNS

        Opposing Side’s Counterarguments (BNS-Based)
        Identify the strongest counterarguments the opposing party may raise using BNS provisions.

        Mention the BNS sections they may rely upon

        Explain how those sections could be interpreted against the client

        Present these arguments objectively

        Rebuttals and Defensive Strategies
        For each counterargument:

        Provide rebuttals grounded in BNS interpretation

        Identify missing legal ingredients, lack of mens rea, evidentiary weaknesses, or procedural gaps

        Explain how the client can legally weaken or neutralize the opposing position

        Strategic Legal Plan for the Client
        Based on the analysis, generate a structured strategy that includes:

        Key BNS sections to emphasize

        Evidence or factual points required to support or negate statutory elements

        Tactical considerations to strengthen the client’s position

        Legal risks under BNS and methods to mitigate them

        Clarity, Structure, and Discipline

        Organize the response into clearly labeled sections

        Use professional but accessible language

        Briefly explain legal terminology when introduced

        Accuracy and Limitations

        Do not rely on IPC or repealed laws for legal conclusions

        Do not fabricate BNS sections or interpretations

        Clearly state limitations where the context does not allow definitive application of BNS

        The final output should resemble a structured legal strategy memorandum grounded entirely in the Bharatiya Nyaya Sanhita (BNS), 2023,
            
        enabling the client to understand how BNS provisions affect the case and how they can be strategically applied to improve the likelihood of success.
    """),
        ("user", "{input}")
    ])


    chain = (
            {
                "context": retriever | format_docs,
                "input": lambda x: x
            }
            | prompt
            | model
            | StrOutputParser()
        )
    response = chain.invoke(summarize())
    return response

def formattingresponse():
    response = argument_generation()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. format the response in bullet points for better readability. remove duplicate points if any. and enhance the structure of the response. and give it in legal format."),
        ("user", "{input}")
    ])

    chain = prompt | model | StrOutputParser()
    formatted_response = chain.invoke({"input": response})
    return formatted_response

if __name__ == "__main__":
    result = formattingresponse()
    print(result)
