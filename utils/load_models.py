import os

from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

def load_model(model_name):
    # if model_name == "gemini-pro":
    #     return ChatGoogleGenerativeAI(model="gemini-1.5-pro") # To deal with text, due to issue in accessing model. I'm using another model via Groq.
    if model_name == "deepseek-r1-distill-llama-70b":
        return ChatGroq(model="deepseek-r1-distill-llama-70b")
    elif model_name == "gemini-1.5-flash":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash") # To deal with images
    elif model_name == "embedding":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001") # For embeddings
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    model = load_model("deepseek-r1-distill-llama-70b")
    # print(model.invoke("Hello"))

    