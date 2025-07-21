from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.load_models import load_model

def get_table_summary(Table):
    try:
        table_summaries = None
        model = load_model("deepseek-r1-distill-llama-70b")
        prompt_text = """You are an AI Assistant tasked with summarizing tables for retrieval. \
            These summaries will be embedded and used to retrieve the raw table elements. \
            Give a concise summary of the table that is well optimized for retrieval. Table:{element} """

        prompt = ChatPromptTemplate.from_template(prompt_text) 

        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        table_summaries = []
        table_summaries = summarize_chain.batch(Table, {"max_concurrency": 5})
    except Exception as error:
        print("Error in get_table_summary --> ", error)
    return table_summaries