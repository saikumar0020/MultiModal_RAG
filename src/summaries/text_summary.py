from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.load_models import load_model

def get_text_summary(NarrativeText):
    try:
        text_summary = None
        model = load_model("deepseek-r1-distill-llama-70b")
        prompt_text="""You are an assistant tasked with summarizing text for retrieval. \
            These summaries will be embedded and used to retrieve the raw text elements. \
            Give a concise summary of the table or text that is well optimized for retrieval.text: {element} """

        prompt=ChatPromptTemplate.from_template(prompt_text) 

        summarize_chain = {"element": lambda x: x} |prompt| model | StrOutputParser()

        NarrativeText=NarrativeText[:10] # We can do chunking but for time being and explanation we are taking Subset.

        text_summary=[]

        text_summary=summarize_chain.batch(NarrativeText,{"max_concurrency": 5})
    except Exception as error:
        print("Error in get_text_summary --> ", error)
    return text_summary