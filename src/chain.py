import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from src.image_processing.image_data_processing import split_image_text_types, img_prompt_func
from src.dataingestion.ingestion_retriever import get_retriever
from utils.load_models import load_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.dataextraction.pdf_extraction import categorize_pdf_elements, get_raw_data_from_pdf
from src.summaries.text_summary import get_text_summary
from src.summaries.table_summary import get_table_summary
from src.summaries.image_summary import get_image_details

from config import base_dir

def multi_modal_rag_chain(retriever, model):
  chain=(
      {"context":retriever | RunnableLambda(split_image_text_types),
       "question": RunnablePassthrough()
      }
      | RunnableLambda(img_prompt_func)

      |model
      |StrOutputParser()
  )

  return chain


def get_response(query):
    image_model = load_model("gemini-1.5-flash")

    # base_path = r"C:\Personal Storage\Workspace\Projects\Multimodal_RAG"
    base_path = base_dir
    fpath = os.path.join(base_path, "extracted_data")
    cache_file = os.path.join(base_path, "image_details", 
                              "img_sumaries_base64.pkl")


    raw_pdf_elements = get_raw_data_from_pdf(base_path)

    categorized = categorize_pdf_elements(raw_pdf_elements)

    NarrativeText = categorized["NarrativeText"]
    Table = categorized["Table"]
    

    text_summary = get_text_summary(NarrativeText)

    table_summaries = get_table_summary(Table)

    img_base64_list, image_summaries = get_image_details(fpath, cache_file)

    retriever_multi_vector_img = get_retriever(text_summary, NarrativeText, 
                                                table_summaries, Table, 
                                                image_summaries, img_base64_list)
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img, image_model)

    response = chain_multimodal_rag.invoke(query)
    return response