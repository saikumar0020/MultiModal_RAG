import uuid
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from utils.load_models import load_model

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
  print("========== From Retriever ===============")

  store=InMemoryStore()
  id_key="doc_id"

  retriever=MultiVectorRetriever(
      vectorstore=vectorstore,
      docstore=store,
      id_key=id_key,
  )

  def add_documents(retriever, doc_summaries, doc_contents):

      doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

      summary_docs = [
              Document(page_content=summary, metadata={id_key: doc_ids[i]})

              for i, summary in enumerate(doc_summaries)
          ]

      retriever.vectorstore.add_documents(summary_docs)
      retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


  if text_summaries:
        add_documents(retriever, text_summaries, texts)
  if table_summaries:
        add_documents(retriever, table_summaries, tables)
  if image_summaries:
        add_documents(retriever, image_summaries, images)

  return retriever


def get_retriever(text_summary, NarrativeText, 
                  table_summaries, Table, 
                  image_summaries, img_base64_list):
    try:
        retriever_multi_vector_img = None
        embedding_model=load_model("embedding")

        vectorstore=Chroma(collection_name="MMRAG",embedding_function=embedding_model)

        retriever_multi_vector_img = create_multi_vector_retriever(
            vectorstore,
            text_summary,
            NarrativeText,
            table_summaries,
            Table,
            image_summaries,
            img_base64_list,
            )
    except Exception as error:
         print("Error in get_retriever --> ", error)
    return retriever_multi_vector_img


