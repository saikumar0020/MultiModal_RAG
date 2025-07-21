import os
import pickle

def get_raw_data_from_pdf(base_path):
    try:
        raw_pdf_elements = None
        pickle_file_path = os.path.join(base_path, "raw_data_extracted", "raw_pdf_elements.pkl")
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, "rb") as f:
                raw_pdf_elements = pickle.load(f)
            print("Loaded raw_pdf_elements from cache.")
        else:
        
            print("file not found")

            from unstructured.partition.pdf import partition_pdf

            raw_pdf_elements=partition_pdf(
                filename="/content/RAG-For-NLP.pdf",            # Path of the PDF file
                strategy="hi_res",                              # Use high-resolution strategy for better layout parsing
                extract_images_in_pdf=True,                     # Enable extraction of images embedded in the PDF
                extract_image_block_types=["Image","Table"],    # Specify which block types to extract (images and tables)
                extract_image_block_to_payload=False,           # Do not include extracted image blocks as payloads in the elements
                extract_image_block_output_dir="extracted_data" # Directory to store extracted image and table blocks
                )
            with open("raw_data_extracted/raw_pdf_elements.pkl", "wb") as f:
                pickle.dump(raw_pdf_elements, f)
            print("PDF processed and cached.")


    except Exception as error:
        print("Error in get_raw_data_from_pdf --> ", error)
    return raw_pdf_elements

def categorize_pdf_elements(raw_pdf_elements):
    """
    Categorizes elements from a PDF into different types.

    Parameters:
        raw_pdf_elements (List): List of elements extracted from a PDF using partition_pdf

    Returns:
        dict: A dictionary with keys as element types and values as lists of stringified content
    """
    # Initialize empty lists
    categorized = {
        "Header": [],
        "Footer": [],
        "Title": [],
        "NarrativeText": [],
        "Text": [],
        "ListItem": [],
        "Image": [],
        "Table": []
    }

    # Iterate and categorize
    for element in raw_pdf_elements:
        element_type = str(type(element))
        if "Header" in element_type:
            categorized["Header"].append(str(element))
        elif "Footer" in element_type:
            categorized["Footer"].append(str(element))
        elif "Title" in element_type:
            categorized["Title"].append(str(element))
        elif "NarrativeText" in element_type:
            categorized["NarrativeText"].append(str(element))
        elif "Text" in element_type:
            categorized["Text"].append(str(element))
        elif "ListItem" in element_type:
            categorized["ListItem"].append(str(element))
        elif "Image" in element_type:
            categorized["Image"].append(str(element))
        elif "Table" in element_type:
            categorized["Table"].append(str(element))

    return categorized

if __name__ == "__main__":
    get_raw_data_from_pdf("")