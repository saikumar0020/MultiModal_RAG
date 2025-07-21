import io
import re
from IPython.display import HTML, display
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from langchain_core.documents import Document


def is_image_data(b64data):
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False
    

def looks_like_base64(sb):
  return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def resize_base64_image(base64_string,size=(128,128)):
  # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """

    print("=========== From split_image_text_types =============")
    b64_images = []
    texts = []

    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)

    return {"images": b64_images, "texts": texts}


def display_base64_image(base64_str: str):
    """
    Decodes and displays a base64-encoded image using matplotlib.

    Parameters:
    - base64_str (str): Base64-encoded image string
    """
    try:
        # Decode the base64 string to binary image data
        image_data = base64.b64decode(base64_str)

        # Open the image using PIL
        image = Image.open(BytesIO(image_data))

        # Display the image using matplotlib
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    except Exception as e:
        print("Failed to decode and display image:", e)


def img_prompt_func(data_dict):
  """
  Join the context into a single string
  """
  print("=========== From img_prompt_func =============")

  formatted_texts = "\n".join(data_dict["context"]["texts"])
  messages = []

  # Adding image(s) to the messages if present
  if data_dict["context"]["images"]:
      for image in data_dict["context"]["images"]:
          display_base64_image(image)
          image_message = {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{image}"},
          }
          messages.append(image_message)

  # Adding the text for analysis
  text_message = {
      "type": "text",
      "text": (
          "You are a helpful assistant.\n"
          "You will be given a mixed info(s) .\n"
          "Use this information to provide relevant information to the user question. \n"
          f"User-provided question: {data_dict['question']}\n\n"
          "Text and / or tables:\n"
          f"{formatted_texts}"
      ),
  }
  messages.append(text_message)
  return [HumanMessage(content=messages)]

