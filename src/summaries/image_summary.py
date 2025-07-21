import base64
import os
import pickle
from langchain_core.messages import HumanMessage
from utils.load_models import load_model

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_summarize(img_base64, prompt, chat):
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path, chat):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """
    img_base64_list = []
    image_summaries = []
    prompt = (
        "You are an assistant tasked with summarizing images for retrieval. "
        "These summaries will be embedded and used to retrieve the raw image. "
        "Give a concise summary of the image that is well optimized for retrieval."
    )
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt, chat))
    return img_base64_list, image_summaries

def get_image_details(fpath, cache_file):
    img_base64_list =None
    image_summaries = None
    # Load model
    image_model = load_model("gemini-1.5-flash")
    # Set image directory and cache file path
    # Check cache
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print("Loaded from cache.")
        img_base64_list = data["img_base64"]
        image_summaries = data["summaries"]
    else:
        # Generate base64 image list and summaries
        img_base64_list, image_summaries = generate_img_summaries(fpath, image_model)
        # Save all into a cache file
        with open(cache_file, "wb") as f:
            pickle.dump({
                "summaries": image_summaries,
                "img_base64": img_base64_list
            }, f)
        print("Processed and cached image summaries and base64 images.")
    
    return img_base64_list, image_summaries

