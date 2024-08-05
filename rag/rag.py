import base64
import os
import re
import io
from langchain.vectorstores import Chroma
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import uuid
from flask import Flask, request, jsonify
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import numpy as np

model_id = "xtuner/llava-phi-3-mini-hf"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model(model_path):
    global model, processor
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_path)


# Define a custom embedding function using HuggingFace's SentenceTransformer
class HuggingFaceEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever


# Initialize the vectorstore with HuggingFace embeddings

app = Flask(__name__)


@app.route('/rag', methods=['POST'])
def generate():
    global vectorstore, retriever_multi_vector_img
    
    # Check if the request is JSON or form-data
    if request.is_json:
        data = request.json
        text_summaries = data.get('text_summaries', '')
        Text = data.get('Text', '')
        tab_summ = data.get('tab_summ', '')
        tab = data.get('tab', '')
        imag_summ = data.get('imag_summ', '')
        image_encode_base64 = data.get('image_encode_base64', '')
        # For JSON requests, we expect the image to be base64 encoded
        if 'query_image' not in data:
            return jsonify({'error': 'No image data in the request'}), 400

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        Text,
        tab_summ,
        tab,
        imag_summ,
        image_encode_base64,
    )

def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
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

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
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
    b64_images = []
    texts = []
    
    docs = [docs[0]]
    
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

def process_llava_input(input_dict, query):
    context = input_dict["texts"]
    images = input_dict["images"]
    
    # Convert base64 images to PIL Images
    pil_images = [Image.open(io.BytesIO(base64.b64decode(img))).convert('RGB') for img in images]
    
    # Convert PIL Images to numpy arrays
    np_images = [np.array(img).astype(np.float32) for img in pil_images]
    
    # create query with  context
    prompt = f"""<|user|>\n<image>You are an assistant tasked to answer the query based on input image and query. \ 
        If there is no relevant image or if answer is not found then say there is no relevant text or image in the docs. \
        Give a concise summary of the given input image and context: {context} \
        Query: {query} Note: two time @ is just for indexing without any meaning @@<|end|>\n<|assistant|>\n"""
    
    return prompt, np_images

def query_gen(query):
    try:
        global retriever_multi_vector_img, model, processor
        ret = retriever_multi_vector_img.invoke(query)
        ret = [ret[0]]
        split = split_image_text_types(ret)
        processed_input = process_llava_input(split, query)
        inputs = processor(processed_input[0], processed_input[1], return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        out = processor.decode(output[0], skip_special_tokens=True)
        return out.split("@@")[1]
    except Exception as err:
        return str(err)
    
@app.route('/query', methods=['POST'])
def query():
    # Check if the request is JSON or form-data
    if request.is_json:
        data = request.json
        query = data.get('query', '')
        if query == '':
            return jsonify({'error': 'Query is missing'}), 400
    else:
        # Handle form-data
        query = request.form.get('query', '')
        if query == '':
            return jsonify({'error': 'Query is missing'}), 400

    # Process the query
    try:
        res = query_gen(query)
        return jsonify({'summary': res}), 200
    except Exception as err:
        return jsonify({'error': str(err)}), 500

if __name__ == '__main__':
    print('Loading model...')
    get_model(model_id)
    global vectorstore
    vectorstore = Chroma(
        collection_name="mm_rag", 
        embedding_function=HuggingFaceEmbeddings()
    )
    print('Loading model completed...')
    app.run(port=3000)