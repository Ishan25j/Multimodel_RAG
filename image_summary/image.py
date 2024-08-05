from flask import Flask, request, jsonify
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import numpy as np
from PIL import Image
import io

model_id = "xtuner/llava-phi-3-mini-hf"

UPLOAD_FOLDER = 'uploads'
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/generate', methods=['POST'])
def generate():
    global model, processor
    
    # Check if the request is JSON or form-data
    if request.is_json:
        data = request.json
        Title = data.get('title', '')
        img_text = data.get('text', '')
        # For JSON requests, we expect the image to be base64 encoded
        if 'query_image' not in data:
            return jsonify({'error': 'No image data in the request'}), 400
        image_data = data['query_image']
        image = Image.open(io.BytesIO(image_data.encode('utf-8')))
    else:
        # Handle form-data
        Title = request.form.get('Content Title', '')
        img_text = request.form.get('img_text', '')
        if 'query_image' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        query_image = request.files['query_image']
        if query_image.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if query_image and allowed_file(query_image.filename):
            image = Image.open(query_image)
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    # Process the image
    image = np.array(image).astype(np.float32)
    question = f"""<|user|>\n<image>You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval. Use extra from the following text from the image.
    Context Title: {Title}\n \
    Text: {img_text if img_text else 'No text found in the image'} Note: two time @ is just for indexing without any meaning @@<|end|>\n<|assistant|>\n"""    
    try:
        inputs = processor(question, image, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        res = processor.decode(output[0][2:], skip_special_tokens=True)
        return jsonify({'summary': res}), 200
    except Exception as err:
        return jsonify({'error': str(err)}), 500

if __name__ == '__main__':
    print('Loading model...')
    get_model(model_id)
    print('Loading model completed...')
    app.run(port=3000)
