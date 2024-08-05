from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model_id = "microsoft/Phi-3-mini-128k-instruct"
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

def get_model(model_path):
    global model, tokenizer, pipe
    
    model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    global pipe
    
    # Check if the request is JSON
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
        res = pipe(query, **generation_args)
        return jsonify({'summary': res[0]['generated_text']}), 200
    except Exception as err:
        return jsonify({'error': str(err)}), 500

if __name__ == '__main__':
    print('Loading model...')
    get_model(model_id)
    print('Loading model completed...')
    app.run(port=3000)
