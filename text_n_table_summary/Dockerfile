FROM huggingface/transformers-pytorch-gpu

# Install the required packages
RUN python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-128k-instruct')"
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-128k-instruct')"

RUN pip install -U flask pillow langchain sentencetransformers

# Copy the code
COPY . /app

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 4000

# Run the application
CMD ["python", "image.py"]