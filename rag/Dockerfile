FROM huggingface/transformers-pytorch-gpu

# Install the required packages
RUN python3 -c "from transformers import LlavaForConditionalGeneration; LlavaForConditionalGeneration.from_pretrained('xtuner/llava-phi-3-mini-hf')"
RUN python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('xtuner/llava-phi-3-mini-hf')"

RUN pip install -U flask pillow langchain sentencetransformers

# Copy the code
COPY . /app

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "image.py"]