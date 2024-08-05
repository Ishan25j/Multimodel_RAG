FROM huggingface/transformers-pytorch-gpu

# Install the required packages
RUN python3 -c "from transformers import LlavaForConditionalGeneration; LlavaForConditionalGeneration.from_pretrained('xtuner/llava-phi-3-mini-hf')"
RUN python3 -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('xtuner/llava-phi-3-mini-hf')"

RUN pip install -U flask pillow

# Copy the code
COPY . /app

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 3000

# Run the application
CMD ["python", "image.py"]