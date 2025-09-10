from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO

# Configure FP4 quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4"  # Use FP4 quantization
)

# Load Kosmos-2.5 with FP4 quantization
model = AutoModel.from_pretrained(
    "microsoft/kosmos-2.5",
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5")

# Download test image
image_url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Prepare input
prompt = "<ocr>"  # OCR task
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        num_beams=1,
    )

# Decode result
result = processor.decode(outputs[0], skip_special_tokens=True)
print("OCR Result:")
print(result)

# For grounding tasks, you can use:
# prompt = "<grounding>An image of"
# This will generate bounding boxes for objects

# For MD-OCR tasks:
# prompt = "<md>"
# This will generate markdown-formatted OCR output
