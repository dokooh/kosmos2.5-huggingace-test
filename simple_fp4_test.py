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
print("Loading Kosmos-2.5 with FP4 quantization...")
model = AutoModel.from_pretrained(
    "microsoft/kosmos-2.5",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2.5", trust_remote_code=True)

# Download test image
print("Downloading test image...")
image_url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Test OCR task
print("\n=== Testing OCR Task ===")
prompt = "<ocr>"
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate output
with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=256,
    )

# Decode and process the output
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
processed_text, _ = processor.post_process_generation(generated_text, cleanup_and_extract=False)
print("OCR Result:")
print(processed_text)

# Test grounding task
print("\n=== Testing Grounding Task ===")
prompt = "<grounding>An image of"
inputs = processor(text=prompt, images=image, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=256,
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
processed_text, entities = processor.post_process_generation(generated_text, cleanup_and_extract=True)
print("Grounding Result:")
print(f"Text: {processed_text}")
print(f"Entities: {entities}")

# Test Markdown OCR
print("\n=== Testing Markdown OCR Task ===")
prompt = "<md>"
inputs = processor(text=prompt, images=image, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=256,
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
processed_text, _ = processor.post_process_generation(generated_text, cleanup_and_extract=False)
print("Markdown OCR Result:")
print(processed_text)
