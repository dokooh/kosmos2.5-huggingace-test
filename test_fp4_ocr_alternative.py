"""
Test script for quantized models - checks which methods actually work
"""

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import os
import time

def test_quantized_model(model_path, method_name):
    """Test a single quantized model"""
    print(f"\nTesting {method_name}...")
    
    try:
        # Load model
        start_time = time.time()
        model = AutoModel.from_pretrained(
            model_path,
            device_map="auto", 
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        load_time = time.time() - start_time
        
        # Create test image
        image = Image.new('RGB', (300, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((20, 20), "Test OCR Text", fill=(0, 0, 0), font=font)
        draw.text((20, 50), "Line 2: Numbers 123", fill=(0, 0, 0), font=font)
        draw.text((20, 80), "Line 3: Email test@example.com", fill=(0, 0, 0), font=font)
        
        # Test inference
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        
        inputs = processor(text="<ocr>", images=image, return_tensors="pt")
        
        # Convert inputs to correct dtype/device
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    processed_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                else:
                    processed_inputs[key] = value.to(device=model_device)
            else:
                processed_inputs[key] = value
        
        # Run inference
        inference_start = time.time()
        with torch.no_grad():
            outputs = model(**processed_inputs)
        inference_time = time.time() - inference_start
        
        print(f"  ✓ {method_name}")
        print(f"    Load time: {load_time:.2f}s")
        print(f"    Inference time: {inference_time:.3f}s")
        print(f"    Model dtype: {model_dtype}")
        print(f"    Output type: {type(outputs)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ {method_name} failed: {e}")
        return False

def main():
    quantized_dir = "./quantized_models"
    
    if not os.path.exists(quantized_dir):
        print(f"Quantized models directory not found: {quantized_dir}")
        print("Run alternative_quantization.py first!")
        return
    
    # Find all quantized models
    methods = []
    for item in os.listdir(quantized_dir):
        model_path = os.path.join(quantized_dir, item)
        if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            methods.append((model_path, item))
    
    if not methods:
        print("No quantized models found!")
        return
    
    print("🧪 Testing Quantized Models")
    print("=" * 40)
    
    successful_methods = []
    
    for model_path, method_name in methods:
        if test_quantized_model(model_path, method_name):
            successful_methods.append(method_name)
    
    print(f"\n📊 Summary:")
    print(f"  Total methods: {len(methods)}")
    print(f"  Successful: {len(successful_methods)}")
    print(f"  Success rate: {len(successful_methods)/len(methods)*100:.1f}%")
    
    if successful_methods:
        print(f"\n✅ Working methods:")
        for method in successful_methods:
            print(f"    - {method}")
    else:
        print(f"\n❌ No methods worked - check your setup")

if __name__ == "__main__":
    main()
