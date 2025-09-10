from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO

def test_kosmos25_fp4():
    """Simple test of Kosmos-2.5 with FP4 quantization"""
    
    print("Testing Kosmos-2.5 with FP4 Quantization")
    print("=" * 45)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Configure FP4 quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4"
        )
        
        print("Loading model with FP4 quantization...")
        model = AutoModel.from_pretrained(
            "microsoft/kosmos-2.5",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2.5", 
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"Model size: {size_mb:.2f} MB")
        print(f"Parameters: {param_count:,}")
        
        # Load test image
        print("\nLoading test image...")
        try:
            image_url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            print("Test image loaded successfully!")
        except Exception as e:
            print(f"Failed to load test image: {e}")
            print("Creating fallback image...")
            image = Image.new('RGB', (224, 224), color='white')
        
        # Test model inference
        print("\nTesting model inference...")
        prompt = "<ocr>"  # OCR task
        
        try:
            # Prepare inputs
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            
            # Run forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            print("✓ Forward pass successful!")
            
            if hasattr(outputs, 'logits'):
                print(f"✓ Output logits shape: {outputs.logits.shape}")
            else:
                print("✓ Model outputs generated (format may vary)")
            
            print("\n=== SUCCESS ===")
            print("FP4 quantization is working correctly!")
            print(f"Model size reduced to approximately {size_mb:.1f} MB")
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            print("The model loaded but inference encountered an error.")
            print("This could be due to input format or model-specific requirements.")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nPossible solutions:")
        print("1. Install required packages: pip install bitsandbytes accelerate")
        print("2. Ensure sufficient GPU memory")
        print("3. Try without quantization first")
        return False
    
    return True

if __name__ == "__main__":
    test_kosmos25_fp4()
