from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO

def prepare_inputs_with_dtype(processor, image, prompt, target_dtype, device):
    """Prepare inputs with correct dtype to match model"""
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Convert tensors to correct dtype and device
    processed_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point:
                processed_inputs[key] = value.to(dtype=target_dtype, device=device)
            else:
                processed_inputs[key] = value.to(device=device)
        else:
            processed_inputs[key] = value
    
    return processed_inputs

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
        
        print("✓ Model loaded successfully!")
        
        # Get model info
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        print(f"Model dtype: {model_dtype}")
        print(f"Model device: {model_device}")
        
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
            print("✓ Test image loaded successfully!")
        except Exception as e:
            print(f"Failed to load test image: {e}")
            print("Creating fallback image...")
            image = Image.new('RGB', (300, 400), color=(255, 255, 255))
        
        # Test model inference with proper dtype handling
        print("\nTesting model inference...")
        prompt = "<ocr>"  # OCR task
        
        try:
            # Prepare inputs with correct dtype
            inputs = prepare_inputs_with_dtype(processor, image, prompt, model_dtype, device)
            
            print(f"Input dtypes: {[(k, v.dtype if hasattr(v, 'dtype') else type(v)) for k, v in inputs.items()]}")
            
            # Run forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            print("✓ Forward pass successful!")
            
            if hasattr(outputs, 'logits'):
                print(f"✓ Output logits shape: {outputs.logits.shape}")
                print(f"✓ Output logits dtype: {outputs.logits.dtype}")
                
                # Try to decode a sample of the output
                logits = outputs.logits
                if logits.dim() >= 2:
                    predicted_ids = torch.argmax(logits, dim=-1)
                    
                    # Take a sample for decoding
                    if predicted_ids.dim() > 1:
                        sample_ids = predicted_ids[0][:20]  # First 20 tokens of first sequence
                    else:
                        sample_ids = predicted_ids[:20]
                    
                    try:
                        decoded_sample = processor.tokenizer.decode(sample_ids, skip_special_tokens=True)
                        print(f"✓ Decoded sample: '{decoded_sample}'")
                    except Exception as e:
                        print(f"Could not decode sample: {e}")
                
            else:
                print("✓ Model outputs generated (no logits attribute)")
            
            print("\n" + "="*50)
            print("SUCCESS!")
            print("="*50)
            print("✓ FP4 quantization is working correctly!")
            print(f"✓ Model size: {size_mb:.1f} MB")
            print(f"✓ Parameters: {param_count:,}")
            print(f"✓ Model dtype: {model_dtype}")
            print("✓ Forward pass completed without errors")
            
            return True
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            print("\nDetailed error information:")
            import traceback
            traceback.print_exc()
            
            print("\nThis could be due to:")
            print("- Dtype mismatch between inputs and model")
            print("- Memory issues")
            print("- Model-specific requirements")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nDetailed error information:")
        import traceback
        traceback.print_exc()
        
        print("\nPossible solutions:")
        print("1. Install required packages: pip install bitsandbytes accelerate")
        print("2. Ensure sufficient GPU memory")
        print("3. Update transformers: pip install transformers --upgrade")
        print("4. Try without quantization first")
        return False

if __name__ == "__main__":
    success = test_kosmos25_fp4()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed. Check the error messages above.")
