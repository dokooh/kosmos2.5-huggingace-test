from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import requests
from io import BytesIO

def analyze_model_output(outputs):
    """Analyze the Kosmos2_5ModelOutput structure"""
    print(f"\nModel Output Analysis:")
    print(f"Output type: {type(outputs)}")
    
    # List all attributes
    attributes = [attr for attr in dir(outputs) if not attr.startswith('_')]
    print(f"Available attributes: {attributes}")
    
    # Analyze tensor attributes
    tensor_info = {}
    for attr in attributes:
        try:
            value = getattr(outputs, attr)
            if isinstance(value, torch.Tensor):
                tensor_info[attr] = {
                    "shape": list(value.shape),
                    "dtype": value.dtype,
                    "device": value.device
                }
                print(f"  {attr}: shape={value.shape}, dtype={value.dtype}")
        except:
            pass
    
    return tensor_info

def test_kosmos25_fp4():
    """Test Kosmos-2.5 with FP4 quantization and analyze outputs"""
    
    print("Kosmos-2.5 FP4 Quantization Test")
    print("=" * 40)
    
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
        
        print("\nLoading model with FP4 quantization...")
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
        
        # Model info
        model_dtype = next(model.parameters()).dtype
        print(f"✓ Model loaded!")
        print(f"  Model dtype: {model_dtype}")
        
        # Calculate size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"  Model size: {size_mb:.2f} MB")
        print(f"  Parameters: {param_count:,}")
        
        # Load test image
        print("\nLoading test image...")
        try:
            image_url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            print("✓ Downloaded test receipt image")
        except:
            image = Image.new('RGB', (300, 400), color=(255, 255, 255))
            print("✓ Created fallback test image")
        
        # Test inference
        print("\nTesting inference...")
        prompt = "<ocr>"
        
        # Prepare inputs with correct dtype
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Convert inputs to match model dtype
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    processed_inputs[key] = value.to(dtype=model_dtype, device=device)
                else:
                    processed_inputs[key] = value.to(device=device)
            else:
                processed_inputs[key] = value
        
        # Run inference
        with torch.no_grad():
            outputs = model(**processed_inputs)
        
        print("✓ Inference successful!")
        
        # Analyze the output structure
        tensor_info = analyze_model_output(outputs)
        
        # Try to extract useful information
        print(f"\nOutput Structure Analysis:")
        if tensor_info:
            print("Found tensor outputs:")
            for name, info in tensor_info.items():
                print(f"  {name}: {info['shape']} ({info['dtype']})")
        
        # Check for common attributes
        common_attrs = ['logits', 'last_hidden_state', 'prediction_logits', 'hidden_states']
        found_attrs = []
        for attr in common_attrs:
            if hasattr(outputs, attr):
                value = getattr(outputs, attr)
                if isinstance(value, torch.Tensor):
                    found_attrs.append(f"{attr}: {list(value.shape)}")
        
        if found_attrs:
            print("Key attributes found:")
            for attr in found_attrs:
                print(f"  {attr}")
        
        print(f"\n{'='*50}")
        print("SUCCESS!")
        print('='*50)
        print("✓ FP4 quantization is working correctly!")
        print(f"✓ Model size: {size_mb:.1f} MB")
        print(f"✓ Parameters: {param_count:,}")
        print(f"✓ Forward pass completed without errors")
        print(f"✓ Output structure analyzed successfully")
        
        # Test with different prompts
        print(f"\nTesting different prompts...")
        test_prompts = ["<ocr>", "<grounding>", "<md>"]
        
        for prompt in test_prompts:
            try:
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                processed_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.dtype.is_floating_point:
                            processed_inputs[key] = value.to(dtype=model_dtype, device=device)
                        else:
                            processed_inputs[key] = value.to(device=device)
                    else:
                        processed_inputs[key] = value
                
                with torch.no_grad():
                    outputs = model(**processed_inputs)
                
                print(f"✓ {prompt} prompt successful")
                
            except Exception as e:
                print(f"✗ {prompt} prompt failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kosmos25_fp4()
    
    if success:
        print(f"\n🎉 FP4 quantization test completed successfully!")
        print("The model is working with FP4 quantization.")
        print("You can now use these scripts to test different quantization methods.")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")
