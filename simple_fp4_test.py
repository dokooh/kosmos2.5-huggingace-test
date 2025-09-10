"""
Debug script to understand Kosmos-2.5 model structure and test dtype compatibility
"""

from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

def test_dtype_compatibility():
    """Test different dtype configurations"""
    print("Testing Kosmos-2.5 Model Structure and Dtype Compatibility")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test different configurations
    configs = [
        {"name": "FP32 Default", "kwargs": {}},
        {"name": "FP16", "kwargs": {"torch_dtype": torch.float16}},
        {"name": "BF16", "kwargs": {"torch_dtype": torch.bfloat16}} if torch.cuda.is_bf16_supported() else None,
    ]
    
    # Remove None configs
    configs = [c for c in configs if c is not None]
    
    for config in configs:
        print(f"\n{'-'*50}")
        print(f"Testing: {config['name']}")
        print(f"{'-'*50}")
        
        try:
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if device == "cuda" else None,
                **config["kwargs"]
            }
            
            print(f"Loading with kwargs: {model_kwargs}")
            model = AutoModel.from_pretrained(
                "microsoft/kosmos-2.5",
                **model_kwargs
            )
            
            processor = AutoProcessor.from_pretrained(
                "microsoft/kosmos-2.5",
                trust_remote_code=True
            )
            
            print("✓ Model loaded successfully!")
            print(f"Model type: {type(model)}")
            print(f"Model class: {model.__class__.__name__}")
            
            # Get model info
            model_dtype = next(model.parameters()).dtype
            model_device = next(model.parameters()).device
            print(f"Model dtype: {model_dtype}")
            print(f"Model device: {model_device}")
            
            # Test basic forward pass with dtype matching
            print("\nTesting forward pass...")
            
            # Create test image and inputs
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
            prompt = "<ocr>"
            
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            print(f"Original input dtypes: {[(k, v.dtype if hasattr(v, 'dtype') else type(v)) for k, v in inputs.items()]}")
            
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
            
            print(f"Processed input dtypes: {[(k, v.dtype if hasattr(v, 'dtype') else type(v)) for k, v in processed_inputs.items()]}")
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**processed_inputs)
            
            print("✓ Forward pass successful!")
            print(f"Output type: {type(outputs)}")
            
            if hasattr(outputs, 'logits'):
                print(f"✓ Logits shape: {outputs.logits.shape}")
                print(f"✓ Logits dtype: {outputs.logits.dtype}")
            
            # Clean up
            del model
            del processor
            if device == "cuda":
                torch.cuda.empty_cache()
            
            print(f"✓ {config['name']} test completed successfully!")
            
        except Exception as e:
            print(f"✗ Error in {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            
            if device == "cuda":
                torch.cuda.empty_cache()

def check_environment():
    """Check the environment setup"""
    print("\n" + "="*60)
    print("ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check for required packages
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed")
    
    try:
        import bitsandbytes
        print(f"BitsAndBytes available: Yes")
    except ImportError:
        print("BitsAndBytes not installed")
    
    try:
        import accelerate
        print(f"Accelerate available: Yes")
    except ImportError:
        print("Accelerate not installed")

if __name__ == "__main__":
    check_environment()
    test_dtype_compatibility()
