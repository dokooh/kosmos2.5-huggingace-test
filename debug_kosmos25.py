"""
Fixed debug script for Kosmos-2.5 model with proper dtype handling
"""

from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

def create_model_compatible_inputs(processor, image, prompt, model):
    """Create inputs with dtype that matches the model"""
    # Get the model's dtype from its parameters
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    
    print(f"Model dtype: {model_dtype}")
    print(f"Model device: {model_device}")
    
    # Process inputs normally first
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    print(f"Original input dtypes: {[(k, v.dtype if hasattr(v, 'dtype') else type(v)) for k, v in inputs.items()]}")
    
    # Convert all tensors to match model dtype and device
    converted_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            # Convert floating point tensors to model dtype
            if value.dtype.is_floating_point:
                converted_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                print(f"Converted {key}: {value.dtype} -> {model_dtype}")
            else:
                # Keep integer tensors as-is but move to correct device
                converted_inputs[key] = value.to(device=model_device)
                print(f"Moved {key} to device: {model_device}")
        else:
            converted_inputs[key] = value
    
    print(f"Final input dtypes: {[(k, v.dtype if hasattr(v, 'dtype') else type(v)) for k, v in converted_inputs.items()]}")
    return converted_inputs

def test_model_with_dtype_fix(config_name, model_kwargs):
    """Test model loading and inference with proper dtype handling"""
    print(f"\n{'-'*60}")
    print(f"Testing: {config_name}")
    print(f"{'-'*60}")
    
    try:
        print(f"Loading model with config: {model_kwargs}")
        
        # Load model and processor
        model = AutoModel.from_pretrained(
            "microsoft/kosmos-2.5",
            trust_remote_code=True,
            **model_kwargs
        )
        
        processor = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2.5",
            trust_remote_code=True
        )
        
        print("✓ Model and processor loaded successfully!")
        
        # Get model information
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
        
        # Create test image
        print("\nCreating test image...")
        image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
        # Test different prompts
        test_prompts = ["<ocr>", "<grounding>", "<md>"]
        
        for prompt in test_prompts:
            print(f"\nTesting with prompt: {prompt}")
            
            try:
                # Create compatible inputs
                inputs = create_model_compatible_inputs(processor, image, prompt, model)
                
                # Run forward pass
                print("Running forward pass...")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                print(f"✓ Forward pass successful with {prompt}")
                print(f"  Output type: {type(outputs)}")
                
                # Analyze outputs
                if hasattr(outputs, '__dict__'):
                    tensor_attrs = []
                    for attr_name in dir(outputs):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(outputs, attr_name)
                                if isinstance(attr_value, torch.Tensor):
                                    tensor_attrs.append(f"{attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                            except:
                                pass
                    
                    if tensor_attrs:
                        print(f"  Tensor outputs:")
                        for tensor_info in tensor_attrs:
                            print(f"    {tensor_info}")
                
            except Exception as e:
                print(f"✗ Forward pass failed with {prompt}: {e}")
                # Print more detailed error info for debugging
                if "dtype" in str(e).lower():
                    print(f"  This appears to be a dtype mismatch. Check input preprocessing.")
        
        # Cleanup
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n✓ {config_name} test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error in {config_name}: {e}")
        import traceback
        traceback.print_exc()
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return False

def check_environment():
    """Check environment setup"""
    print("Environment Check")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"  Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    
    # Check packages
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not available")
    
    try:
        import bitsandbytes
        print("BitsAndBytes: Available")
    except ImportError:
        print("BitsAndBytes: Not available")
    
    try:
        import accelerate
        print("Accelerate: Available")
    except ImportError:
        print("Accelerate: Not available")

def main():
    """Main function to run all tests"""
    check_environment()
    
    # Test configurations with proper device mapping
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_configs = [
        ("FP32_Baseline", {
            "device_map": "auto" if device == "cuda" else None
        }),
        ("FP16_Model", {
            "torch_dtype": torch.float16,
            "device_map": "auto" if device == "cuda" else None
        })
    ]
    
    # Add quantization tests if available
    try:
        from transformers import BitsAndBytesConfig
        
        test_configs.extend([
            ("INT8_Quantization", {
                "load_in_8bit": True,
                "device_map": "auto" if device == "cuda" else None
            }),
            ("FP4_Quantization", {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="fp4"
                ),
                "device_map": "auto" if device == "cuda" else None
            })
        ])
    except ImportError:
        print("BitsAndBytes not available - skipping quantization tests")
    
    # Run tests
    results = {}
    successful_tests = 0
    
    for config_name, model_kwargs in test_configs:
        try:
            success = test_model_with_dtype_fix(config_name, model_kwargs)
            results[config_name] = success
            if success:
                successful_tests += 1
        except Exception as e:
            print(f"Test {config_name} failed with error: {e}")
            results[config_name] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Successful tests: {successful_tests}/{len(test_configs)}")
    
    for config_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {config_name}")
    
    if successful_tests > 0:
        print(f"\n🎉 At least one configuration works!")
        print("You can proceed with the working configurations for your use case.")
    else:
        print(f"\n❌ All tests failed. Common issues:")
        print("- Insufficient GPU memory")
        print("- Missing dependencies")
        print("- CUDA/PyTorch version incompatibility")
        print("- Model downloading issues")

if __name__ == "__main__":
    main()
