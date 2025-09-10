"""
Script to thoroughly explore the Kosmos-2.5 model outputs and understand the structure
"""

from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
import torch
from PIL import Image
import json

def deep_analyze_output(obj, name="output", max_depth=3, current_depth=0):
    """Recursively analyze an object structure"""
    if current_depth > max_depth:
        return f"Max depth reached for {name}"
    
    analysis = {}
    
    if isinstance(obj, torch.Tensor):
        return {
            "type": "Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "requires_grad": obj.requires_grad,
            "sample_values": obj.flatten()[:5].tolist() if obj.numel() > 0 else []
        }
    elif hasattr(obj, '__dict__'):
        analysis["type"] = str(type(obj))
        analysis["attributes"] = {}
        
        for attr_name in dir(obj):
            if not attr_name.startswith('_') and not callable(getattr(obj, attr_name, None)):
                try:
                    attr_value = getattr(obj, attr_name)
                    analysis["attributes"][attr_name] = deep_analyze_output(
                        attr_value, f"{name}.{attr_name}", max_depth, current_depth + 1
                    )
                except Exception as e:
                    analysis["attributes"][attr_name] = f"Error accessing: {e}"
    else:
        return {"type": str(type(obj)), "value": str(obj)[:100]}
    
    return analysis

def test_kosmos_with_analysis():
    """Test Kosmos-2.5 and provide deep analysis of outputs"""
    
    print("Kosmos-2.5 Deep Output Analysis")
    print("=" * 50)
    
    try:
        # Load model with FP4 quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4"
        )
        
        print("Loading model...")
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
        
        print("✓ Model loaded successfully")
        
        # Create test image
        image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
        # Test different prompts
        test_cases = [
            {"prompt": "<ocr>", "description": "OCR task"},
            {"prompt": "<grounding>", "description": "Grounding task"},
            {"prompt": "<md>", "description": "Markdown task"}
        ]
        
        model_dtype = next(model.parameters()).dtype
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'-'*30}")
            print(f"Test {i+1}: {test_case['description']}")
            print(f"Prompt: {test_case['prompt']}")
            print(f"{'-'*30}")
            
            try:
                # Prepare inputs
                inputs = processor(text=test_case['prompt'], images=image, return_tensors="pt")
                
                # Convert to correct dtype
                processed_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.dtype.is_floating_point:
                            processed_inputs[key] = value.to(dtype=model_dtype)
                        else:
                            processed_inputs[key] = value
                    else:
                        processed_inputs[key] = value
                
                # Run inference
                with torch.no_grad():
                    outputs = model(**processed_inputs)
                
                print("✓ Inference successful")
                
                # Deep analysis
                analysis = deep_analyze_output(outputs, "model_output", max_depth=2)
                
                print(f"Output type: {analysis.get('type', 'Unknown')}")
                
                if "attributes" in analysis:
                    print("Available attributes:")
                    for attr_name, attr_info in analysis["attributes"].items():
                        if isinstance(attr_info, dict) and attr_info.get("type") == "Tensor":
                            print(f"  {attr_name}: Tensor {attr_info['shape']} ({attr_info['dtype']})")
                        else:
                            print(f"  {attr_name}: {attr_info.get('type', 'Unknown')}")
                
                # Save detailed analysis to file
                output_file = f"kosmos_output_analysis_{test_case['prompt'].replace('<', '').replace('>', '')}.json"
                with open(output_file, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                print(f"Detailed analysis saved to {output_file}")
                
            except Exception as e:
                print(f"✗ Test failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*50}")
        print("Analysis Complete!")
        print("Check the generated JSON files for detailed output structure.")
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_kosmos_with_analysis()
