"""
Debug script to understand Kosmos-2.5 model structure and capabilities
"""

from transformers import AutoModel, AutoProcessor
import torch

def debug_kosmos25():
    print("Debugging Kosmos-2.5 Model Structure")
    print("=" * 40)
    
    try:
        # Load model without quantization first
        print("Loading model...")
        model = AutoModel.from_pretrained(
            "microsoft/kosmos-2.5",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained(
            "microsoft/kosmos-2.5",
            trust_remote_code=True
        )
        
        print("✓ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model class: {model.__class__.__name__}")
        
        # Check available methods
        print("\nAvailable methods:")
        methods = [method for method in dir(model) if not method.startswith('_')]
        for method in sorted(methods)[:20]:  # Show first 20 methods
            print(f"  - {method}")
        
        # Check if it has generate method
        has_generate = hasattr(model, 'generate')
        print(f"\nHas 'generate' method: {has_generate}")
        
        # Check model architecture
        print(f"\nModel config: {model.config}")
        
        # Test basic forward pass
        print("\nTesting basic forward pass...")
        from PIL import Image
        
        # Create a simple test image
        image = Image.new('RGB', (224, 224), color='white')
        prompt = "<ocr>"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        print(f"Input keys: {list(inputs.keys())}")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"Output type: {type(outputs)}")
        if hasattr(outputs, 'logits'):
            print(f"Logits shape: {outputs.logits.shape}")
        
        print("\n✓ Basic forward pass successful!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_kosmos25()
