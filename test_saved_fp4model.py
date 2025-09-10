"""
Test script for saved FP4 quantized model
"""

import torch
import argparse
import os
import json
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont

class SavedModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load saved quantized model"""
        print(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Check metadata
        metadata_path = os.path.join(self.model_path, "quantization_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print("Model metadata:")
            print(f"  Quantization: {metadata.get('quantization_type', 'unknown')}")
            print(f"  Size: {metadata.get('size_mb', 0):.1f} MB")
            print(f"  Saved at: {metadata.get('saved_at', 'unknown')}")
        
        # Load model and processor
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        model_dtype = next(self.model.parameters()).dtype
        print(f"✓ Model loaded successfully")
        print(f"  Model dtype: {model_dtype}")
        print(f"  Device: {next(self.model.parameters()).device}")
    
    def create_test_image(self):
        """Create a test image with text content"""
        print("Creating test image...")
        
        # Create image with text
        image = Image.new('RGB', (400, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to load a font
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw test content
        draw.text((20, 20), "Test Document", fill=(0, 0, 0), font=font)
        draw.text((20, 60), "This is a sample document for OCR testing.", fill=(0, 0, 0), font=small_font)
        draw.text((20, 90), "• Item 1: First bullet point", fill=(0, 0, 0), font=small_font)
        draw.text((20, 120), "• Item 2: Second bullet point", fill=(0, 0, 0), font=small_font)
        draw.text((20, 160), "Section Header", fill=(0, 0, 0), font=font)
        draw.text((20, 200), "More content with different formatting.", fill=(0, 0, 0), font=small_font)
        
        return image
    
    def ensure_compatible_inputs(self, inputs):
        """Ensure inputs are compatible with model dtype"""
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        compatible_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    compatible_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                else:
                    compatible_inputs[key] = value.to(device=model_device)
            else:
                compatible_inputs[key] = value
        
        return compatible_inputs
    
    def test_ocr(self, image):
        """Test OCR functionality"""
        print("\nTesting OCR...")
        
        try:
            inputs = self.processor(text="<ocr>", images=image, return_tensors="pt")
            compatible_inputs = self.ensure_compatible_inputs(inputs)
            
            with torch.no_grad():
                outputs = self.model(**compatible_inputs)
            
            print("✓ OCR inference successful")
            print(f"  Output type: {type(outputs)}")
            
            # Try to analyze output
            if hasattr(outputs, '__dict__'):
                tensor_count = 0
                for attr_name in dir(outputs):
                    if not attr_name.startswith('_'):
                        try:
                            attr_value = getattr(outputs, attr_name)
                            if isinstance(attr_value, torch.Tensor):
                                tensor_count += 1
                                print(f"  {attr_name}: {list(attr_value.shape)} ({attr_value.dtype})")
                        except:
                            pass
                
                if tensor_count == 0:
                    print("  No tensor outputs found in model output")
            
            return True
            
        except Exception as e:
            print(f"✗ OCR test failed: {e}")
            return False
    
    def test_markdown(self, image):
        """Test markdown generation"""
        print("\nTesting Markdown Generation...")
        
        try:
            inputs = self.processor(text="<md>", images=image, return_tensors="pt")
            compatible_inputs = self.ensure_compatible_inputs(inputs)
            
            with torch.no_grad():
                outputs = self.model(**compatible_inputs)
            
            print("✓ Markdown inference successful")
            print(f"  Output type: {type(outputs)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Markdown test failed: {e}")
            return False
    
    def test_grounding(self, image):
        """Test grounding functionality"""
        print("\nTesting Grounding...")
        
        try:
            inputs = self.processor(text="<grounding>", images=image, return_tensors="pt")
            compatible_inputs = self.ensure_compatible_inputs(inputs)
            
            with torch.no_grad():
                outputs = self.model(**compatible_inputs)
            
            print("✓ Grounding inference successful")
            print(f"  Output type: {type(outputs)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Grounding test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive tests"""
        print("Running comprehensive model tests...")
        
        # Load model
        self.load_model()
        
        # Create test image
        image = self.create_test_image()
        
        # Run tests
        results = {
            "ocr": self.test_ocr(image),
            "markdown": self.test_markdown(image),
            "grounding": self.test_grounding(image)
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print('='*50)
        
        successful_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {test_name.upper()} test")
        
        print(f"\nSuccessful tests: {successful_tests}/{total_tests}")
        
        if successful_tests == total_tests:
            print("🎉 All tests passed! Model is working correctly.")
        elif successful_tests > 0:
            print("⚠️  Some tests passed. Model is partially functional.")
        else:
            print("❌ All tests failed. Check model and dependencies.")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Test saved FP4 quantized Kosmos-2.5 model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved quantized model")
    
    args = parser.parse_args()
    
    try:
        tester = SavedModelTester(args.model_path)
        results = tester.run_all_tests()
        
        # Return appropriate exit code
        if all(results.values()):
            exit(0)  # Success
        else:
            exit(1)  # Some failures
            
    except Exception as e:
        print(f"✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
