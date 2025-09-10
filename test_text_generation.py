"""
Test script to verify text generation capabilities of saved model
"""

import torch
import argparse
import os
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import time

class TextGenerationTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the saved model"""
        print(f"Loading model from: {self.model_path}")
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print("✓ Model loaded successfully")
        
        # Check if model has generate method
        if hasattr(self.model, 'generate'):
            print("✓ Model has generate() method")
        else:
            print("⚠️  Model does not have generate() method")
            print("Available methods:", [m for m in dir(self.model) if not m.startswith('_')])
    
    def create_test_image(self):
        """Create a simple test image with clear text"""
        image = Image.new('RGB', (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Simple, clear text
        draw.text((20, 20), "Hello World", fill=(0, 0, 0), font=font)
        draw.text((20, 60), "This is a test", fill=(0, 0, 0), font=font)
        draw.text((20, 100), "123 Main Street", fill=(0, 0, 0), font=font)
        draw.text((20, 140), "Phone: 555-1234", fill=(0, 0, 0), font=font)
        
        return image
    
    def prepare_inputs(self, image, prompt):
        """Prepare inputs with correct dtype"""
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    processed_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                else:
                    processed_inputs[key] = value.to(device=model_device)
            else:
                processed_inputs[key] = value
        
        return processed_inputs
    
    def test_basic_generation(self, image, prompt, max_tokens=256):
        """Test basic text generation"""
        print(f"\nTesting generation with prompt: '{prompt}'")
        
        try:
            inputs = self.prepare_inputs(image, prompt)
            input_ids = inputs["input_ids"]
            
            print(f"Input shape: {input_ids.shape}")
            print(f"Input tokens: {input_ids.shape[1]}")
            
            # Simple generation
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    **{k: v for k, v in inputs.items() if k != "input_ids"}
                )
            
            # Decode result
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            generated_text = self.processor.tokenizer.decode(
                new_tokens[0], 
                skip_special_tokens=True
            )
            
            print(f"✓ Generation successful")
            print(f"Generated tokens: {new_tokens.shape[1]}")
            print(f"Generated text: '{generated_text}'")
            print(f"Text length: {len(generated_text)}")
            
            return True, generated_text
            
        except Exception as e:
            print(f"✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, ""
    
    def test_multiple_prompts(self, image):
        """Test generation with multiple prompts"""
        prompts = [
            "<ocr>",
            "<md>", 
            "<grounding>",
            "What is in this image?",
            "Extract text from this image:"
        ]
        
        results = {}
        
        for prompt in prompts:
            success, text = self.test_basic_generation(image, prompt, max_tokens=128)
            results[prompt] = {
                "success": success,
                "text": text,
                "has_content": bool(text.strip())
            }
        
        return results
    
    def run_comprehensive_test(self):
        """Run comprehensive text generation tests"""
        print("Comprehensive Text Generation Test")
        print("=" * 50)
        
        # Load model
        self.load_model()
        
        # Create test image
        print("\nCreating test image...")
        image = self.create_test_image()
        
        # Test multiple prompts
        print("\nTesting multiple prompts...")
        results = self.test_multiple_prompts(image)
        
        # Print summary
        print(f"\n{'='*50}")
        print("TEST RESULTS SUMMARY")
        print('='*50)
        
        successful_prompts = 0
        content_prompts = 0
        
        for prompt, result in results.items():
            status = "✓" if result["success"] else "✗"
            content = "📝" if result["has_content"] else "📄"
            
            print(f"{status} {content} {prompt}")
            if result["success"]:
                successful_prompts += 1
                if result["has_content"]:
                    content_prompts += 1
                    print(f"    → '{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}'")
        
        print(f"\nStatistics:")
        print(f"  Successful generations: {successful_prompts}/{len(results)}")
        print(f"  Generations with content: {content_prompts}/{len(results)}")
        
        if content_prompts > 0:
            print(f"\n🎉 Model can generate text! Use prompts that produced content.")
        elif successful_prompts > 0:
            print(f"\n⚠️  Model generates but outputs are empty. Try:")
            print("    - Different generation parameters")
            print("    - Longer max_new_tokens")
            print("    - Different sampling methods")
        else:
            print(f"\n❌ Model cannot generate text. Check:")
            print("    - Model compatibility")
            print("    - Input preprocessing")
            print("    - Model loading errors")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Test text generation capabilities")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model")
    
    args = parser.parse_args()
    
    tester = TextGenerationTester(args.model_path)
    results = tester.run_comprehensive_test()
    
    # Save results
    import json
    output_file = "generation_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
