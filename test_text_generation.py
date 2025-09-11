"""
Fixed text generation script that properly accesses the language model component
"""

import torch
import argparse
import os
from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import time

class FixedTextGenerationTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the saved model and inspect its structure"""
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
        
        # Inspect model structure
        print("\nModel structure analysis:")
        print(f"Model type: {type(self.model)}")
        
        # Check for language model component
        if hasattr(self.model, 'language_model'):
            print("✓ Found language_model component")
            if hasattr(self.model.language_model, 'generate'):
                print("✓ language_model has generate() method")
            else:
                print("✗ language_model doesn't have generate() method")
        else:
            print("✗ No language_model component found")
        
        # Check for text model component (alternative)
        if hasattr(self.model, 'text_model'):
            print("✓ Found text_model component")
            if hasattr(self.model.text_model, 'generate'):
                print("✓ text_model has generate() method")
        
        # List all main attributes
        main_attrs = [attr for attr in dir(self.model) if not attr.startswith('_') and not callable(getattr(self.model, attr, None))]
        print(f"Main model attributes: {main_attrs}")
    
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
    
    def extract_text_from_forward_pass(self, image, prompt, max_tokens=256):
        """Extract text using forward pass and logits decoding"""
        print(f"\nTesting forward pass with prompt: '{prompt}'")
        
        try:
            inputs = self.prepare_inputs(image, prompt)
            
            # Run forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            print(f"✓ Forward pass successful")
            print(f"Output type: {type(outputs)}")
            
            # Analyze outputs to find text logits
            text_results = []
            
            # Check different possible output attributes
            possible_logits = ['logits', 'prediction_logits', 'language_model_outputs', 'text_outputs']
            
            for attr_name in possible_logits:
                if hasattr(outputs, attr_name):
                    try:
                        logits = getattr(outputs, attr_name)
                        if isinstance(logits, torch.Tensor) and logits.dim() >= 2:
                            print(f"Found {attr_name}: {logits.shape}")
                            
                            # Try to decode
                            predicted_ids = torch.argmax(logits, dim=-1)
                            
                            if predicted_ids.dim() > 1:
                                # Take first batch
                                token_ids = predicted_ids[0]
                            else:
                                token_ids = predicted_ids
                            
                            # Limit tokens
                            token_ids = token_ids[:max_tokens]
                            
                            # Try to decode
                            try:
                                decoded_text = self.processor.tokenizer.decode(
                                    token_ids, 
                                    skip_special_tokens=True
                                )
                                
                                if decoded_text.strip():
                                    text_results.append({
                                        'source': attr_name,
                                        'text': decoded_text.strip(),
                                        'tokens_used': len(token_ids)
                                    })
                                    print(f"✓ Decoded text from {attr_name}: '{decoded_text[:100]}{'...' if len(decoded_text) > 100 else ''}'")
                            except Exception as decode_error:
                                print(f"⚠️  Failed to decode {attr_name}: {decode_error}")
                    except Exception as attr_error:
                        print(f"⚠️  Error accessing {attr_name}: {attr_error}")
            
            return True, text_results
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False, []
    
    def test_language_model_generation(self, image, prompt, max_tokens=256):
        """Test generation using the language model component"""
        print(f"\nTesting language model generation with prompt: '{prompt}'")
        
        try:
            # Check if we have language_model
            if not hasattr(self.model, 'language_model'):
                print("✗ No language_model component found")
                return False, ""
            
            language_model = self.model.language_model
            
            if not hasattr(language_model, 'generate'):
                print("✗ language_model doesn't have generate method")
                return False, ""
            
            # Prepare inputs for the full model first
            inputs = self.prepare_inputs(image, prompt)
            
            # Run forward pass to get embeddings
            with torch.no_grad():
                model_outputs = self.model(**inputs)
            
            # Try to extract input_ids for generation
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                print("✗ No input_ids found")
                return False, ""
            
            # Try generation with language model
            generated_ids = language_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            
            # Decode result
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            generated_text = self.processor.tokenizer.decode(
                new_tokens[0], 
                skip_special_tokens=True
            )
            
            print(f"✓ Language model generation successful")
            print(f"Generated tokens: {new_tokens.shape[1]}")
            print(f"Generated text: '{generated_text}'")
            
            return True, generated_text
            
        except Exception as e:
            print(f"✗ Language model generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, ""
    
    def run_comprehensive_test(self):
        """Run comprehensive text extraction tests"""
        print("Fixed Text Generation/Extraction Test")
        print("=" * 50)
        
        # Load model
        self.load_model()
        
        # Create test image
        print("\nCreating test image...")
        image = self.create_test_image()
        
        # Test prompts
        prompts = ["<ocr>", "<md>", "<grounding>"]
        
        all_results = {}
        
        for prompt in prompts:
            print(f"\n{'='*30}")
            print(f"Testing prompt: {prompt}")
            print('='*30)
            
            # Method 1: Forward pass with logits extraction
            success1, text_results = self.extract_text_from_forward_pass(image, prompt)
            
            # Method 2: Language model generation
            success2, generated_text = self.test_language_model_generation(image, prompt)
            
            all_results[prompt] = {
                'forward_pass': {
                    'success': success1,
                    'results': text_results
                },
                'language_model': {
                    'success': success2,
                    'text': generated_text
                }
            }
        
        # Print final summary
        print(f"\n{'='*50}")
        print("FINAL TEST SUMMARY")
        print('='*50)
        
        for prompt, results in all_results.items():
            print(f"\nPrompt: {prompt}")
            
            if results['forward_pass']['success']:
                print("  ✓ Forward pass successful")
                if results['forward_pass']['results']:
                    for result in results['forward_pass']['results']:
                        print(f"    {result['source']}: '{result['text'][:50]}{'...' if len(result['text']) > 50 else ''}'")
                else:
                    print("    (No text extracted)")
            else:
                print("  ✗ Forward pass failed")
            
            if results['language_model']['success']:
                print("  ✓ Language model generation successful")
                if results['language_model']['text']:
                    print(f"    Text: '{results['language_model']['text'][:50]}{'...' if len(results['language_model']['text']) > 50 else ''}'")
                else:
                    print("    (Empty text generated)")
            else:
                print("  ✗ Language model generation failed")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Fixed text generation/extraction test")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to saved model")
    
    args = parser.parse_args()
    
    tester = FixedTextGenerationTester(args.model_path)
    results = tester.run_comprehensive_test()
    
    # Save results
    import json
    output_file = "fixed_generation_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
