"""
Enhanced OCR inference with proper text generation for Kosmos-2.5
"""

import torch
import argparse
import os
import json
import time
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import re

class EnhancedKosmos25OCR:
    def __init__(self, model_path="./fp4_quantized_model"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the saved FP4 quantized model"""
        print(f"Loading FP4 quantized model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
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
    
    def prepare_inputs(self, image, prompt="<ocr>"):
        """Prepare inputs with proper dtype handling"""
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Convert to compatible dtype and device
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
    
    def generate_text(self, image, prompt="<ocr>", max_new_tokens=512):
        """Generate text using the model's generation capabilities"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"Generating text with prompt: {prompt}")
        
        try:
            # Prepare inputs
            inputs = self.prepare_inputs(image, prompt)
            
            # Get input_ids for generation
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("No input_ids found in processed inputs")
            
            # Generation parameters
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Use greedy decoding for consistency
                "num_beams": 1,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            # Add other inputs if available
            for key, value in inputs.items():
                if key not in generation_kwargs and key != "input_ids":
                    generation_kwargs[key] = value
            
            print("Running text generation...")
            start_time = time.time()
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(**generation_kwargs)
            
            generation_time = time.time() - start_time
            print(f"✓ Text generation completed in {generation_time:.2f}s")
            
            # Decode the generated text
            # Remove the input prompt from generated text
            input_length = input_ids.shape[1]
            generated_text_ids = generated_ids[:, input_length:]
            
            # Decode
            generated_text = self.processor.tokenizer.decode(
                generated_text_ids[0], 
                skip_special_tokens=True
            ).strip()
            
            print(f"Raw generated text length: {len(generated_text)}")
            
            return {
                "success": True,
                "generated_text": generated_text,
                "generation_time": generation_time,
                "input_length": input_length,
                "output_length": generated_text_ids.shape[1]
            }
            
        except Exception as e:
            print(f"✗ Text generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "generated_text": "",
                "generation_time": 0
            }
    
    def post_process_ocr(self, raw_text):
        """Post-process OCR text to clean it up"""
        if not raw_text:
            return ""
        
        # Remove special tokens and cleanup
        cleaned_text = raw_text.replace("<ocr>", "").replace("</ocr>", "")
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)  # Remove remaining XML tags
        
        # Clean up whitespace
        lines = [line.strip() for line in cleaned_text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        # Join with proper spacing
        result = '\n'.join(lines)
        
        return result.strip()
    
    def run_ocr(self, image):
        """Run OCR with enhanced text generation"""
        print("Running enhanced OCR...")
        
        # Try different generation approaches
        approaches = [
            {"prompt": "<ocr>", "max_tokens": 512, "description": "Standard OCR"},
            {"prompt": "<ocr>", "max_tokens": 1024, "description": "Extended OCR"},
        ]
        
        results = []
        
        for approach in approaches:
            print(f"\nTrying: {approach['description']}")
            
            result = self.generate_text(
                image, 
                prompt=approach["prompt"],
                max_new_tokens=approach["max_tokens"]
            )
            
            if result["success"]:
                # Post-process the text
                processed_text = self.post_process_ocr(result["generated_text"])
                result["processed_text"] = processed_text
                result["approach"] = approach["description"]
                
                print(f"Generated text preview: {processed_text[:100]}...")
                
                if processed_text:
                    print(f"✓ Found non-empty OCR result with {approach['description']}")
                    results.append(result)
                    break  # Use first successful result
                else:
                    print(f"⚠️  Empty result with {approach['description']}")
            else:
                print(f"✗ Failed with {approach['description']}: {result['error']}")
        
        # Return best result or empty result
        if results:
            return results[0]
        else:
            return {
                "success": False,
                "error": "No OCR text could be generated",
                "processed_text": "",
                "generated_text": ""
            }
    
    def load_image(self, image_input):
        """Load image from various sources"""
        if image_input.startswith(('http://', 'https://')):
            print(f"Loading image from URL: {image_input}")
            try:
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from URL: {e}")
        
        elif os.path.exists(image_input):
            print(f"Loading image from file: {image_input}")
            try:
                image = Image.open(image_input)
                return image.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from file: {e}")
        
        elif image_input == "default":
            print("Creating default test image with text...")
            from PIL import ImageDraw, ImageFont
            
            # Create a more realistic test image
            image = Image.new('RGB', (600, 400), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            try:
                font_large = ImageFont.truetype("arial.ttf", 24)
                font_medium = ImageFont.truetype("arial.ttf", 18)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw realistic document content
            y = 30
            draw.text((50, y), "Invoice #12345", fill=(0, 0, 0), font=font_large)
            y += 40
            draw.text((50, y), "Date: January 15, 2024", fill=(0, 0, 0), font=font_medium)
            y += 30
            draw.text((50, y), "Customer: John Smith", fill=(0, 0, 0), font=font_medium)
            y += 30
            draw.text((50, y), "Address: 123 Main Street, City, State 12345", fill=(0, 0, 0), font=font_small)
            y += 40
            draw.text((50, y), "Items:", fill=(0, 0, 0), font=font_medium)
            y += 25
            draw.text((70, y), "• Product A - $25.00", fill=(0, 0, 0), font=font_small)
            y += 20
            draw.text((70, y), "• Product B - $15.50", fill=(0, 0, 0), font=font_small)
            y += 20
            draw.text((70, y), "• Service Fee - $10.00", fill=(0, 0, 0), font=font_small)
            y += 30
            draw.text((50, y), "Total: $50.50", fill=(0, 0, 0), font=font_medium)
            
            return image
        
        else:
            raise ValueError(f"Invalid image input: {image_input}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced OCR inference using FP4 quantized Kosmos-2.5")
    parser.add_argument("--model_path", type=str, default="./fp4_quantized_model",
                       help="Path to saved FP4 quantized model")
    parser.add_argument("--image", type=str, default="default",
                       help="Image path, URL, or 'default' for test image")
    parser.add_argument("--output", type=str,
                       help="Output file to save OCR results")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize OCR
    ocr = EnhancedKosmos25OCR(model_path=args.model_path)
    
    try:
        # Load model
        ocr.load_model()
        
        # Load image
        image = ocr.load_image(args.image)
        print(f"Image size: {image.size}")
        
        # Run OCR
        result = ocr.run_ocr(image)
        
        # Display results
        print(f"\n{'='*60}")
        print("ENHANCED OCR RESULTS")
        print('='*60)
        
        if result["success"]:
            processed_text = result.get("processed_text", "")
            raw_text = result.get("generated_text", "")
            
            print(f"✓ OCR successful")
            print(f"  Generation time: {result.get('generation_time', 0):.2f}s")
            print(f"  Output tokens: {result.get('output_length', 0)}")
            print(f"  Approach: {result.get('approach', 'Unknown')}")
            
            if processed_text:
                print(f"\nExtracted Text:")
                print("-" * 40)
                print(processed_text)
                
                print(f"\nRaw Generated Text:")
                print("-" * 40)
                print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
            else:
                print("\n⚠️  No text could be extracted from the image")
                print("This might indicate:")
                print("- The image contains no readable text")
                print("- The model needs different generation parameters")
                print("- The image preprocessing needs adjustment")
        else:
            print(f"✗ OCR failed: {result.get('error', 'Unknown error')}")
        
        # Save results
        if args.output:
            output_data = {
                "image_source": args.image,
                "model_path": args.model_path,
                "result": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
