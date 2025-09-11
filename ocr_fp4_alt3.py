"""
Working OCR Inference for Quantized Kosmos-2.5
Uses only generation methods that actually work - NO forward pass logits
"""

import torch
import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict

from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import re

class WorkingOCREngine:
    """OCR using only verified working generation methods"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_quantized_model(self):
        """Load quantized model and check capabilities"""
        print(f"🔄 Loading model: {self.model_path}")
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(self.model).__name__}")
        print(f"   Model dtype: {next(self.model.parameters()).dtype}")
        
        # Check generation methods
        has_lm_gen = hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'generate')
        has_direct_gen = hasattr(self.model, 'generate')
        
        print(f"   Language model generate: {'✅' if has_lm_gen else '❌'}")
        print(f"   Direct model generate: {'✅' if has_direct_gen else '❌'}")
        
        if not (has_lm_gen or has_direct_gen):
            raise ValueError("No generation methods available!")
        
        return True
    
    def prepare_inputs_with_dtype_conversion(self, image: Image.Image, prompt: str = "<ocr>") -> Dict:
        """Prepare inputs with proper dtype conversion for quantized models"""
        # Get model's actual dtype and device
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        # Process inputs normally
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Convert all tensor inputs to match model
        converted_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    # Convert floating point tensors to model's dtype
                    converted_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                else:
                    # Keep integer tensors as-is but move to correct device
                    converted_inputs[key] = value.to(device=model_device)
            else:
                converted_inputs[key] = value
        
        return converted_inputs
    
    def extract_text_language_model_generate(self, image: Image.Image, prompt: str = "<ocr>") -> str:
        """Extract text using model.language_model.generate() - PREFERRED METHOD"""
        if not hasattr(self.model, 'language_model') or not hasattr(self.model.language_model, 'generate'):
            return ""
        
        print(f"    🔄 Using language_model.generate()...")
        
        try:
            inputs = self.prepare_inputs_with_dtype_conversion(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                print(f"    ❌ No input_ids found")
                return ""
            
            # Generate using language model component
            with torch.no_grad():
                generated_ids = self.model.language_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract only NEW tokens (skip input prompt)
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                cleaned_text = self.clean_ocr_text(generated_text, prompt)
                print(f"    ✅ Generated {len(cleaned_text)} characters")
                return cleaned_text
            else:
                print(f"    ❌ No new tokens generated")
                return ""
        
        except Exception as e:
            print(f"    ❌ Language model generation failed: {e}")
            return ""
    
    def extract_text_direct_generate(self, image: Image.Image, prompt: str = "<ocr>") -> str:
        """Extract text using model.generate() - FALLBACK METHOD"""
        if not hasattr(self.model, 'generate'):
            return ""
        
        print(f"    🔄 Using direct model.generate()...")
        
        try:
            inputs = self.prepare_inputs_with_dtype_conversion(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                print(f"    ❌ No input_ids found")
                return ""
            
            # Generate using direct model method
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    **{k: v for k, v in inputs.items() if k != "input_ids"}
                )
            
            # Extract only NEW tokens
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                cleaned_text = self.clean_ocr_text(generated_text, prompt)
                print(f"    ✅ Generated {len(cleaned_text)} characters")
                return cleaned_text
            else:
                print(f"    ❌ No new tokens generated")
                return ""
        
        except Exception as e:
            print(f"    ❌ Direct generation failed: {e}")
            return ""
    
    def clean_ocr_text(self, raw_text: str, prompt: str) -> str:
        """Clean and post-process OCR text"""
        if not raw_text:
            return ""
        
        # Remove prompt from output
        cleaned = raw_text.replace(prompt, "").strip()
        
        # Remove XML/HTML-like tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove excessive repeated characters (generation artifact)
        cleaned = re.sub(r'(.)\1{5,}', r'\1', cleaned)
        
        # Split into lines and filter
        lines = cleaned.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:
                # Skip lines that are just repeated characters
                if not re.match(r'^(.)\1+$', line):
                    clean_lines.append(line)
        
        result = '\n'.join(clean_lines)
        return result.strip()
    
    def run_working_ocr(self, image: Image.Image) -> Dict:
        """Run OCR using only working generation methods"""
        print(f"\n🔍 Running Working OCR (Generation-Only Methods)")
        print(f"⚠️  SKIPPING forward pass logits (doesn't work)")
        
        # Try generation methods in order of preference
        generation_methods = [
            ("language_model_generate", self.extract_text_language_model_generate),
            ("direct_generate", self.extract_text_direct_generate)
        ]
        
        best_text = ""
        best_method = ""
        method_results = {}
        
        for method_name, method_func in generation_methods:
            print(f"\n--- {method_name.upper()} ---")
            
            try:
                start_time = time.time()
                extracted_text = method_func(image)
                extraction_time = time.time() - start_time
                
                success = len(extracted_text) > 10
                
                method_results[method_name] = {
                    "success": success,
                    "text": extracted_text,
                    "extraction_time": extraction_time,
                    "text_length": len(extracted_text)
                }
                
                if len(extracted_text) > len(best_text):
                    best_text = extracted_text
                    best_method = method_name
                
                status = "✅" if success else "❌"
                print(f"    {status} Result: {len(extracted_text)} chars in {extraction_time:.2f}s")
                
                # Stop if we got good results
                if success and len(extracted_text) > 50:
                    print(f"    ✅ Good result found, stopping")
                    break
                
            except Exception as e:
                print(f"    ❌ Method failed: {e}")
                method_results[method_name] = {
                    "success": False,
                    "text": "",
                    "extraction_time": 0,
                    "text_length": 0,
                    "error": str(e)
                }
        
        return {
            "success": len(best_text) > 10,
            "best_method": best_method,
            "extracted_text": best_text,
            "method_results": method_results,
            "approach": "generation_only",
            "skipped_methods": ["forward_pass_logits"]
        }
    
    def create_test_invoice_image(self) -> Image.Image:
        """Create a realistic invoice image for OCR testing"""
        image = Image.new('RGB', (600, 500), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Try to load fonts
        try:
            font_title = ImageFont.truetype("arial.ttf", 24)
            font_header = ImageFont.truetype("arial.ttf", 18)
            font_body = ImageFont.truetype("arial.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_body = ImageFont.load_default()
        
        # Draw invoice content
        y = 30
        draw.text((50, y), "INVOICE #INV-2024-0123", fill=(0, 0, 0), font=font_title)
        y += 50
        
        draw.text((50, y), "Date: September 11, 2024", fill=(0, 0, 0), font=font_header)
        y += 30
        draw.text((50, y), "Due: October 11, 2024", fill=(0, 0, 0), font=font_body)
        y += 40
        
        draw.text((50, y), "Bill To:", fill=(0, 0, 0), font=font_header)
        y += 25
        draw.text((70, y), "Tech Solutions Inc.", fill=(0, 0, 0), font=font_body)
        y += 20
        draw.text((70, y), "456 Business Ave", fill=(0, 0, 0), font=font_body)
        y += 20
        draw.text((70, y), "San Francisco, CA 94107", fill=(0, 0, 0), font=font_body)
        y += 20
        draw.text((70, y), "billing@techsolutions.com", fill=(0, 0, 0), font=font_body)
        y += 40
        
        draw.text((50, y), "Services:", fill=(0, 0, 0), font=font_header)
        y += 30
        draw.text((70, y), "• Web Development Services - $2,500.00", fill=(0, 0, 0), font=font_body)
        y += 25
        draw.text((70, y), "• Database Setup - $750.00", fill=(0, 0, 0), font=font_body)
        y += 25
        draw.text((70, y), "• Security Audit - $1,250.00", fill=(0, 0, 0), font=font_body)
        y += 40
        
        draw.text((50, y), "TOTAL: $4,500.00", fill=(0, 0, 0), font=font_title)
        y += 40
        
        draw.text((50, y), "Payment Terms: Net 30 days", fill=(0, 0, 0), font=font_body)
        
        return image

def main():
    parser = argparse.ArgumentParser(description="Working OCR for Quantized Kosmos-2.5")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to quantized model directory")
    parser.add_argument("--image_path", type=str,
                       help="Path to image file (creates test image if not provided)")
    parser.add_argument("--output_text", type=str,
                       help="Output text file")
    parser.add_argument("--output_json", type=str,
                       help="Output JSON results file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        return
    
    try:
        # Initialize OCR engine
        ocr = WorkingOCREngine(args.model_path)
        
        # Load model
        ocr.load_quantized_model()
        
        # Load or create test image
        if args.image_path and os.path.exists(args.image_path):
            print(f"📸 Loading image: {args.image_path}")
            image = Image.open(args.image_path).convert('RGB')
        else:
            print(f"📸 Creating test invoice image...")
            image = ocr.create_test_invoice_image()
        
        print(f"Image size: {image.size}")
        
        # Run OCR
        start_time = time.time()
        result = ocr.run_working_ocr(image)
        total_time = time.time() - start_time
        
        # Display results
        print(f"\n{'='*60}")
        print("WORKING OCR RESULTS")
        print('='*60)
        
        if result["success"]:
            print(f"✅ OCR SUCCESS in {total_time:.2f} seconds")
            print(f"Method used: {result['best_method']}")
            print(f"Text length: {len(result['extracted_text'])} characters")
            
            print(f"\n📄 EXTRACTED TEXT:")
            print("-" * 40)
            print(result["extracted_text"])
            
            # Method performance
            print(f"\n📊 METHOD PERFORMANCE:")
            for method, details in result["method_results"].items():
                if details["success"]:
                    print(f"   ✅ {method}: {details['text_length']} chars, {details['extraction_time']:.2f}s")
                else:
                    error = details.get("error", "No text generated")
                    print(f"   ❌ {method}: {error}")
            
            print(f"\nSkipped: {', '.join(result['skipped_methods'])}")
        
        else:
            print(f"❌ OCR FAILED")
            print(f"No working generation methods produced text")
            
            print(f"\nMethod attempts:")
            for method, details in result["method_results"].items():
                error = details.get("error", "No text generated")
                print(f"   ❌ {method}: {error}")
            
            print(f"\nTroubleshooting:")
            print(f"  - Try a different quantized model")
            print(f"  - Check if original model works")
            print(f"  - Verify image has clear, readable text")
        
        # Save results
        if args.output_text and result["success"]:
            with open(args.output_text, 'w', encoding='utf-8') as f:
                f.write(result["extracted_text"])
            print(f"\n💾 Text saved: {args.output_text}")
        
        if args.output_json:
            output_data = {
                "model_path": args.model_path,
                "image_path": args.image_path,
                "total_time": total_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "result": result
            }
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Results saved: {args.output_json}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()