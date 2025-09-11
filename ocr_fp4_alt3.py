"""
Working OCR Inference for Quantized Kosmos-2.5
Fixes generation errors and uses only methods that actually work
NO forward pass logits - only generation methods
"""

import torch
import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional

from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import re

class WorkingOCREngine:
    """OCR engine that actually works with quantized Kosmos-2.5 models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_quantized_model(self):
        """Load quantized model and validate generation capabilities"""
        print(f"🔄 Loading quantized model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # Load model and processor
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
        
        print(f"    ✅ Model loaded: {type(self.model).__name__}")
        print(f"    📊 Model dtype: {next(self.model.parameters()).dtype}")
        print(f"    🖥️  Model device: {next(self.model.parameters()).device}")
        
        # Validate generation methods
        has_language_model = hasattr(self.model, 'language_model')
        has_lm_generate = has_language_model and hasattr(self.model.language_model, 'generate')
        has_direct_generate = hasattr(self.model, 'generate')
        
        print(f"    🔍 Language model component: {'✅' if has_language_model else '❌'}")
        print(f"    🔍 Language model.generate(): {'✅' if has_lm_generate else '❌'}")
        print(f"    🔍 Direct model.generate(): {'✅' if has_direct_generate else '❌'}")
        
        if not (has_lm_generate or has_direct_generate):
            raise ValueError("❌ Fatal error: No generation methods available!")
        
        print(f"    ✅ Generation methods validated")
        return True
    
    def prepare_inputs_with_dtype_handling(self, image: Image.Image, prompt: str = "<ocr>") -> Dict:
        """Prepare inputs with proper dtype conversion for quantized models"""
        # Get model's actual parameters
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Convert all tensors to match model's device and dtype
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
    
    def extract_text_language_model_method(self, image: Image.Image, prompt: str = "<ocr>") -> str:
        """Extract text using model.language_model.generate() - PREFERRED METHOD"""
        if not hasattr(self.model, 'language_model') or not hasattr(self.model.language_model, 'generate'):
            return ""
        
        print(f"        🔄 Using language_model.generate()...")
        
        try:
            inputs = self.prepare_inputs_with_dtype_handling(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                print(f"        ❌ No input_ids in processed inputs")
                return ""
            
            # Generate using language model component
            with torch.no_grad():
                generated_ids = self.model.language_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=300,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract ONLY the new tokens (skip input prompt)
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                cleaned_text = self.clean_and_post_process_text(generated_text, prompt)
                print(f"        ✅ Generated {len(cleaned_text)} characters")
                return cleaned_text
            else:
                print(f"        ❌ No new tokens generated")
                return ""
        
        except Exception as e:
            print(f"        ❌ Language model generation failed: {e}")
            return ""
    
    def extract_text_direct_method(self, image: Image.Image, prompt: str = "<ocr>") -> str:
        """Extract text using model.generate() - FALLBACK METHOD"""
        if not hasattr(self.model, 'generate'):
            return ""
        
        print(f"        🔄 Using direct model.generate()...")
        
        try:
            inputs = self.prepare_inputs_with_dtype_handling(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                print(f"        ❌ No input_ids in processed inputs")
                return ""
            
            # Generate using direct model method
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=300,
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
                
                cleaned_text = self.clean_and_post_process_text(generated_text, prompt)
                print(f"        ✅ Generated {len(cleaned_text)} characters")
                return cleaned_text
            else:
                print(f"        ❌ No new tokens generated")
                return ""
        
        except Exception as e:
            print(f"        ❌ Direct generation failed: {e}")
            return ""
    
    def clean_and_post_process_text(self, raw_text: str, prompt: str) -> str:
        """Clean and post-process extracted OCR text"""
        if not raw_text:
            return ""
        
        # Remove the prompt from output (sometimes gets included)
        cleaned = raw_text.replace(prompt, "").strip()
        
        # Remove XML/HTML-like tags that sometimes appear
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        # Remove excessive repeated characters (generation artifacts)
        cleaned = re.sub(r'(.)\1{6,}', r'\1', cleaned)
        
        # Split into lines and filter out junk
        lines = cleaned.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 1:
                # Skip lines that are just repeated characters or symbols
                if not re.match(r'^(.)\1+$', line) and not re.match(r'^[^\w\s]+$', line):
                    clean_lines.append(line)
        
        result = '\n'.join(clean_lines)
        return result.strip()
    
    def run_working_ocr_extraction(self, image: Image.Image) -> Dict:
        """Run OCR using only working generation methods (NO forward pass logits)"""
        print(f"\n🔍 Running Working OCR Extraction")
        print(f"⚠️  SKIPPING forward pass logits (doesn't work with quantized models)")
        print(f"✅ Using ONLY generation methods that actually produce text")
        
        # Define generation methods in order of preference
        extraction_methods = [
            ("language_model_generate", self.extract_text_language_model_method),
            ("direct_model_generate", self.extract_text_direct_method)
        ]
        
        best_text = ""
        best_method = ""
        method_results = {}
        
        for method_name, method_func in extraction_methods:
            print(f"\n    --- Testing {method_name.upper()} ---")
            
            try:
                start_time = time.time()
                extracted_text = method_func(image)
                extraction_time = time.time() - start_time
                
                success = len(extracted_text) > 15  # Minimum viable text length
                
                method_results[method_name] = {
                    "success": success,
                    "text": extracted_text,
                    "extraction_time": extraction_time,
                    "text_length": len(extracted_text)
                }
                
                # Keep track of best result
                if len(extracted_text) > len(best_text):
                    best_text = extracted_text
                    best_method = method_name
                
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"        {status}: {len(extracted_text)} chars in {extraction_time:.2f}s")
                
                # If we got good results, we can stop trying other methods
                if success and len(extracted_text) > 50:
                    print(f"        ✅ Excellent result found, stopping search")
                    break
                
            except Exception as e:
                print(f"        ❌ Method failed with error: {e}")
                method_results[method_name] = {
                    "success": False,
                    "text": "",
                    "extraction_time": 0,
                    "text_length": 0,
                    "error": str(e)
                }
        
        return {
            "success": len(best_text) > 15,
            "best_method": best_method,
            "extracted_text": best_text,
            "method_results": method_results,
            "approach": "generation_only_methods",
            "skipped_methods": ["forward_pass_logits", "model_forward_with_logits"]
        }
    
    def create_realistic_test_document(self) -> Image.Image:
        """Create a realistic document image for OCR testing"""
        # Create image
        image = Image.new('RGB', (650, 550), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Try to load better fonts, fallback to default
        try:
            font_title = ImageFont.truetype("arial.ttf", 26)
            font_header = ImageFont.truetype("arial.ttf", 19)
            font_body = ImageFont.truetype("arial.ttf", 15)
            font_small = ImageFont.truetype("arial.ttf", 13)
        except:
            # Fallback to default font
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw realistic invoice content
        y = 35
        draw.text((50, y), "INVOICE #INV-2024-09-11-001", fill=(0, 0, 0), font=font_title)
        y += 60
        
        draw.text((50, y), "Invoice Date: September 11, 2024", fill=(0, 0, 0), font=font_header)
        y += 30
        draw.text((50, y), "Due Date: October 11, 2024", fill=(0, 0, 0), font=font_body)
        y += 45
        
        draw.text((50, y), "Bill To:", fill=(0, 0, 0), font=font_header)
        y += 28
        draw.text((80, y), "Advanced AI Solutions LLC", fill=(0, 0, 0), font=font_body)
        y += 22
        draw.text((80, y), "789 Technology Drive, Suite 200", fill=(0, 0, 0), font=font_body)
        y += 22
        draw.text((80, y), "San Francisco, CA 94105", fill=(0, 0, 0), font=font_body)
        y += 22
        draw.text((80, y), "contact@aisolutions.tech", fill=(0, 0, 0), font=font_body)
        y += 22
        draw.text((80, y), "Phone: (555) 987-6543", fill=(0, 0, 0), font=font_body)
        y += 45
        
        draw.text((50, y), "Services Provided:", fill=(0, 0, 0), font=font_header)
        y += 32
        draw.text((80, y), "• AI Model Quantization Services - $3,500.00", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((80, y), "• OCR System Integration - $2,250.00", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((80, y), "• Performance Optimization - $1,750.00", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((80, y), "• Technical Documentation - $500.00", fill=(0, 0, 0), font=font_body)
        y += 45
        
        draw.text((50, y), "TOTAL AMOUNT: $8,000.00", fill=(0, 0, 0), font=font_title)
        y += 50
        
        draw.text((50, y), "Payment Terms: Net 30 days", fill=(0, 0, 0), font=font_small)
        y += 25
        draw.text((50, y), "Thank you for your business!", fill=(0, 0, 0), font=font_small)
        
        return image

def main():
    parser = argparse.ArgumentParser(description="Working OCR Inference for Quantized Kosmos-2.5")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to quantized model directory")
    parser.add_argument("--image_path", type=str,
                       help="Path to image file (optional - creates test image if not provided)")
    parser.add_argument("--output_text", type=str,
                       help="Output text file for extracted text")
    parser.add_argument("--output_json", type=str,
                       help="Output JSON file for complete results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print(f"🚀 Working OCR Inference for Quantized Kosmos-2.5")
    print(f"=" * 60)
    
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model path not found: {args.model_path}")
        return 1
    
    try:
        # Initialize OCR engine
        print(f"📦 Initializing OCR engine...")
        ocr = WorkingOCREngine(args.model_path)
        
        # Load quantized model
        ocr.load_quantized_model()
        
        # Load or create image
        if args.image_path and os.path.exists(args.image_path):
            print(f"\n📸 Loading image: {args.image_path}")
            image = Image.open(args.image_path).convert('RGB')
        else:
            if args.image_path:
                print(f"⚠️  Image not found: {args.image_path}")
            print(f"📸 Creating realistic test document...")
            image = ocr.create_realistic_test_document()
        
        print(f"    Image size: {image.size}")
        
        # Run OCR extraction
        print(f"\n🔄 Starting OCR extraction...")
        start_time = time.time()
        result = ocr.run_working_ocr_extraction(image)
        total_time = time.time() - start_time
        
        # Display results
        print(f"\n{'='*60}")
        print("OCR EXTRACTION RESULTS")
        print('='*60)
        
        if result["success"]:
            print(f"✅ OCR EXTRACTION SUCCESSFUL")
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"🎯 Best method: {result['best_method']}")
            print(f"📊 Text length: {len(result['extracted_text'])} characters")
            
            print(f"\n📄 EXTRACTED TEXT:")
            print("─" * 50)
            print(result["extracted_text"])
            print("─" * 50)
            
            # Show method performance details
            if args.verbose:
                print(f"\n📈 METHOD PERFORMANCE:")
                for method, details in result["method_results"].items():
                    if details["success"]:
                        print(f"   ✅ {method}: {details['text_length']} chars, {details['extraction_time']:.2f}s")
                    else:
                        error = details.get("error", "No text extracted")
                        print(f"   ❌ {method}: {error}")
            
            print(f"\n🚫 Skipped methods: {', '.join(result['skipped_methods'])}")
            print(f"✅ Used approach: {result['approach']}")
        
        else:
            print(f"❌ OCR EXTRACTION FAILED")
            print(f"⏱️  Total time: {total_time:.2f} seconds")
            print(f"\n💥 All generation methods failed to extract text")
            
            print(f"\nMethod failure details:")
            for method, details in result["method_results"].items():
                error = details.get("error", "No text generated")
                print(f"   ❌ {method}: {error}")
            
            print(f"\n🔧 Troubleshooting suggestions:")
            print(f"   - Verify the quantized model works correctly")
            print(f"   - Try a different quantization method from the suite")
            print(f"   - Check if the original (non-quantized) model works")
            print(f"   - Ensure the image contains clear, readable text")
            print(f"   - Try with a different image file")
        
        # Save results if requested
        if args.output_text and result["success"]:
            with open(args.output_text, 'w', encoding='utf-8') as f:
                f.write(result["extracted_text"])
            print(f"\n💾 Extracted text saved: {args.output_text}")
        
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
            print(f"💾 Complete results saved: {args.output_json}")
        
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
