"""
Advanced OCR Inference for Quantized Kosmos-2.5
Handles models without standard generate() methods
Multiple text extraction approaches with automatic fallbacks
"""

import torch
import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import re

class AdvancedOCREngine:
    """Advanced OCR engine with multiple extraction methods for quantized models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extraction_capabilities = {}
        
    def load_quantized_model(self):
        """Load quantized model and determine extraction capabilities"""
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
        
        # Load quantization info if available
        info_path = Path(self.model_path) / "quantization_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.extraction_capabilities = json.load(f).get("generation_capabilities", {})
            print(f"    📋 Loaded extraction capabilities from metadata")
        
        # Test extraction capabilities
        self.test_all_extraction_methods()
        
        return True
    
    def test_all_extraction_methods(self):
        """Test all available extraction methods"""
        print(f"    🔍 Testing extraction methods...")
        
        # Check basic capabilities
        has_language_model = hasattr(self.model, 'language_model')
        has_lm_generate = has_language_model and hasattr(self.model.language_model, 'generate')
        has_direct_generate = hasattr(self.model, 'generate')
        
        print(f"    🔍 Language model component: {'✅' if has_language_model else '❌'}")
        print(f"    🔍 Language model.generate(): {'✅' if has_lm_generate else '❌'}")
        print(f"    🔍 Direct model.generate(): {'✅' if has_direct_generate else '❌'}")
        
        # Quick test of methods
        test_image = Image.new('RGB', (100, 50), color=(255, 255, 255))
        
        if has_lm_generate:
            try:
                self.extract_text_language_model_generate(test_image, test_mode=True)
                print(f"    ✅ Language model generate: Working")
            except:
                print(f"    ❌ Language model generate: Failed")
                has_lm_generate = False
        
        if has_direct_generate:
            try:
                self.extract_text_direct_generate(test_image, test_mode=True)
                print(f"    ✅ Direct generate: Working")
            except:
                print(f"    ❌ Direct generate: Failed")
                has_direct_generate = False
        
        # Test forward pass method
        forward_pass_works = False
        try:
            self.extract_text_forward_pass_method(test_image, test_mode=True)
            forward_pass_works = True
            print(f"    ✅ Forward pass extraction: Working")
        except:
            print(f"    ❌ Forward pass extraction: Failed")
        
        # Test custom generation
        custom_gen_works = False
        try:
            self.extract_text_custom_generation(test_image, test_mode=True)
            custom_gen_works = True
            print(f"    ✅ Custom token generation: Working")
        except:
            print(f"    ❌ Custom token generation: Failed")
        
        if not (has_lm_generate or has_direct_generate or forward_pass_works or custom_gen_works):
            raise ValueError("❌ Fatal error: No working text extraction methods available!")
        
        print(f"    ✅ At least one extraction method is working")
    
    def prepare_inputs_with_proper_conversion(self, image: Image.Image, prompt: str = "<ocr>") -> Dict:
        """Prepare inputs with proper dtype/device conversion for quantized models"""
        # Get model's parameters
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Convert all tensors to match model
        converted_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    converted_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                else:
                    converted_inputs[key] = value.to(device=model_device)
            else:
                converted_inputs[key] = value
        
        return converted_inputs
    
    def extract_text_language_model_generate(self, image: Image.Image, prompt: str = "<ocr>", test_mode: bool = False) -> str:
        """Extract text using model.language_model.generate() - METHOD 1"""
        if not hasattr(self.model, 'language_model') or not hasattr(self.model.language_model, 'generate'):
            return ""
        
        if not test_mode:
            print(f"        🔄 Using language_model.generate()...")
        
        try:
            inputs = self.prepare_inputs_with_proper_conversion(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                return ""
            
            with torch.no_grad():
                generated_ids = self.model.language_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=400 if not test_mode else 3,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract only new tokens
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                if test_mode:
                    return "test_success"
                
                cleaned_text = self.clean_and_post_process_text(generated_text, prompt)
                if not test_mode:
                    print(f"        ✅ Generated {len(cleaned_text)} characters")
                return cleaned_text
            else:
                if not test_mode:
                    print(f"        ❌ No new tokens generated")
                return ""
        
        except Exception as e:
            if not test_mode:
                print(f"        ❌ Language model generation failed: {e}")
            if test_mode:
                raise e
            return ""
    
    def extract_text_direct_generate(self, image: Image.Image, prompt: str = "<ocr>", test_mode: bool = False) -> str:
        """Extract text using model.generate() - METHOD 2"""
        if not hasattr(self.model, 'generate'):
            return ""
        
        if not test_mode:
            print(f"        🔄 Using direct model.generate()...")
        
        try:
            inputs = self.prepare_inputs_with_proper_conversion(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                return ""
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=400 if not test_mode else 3,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    **{k: v for k, v in inputs.items() if k != "input_ids"}
                )
            
            # Extract only new tokens
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                if test_mode:
                    return "test_success"
                
                cleaned_text = self.clean_and_post_process_text(generated_text, prompt)
                if not test_mode:
                    print(f"        ✅ Generated {len(cleaned_text)} characters")
                return cleaned_text
            else:
                if not test_mode:
                    print(f"        ❌ No new tokens generated")
                return ""
        
        except Exception as e:
            if not test_mode:
                print(f"        ❌ Direct generation failed: {e}")
            if test_mode:
                raise e
            return ""
    
    def extract_text_forward_pass_method(self, image: Image.Image, prompt: str = "<ocr>", test_mode: bool = False) -> str:
        """Extract text using forward pass + beam search - METHOD 3"""
        if not test_mode:
            print(f"        🔄 Using forward pass + beam search...")
        
        try:
            inputs = self.prepare_inputs_with_proper_conversion(image, prompt)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Find logits in outputs
                logits = None
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits
                elif hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
                    logits = outputs.prediction_logits
                elif hasattr(outputs, 'language_model_outputs'):
                    lm_outputs = outputs.language_model_outputs
                    if hasattr(lm_outputs, 'logits') and lm_outputs.logits is not None:
                        logits = lm_outputs.logits
                
                if logits is None:
                    if not test_mode:
                        print(f"        ❌ No logits found in model output")
                    return ""
                
                if test_mode:
                    return "test_success"
                
                # Extract text from logits using greedy decoding
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = []
                
                for i in range(min(300, logits.shape[1] - input_length)):
                    if input_length + i >= logits.shape[1]:
                        break
                    
                    token_logits = logits[0, input_length + i, :]
                    next_token = torch.argmax(token_logits, dim=-1).item()
                    
                    if next_token == self.processor.tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token)
                
                if generated_tokens:
                    generated_text = self.processor.tokenizer.decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    ).strip()
                    
                    cleaned_text = self.clean_and_post_process_text(generated_text, prompt)
                    print(f"        ✅ Extracted {len(cleaned_text)} characters via forward pass")
                    return cleaned_text
                else:
                    print(f"        ❌ No tokens extracted from logits")
                    return ""
        
        except Exception as e:
            if not test_mode:
                print(f"        ❌ Forward pass extraction failed: {e}")
            if test_mode:
                raise e
            return ""
    
    def extract_text_custom_generation(self, image: Image.Image, prompt: str = "<ocr>", test_mode: bool = False) -> str:
        """Extract text using custom token-by-token generation - METHOD 4"""
        if not test_mode:
            print(f"        🔄 Using custom token-by-token generation...")
        
        try:
            inputs = self.prepare_inputs_with_proper_conversion(image, prompt)
            
            max_tokens = 3 if test_mode else 300
            generated_tokens = []
            current_inputs = inputs.copy()
            
            with torch.no_grad():
                for step in range(max_tokens):
                    outputs = self.model(**current_inputs)
                    
                    # Find logits
                    logits = None
                    if hasattr(outputs, 'logits') and outputs.logits is not None:
                        logits = outputs.logits
                    elif hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
                        logits = outputs.prediction_logits
                    
                    if logits is None:
                        if not test_mode:
                            print(f"        ❌ No logits available for custom generation")
                        return ""
                    
                    # Get next token
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == self.processor.tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    if test_mode and len(generated_tokens) >= 1:
                        return "test_success"
                    
                    # Update inputs for next iteration
                    current_inputs["input_ids"] = torch.cat([current_inputs["input_ids"], next_token], dim=-1)
            
            if test_mode:
                return "test_success" if generated_tokens else ""
            
            if generated_tokens:
                generated_text = self.processor.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                cleaned_text = self.clean_and_post_process_text(generated_text, prompt)
                print(f"        ✅ Generated {len(cleaned_text)} characters via custom generation")
                return cleaned_text
            else:
                print(f"        ❌ No tokens generated via custom method")
                return ""
        
        except Exception as e:
            if not test_mode:
                print(f"        ❌ Custom generation failed: {e}")
            if test_mode:
                raise e
            return ""
    
    def clean_and_post_process_text(self, raw_text: str, prompt: str) -> str:
        """Clean and post-process extracted text"""
        if not raw_text:
            return ""
        
        # Remove prompt from output
        cleaned = raw_text.replace(prompt, "").strip()
        
        # Remove XML/HTML-like tags
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
    
    def run_advanced_ocr_extraction(self, image: Image.Image) -> Dict:
        """Run OCR using all available extraction methods with automatic fallbacks"""
        print(f"\n🔍 Running Advanced OCR Extraction")
        print(f"✅ Multiple methods with automatic fallbacks")
        print(f"⚠️  Will try methods in order until one produces good text")
        
        # Define extraction methods in order of preference
        extraction_methods = [
            ("language_model_generate", self.extract_text_language_model_generate),
            ("direct_model_generate", self.extract_text_direct_generate),
            ("forward_pass_extraction", self.extract_text_forward_pass_method),
            ("custom_token_generation", self.extract_text_custom_generation)
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
                
                success = len(extracted_text) > 20  # Minimum viable text length
                
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
                
                status = "✅ SUCCESS" if success else "❌ INSUFFICIENT"
                print(f"        {status}: {len(extracted_text)} chars in {extraction_time:.2f}s")
                
                # If we got excellent results, we can stop trying other methods
                if success and len(extracted_text) > 100:
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
            "success": len(best_text) > 20,
            "best_method": best_method,
            "extracted_text": best_text,
            "method_results": method_results,
            "approach": "advanced_multi_method_extraction",
            "total_methods_tried": len(extraction_methods)
        }
    
    def create_comprehensive_test_document(self) -> Image.Image:
        """Create comprehensive test document with various text elements"""
        image = Image.new('RGB', (700, 650), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Try to load fonts, fallback to default
        try:
            font_title = ImageFont.truetype("arial.ttf", 28)
            font_header = ImageFont.truetype("arial.ttf", 20)
            font_body = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw comprehensive document content
        y = 40
        
        # Title
        draw.text((50, y), "COMPREHENSIVE BUSINESS ANALYSIS REPORT", fill=(0, 0, 0), font=font_title)
        y += 70
        
        # Date and metadata
        draw.text((50, y), "Report Date: September 11, 2024", fill=(0, 0, 0), font=font_header)
        y += 30
        draw.text((50, y), "Prepared by: Advanced Analytics Team", fill=(0, 0, 0), font=font_body)
        y += 25
        draw.text((50, y), "Document ID: RPT-2024-Q3-001", fill=(0, 0, 0), font=font_body)
        y += 50
        
        # Executive Summary section
        draw.text((50, y), "Executive Summary", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "This quarterly analysis reveals significant growth trends", fill=(0, 0, 0), font=font_body)
        y += 25
        draw.text((70, y), "across multiple business verticals and market segments.", fill=(0, 0, 0), font=font_body)
        y += 25
        draw.text((70, y), "Key performance indicators show 23% improvement over", fill=(0, 0, 0), font=font_body)
        y += 25
        draw.text((70, y), "the previous quarter with strong momentum continuing.", fill=(0, 0, 0), font=font_body)
        y += 45
        
        # Key Metrics section
        draw.text((50, y), "Key Performance Metrics", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "• Total Revenue: $4.7M (+23% QoQ)", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "• New Customer Acquisitions: 2,156", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "• Customer Satisfaction Score: 4.8/5.0", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "• Market Share Growth: +3.2%", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "• Employee Productivity Index: 94%", fill=(0, 0, 0), font=font_body)
        y += 45
        
        # Strategic Initiatives
        draw.text((50, y), "Strategic Initiatives in Progress", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "1. Digital Transformation Program", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "2. AI-Powered Customer Analytics Platform", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "3. International Market Expansion", fill=(0, 0, 0), font=font_body)
        y += 28
        draw.text((70, y), "4. Sustainable Business Operations Initiative", fill=(0, 0, 0), font=font_body)
        y += 45
        
        # Contact Information
        draw.text((50, y), "Contact Information", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "Email: analytics@businesscorp.com", fill=(0, 0, 0), font=font_small)
        y += 25
        draw.text((70, y), "Phone: (555) 123-4567", fill=(0, 0, 0), font=font_small)
        y += 25
        draw.text((70, y), "Website: www.businesscorp.com/reports", fill=(0, 0, 0), font=font_small)
        y += 25
        draw.text((70, y), "Report Classification: Internal Use Only", fill=(0, 0, 0), font=font_small)
        
        return image

def main():
    parser = argparse.ArgumentParser(description="Advanced OCR Inference for Quantized Kosmos-2.5")
    parser.add_argument
