"""
Working Markdown Generation for Quantized Kosmos-2.5
Converts structured documents to markdown using only working generation methods
"""

import torch
import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, List

from transformers import AutoModel, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import re

class WorkingMarkdownGenerator:
    """Markdown generator using only verified working methods"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_quantized_model(self):
        """Load quantized model and validate capabilities"""
        print(f"🔄 Loading quantized model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
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
        
        # Validate generation methods
        has_lm_gen = hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'generate')
        has_direct_gen = hasattr(self.model, 'generate')
        
        print(f"    🔍 Language model generate: {'✅' if has_lm_gen else '❌'}")
        print(f"    🔍 Direct model generate: {'✅' if has_direct_gen else '❌'}")
        
        if not (has_lm_gen or has_direct_gen):
            raise ValueError("No generation methods available!")
        
        return True
    
    def prepare_inputs_with_dtype_conversion(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs with proper dtype conversion for quantized models"""
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
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
    
    def extract_text_with_multiple_prompts(self, image: Image.Image) -> str:
        """Try multiple prompts to get best text extraction for markdown conversion"""
        
        # Prompts that might work better for different document types
        prompts_to_try = [
            "<md>",           # Direct markdown prompt
            "<ocr>",          # Standard OCR prompt
            "Convert to markdown:",
            "Extract document structure:",
            "Document content:",
            "Text from image:"
        ]
        
        best_result = ""
        best_prompt = ""
        
        for prompt in prompts_to_try:
            print(f"        Trying prompt: '{prompt}'")
            
            # Try language model first (usually better quality)
            result = self.generate_via_language_model(image, prompt)
            if len(result) > len(best_result):
                best_result = result
                best_prompt = f"{prompt} (language_model)"
            
            # Try direct generation if language model didn't produce much
            if len(result) < 30:  # If language model result is poor
                result = self.generate_via_direct_model(image, prompt)
                if len(result) > len(best_result):
                    best_result = result
                    best_prompt = f"{prompt} (direct)"
            
            # Stop if we got a substantial result
            if len(best_result) > 100:
                print(f"        ✅ Good result found with: {best_prompt}")
                break
        
        print(f"        📊 Best extraction: {len(best_result)} chars from {best_prompt}")
        return best_result
    
    def generate_via_language_model(self, image: Image.Image, prompt: str) -> str:
        """Generate using language_model.generate() - PREFERRED METHOD"""
        if not hasattr(self.model, 'language_model') or not hasattr(self.model.language_model, 'generate'):
            return ""
        
        try:
            inputs = self.prepare_inputs_with_dtype_conversion(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                return ""
            
            with torch.no_grad():
                generated_ids = self.model.language_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=600,  # Allow longer text for documents
                    do_sample=True,     # Add some creativity for better formatting
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Extract new tokens only
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                return self.clean_generated_text(generated_text, prompt)
            
            return ""
        
        except Exception:
            return ""
    
    def generate_via_direct_model(self, image: Image.Image, prompt: str) -> str:
        """Generate using model.generate() - FALLBACK METHOD"""
        if not hasattr(self.model, 'generate'):
            return ""
        
        try:
            inputs = self.prepare_inputs_with_dtype_conversion(image, prompt)
            input_ids = inputs.get("input_ids")
            
            if input_ids is None:
                return ""
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=600,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    **{k: v for k, v in inputs.items() if k != "input_ids"}
                )
            
            # Extract new tokens only
            input_length = input_ids.shape[1]
            new_tokens = generated_ids[:, input_length:]
            
            if new_tokens.shape[1] > 0:
                generated_text = self.processor.tokenizer.decode(
                    new_tokens[0], 
                    skip_special_tokens=True
                ).strip()
                
                return self.clean_generated_text(generated_text, prompt)
            
            return ""
        
        except Exception:
            return ""
    
    def clean_generated_text(self, raw_text: str, prompt: str) -> str:
        """Clean generated text and remove artifacts"""
        if not raw_text:
            return ""
        
        # Remove prompt from output
        cleaned = raw_text.replace(prompt, "").strip()
        
        # Remove XML/HTML tags that sometimes appear
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Normalize whitespace but preserve line structure
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n[ \t]*\n', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def convert_text_to_structured_markdown(self, raw_text: str) -> str:
        """Convert extracted text to well-structured markdown"""
        if not raw_text:
            return ""
        
        lines = raw_text.split('\n')
        markdown_lines = []
        
        in_list_context = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_list_context:
                    markdown_lines.append("")  # End list context
                    in_list_context = False
                continue
            
            # Apply markdown formatting rules
            formatted_line = None
            
            # Check for main title (H1)
            if self.is_main_title(line):
                formatted_line = f"# {line}"
                markdown_lines.append(formatted_line)
                markdown_lines.append("")  # Add spacing
                in_list_context = False
            
            # Check for section header (H2)
            elif self.is_section_header(line):
                formatted_line = f"## {line}"
                markdown_lines.append(formatted_line)
                markdown_lines.append("")  # Add spacing
                in_list_context = False
            
            # Check for subsection (H3)
            elif self.is_subsection_header(line):
                formatted_line = f"### {line}"
                markdown_lines.append(formatted_line)
                in_list_context = False
            
            # Check for bullet points
            elif self.is_bullet_point(line):
                clean_bullet = self.clean_bullet_text(line)
                formatted_line = f"- {clean_bullet}"
                markdown_lines.append(formatted_line)
                in_list_context = True
            
            # Check for numbered items
            elif self.is_numbered_item(line):
                formatted_line = line  # Keep numbered items as-is
                markdown_lines.append(formatted_line)
                in_list_context = True
            
            # Regular paragraph text
            else:
                if in_list_context:
                    markdown_lines.append("")  # Add spacing before paragraph
                    in_list_context = False
                
                # Apply inline formatting
                formatted_line = self.apply_inline_markdown_formatting(line)
                markdown_lines.append(formatted_line)
        
        # Join and clean up
        result = '\n'.join(markdown_lines)
        
        # Remove excessive blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def is_main_title(self, line: str) -> bool:
        """Detect if line should be main title (H1)"""
        return (
            len(line) < 80 and
            (line.isupper() or
             any(word in line.lower() for word in [
                 'invoice', 'report', 'document', 'statement', 'summary', 
                 'quarterly', 'annual', 'monthly',
