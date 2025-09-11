"""
Working Markdown Generation for Quantized Kosmos-2.5
Converts documents to structured markdown using only working generation methods
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

class WorkingMarkdownGenerator:
    """Markdown generation using only verified working methods"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_quantized_model(self):
        """Load quantized model"""
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
        
        print(f"✅ Model loaded: {type(self.model).__name__}")
        
        # Check generation capabilities
        has_lm_gen = hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'generate')
        has_direct_gen = hasattr(self.model, 'generate')
        
        print(f"   Language model generate: {'✅' if has_lm_gen else '❌'}")
        print(f"   Direct model generate: {'✅' if has_direct_gen else '❌'}")
        
        if not (has_lm_gen or has_direct_gen):
            raise ValueError("No generation methods available!")
        
        return True
    
    def prepare_inputs_with_dtype_conversion(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare inputs with correct dtype conversion"""
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
    
    def generate_text_with_multiple_prompts(self, image: Image.Image) -> str:
        """Try multiple prompts to get best markdown-formatted text"""
        
        # Prompts that may work better for markdown
        prompts_to_try = [
            "<md>",
            "<ocr>",
            "Convert to markdown:",
            "Extract structured text:",
            "Document content:"
        ]
        
        best_result = ""
        best_prompt = ""
        
        for prompt in prompts_to_try:
            print(f"    Trying prompt: '{prompt}'")
            
            # Try language model first (usually better)
            result = self.generate_via_language_model(image, prompt)
            if len(result) > len(best_result):
                best_result = result
                best_prompt = f"{prompt} (language_model)"
            
            # Try direct generation as backup
            if len(result) < 20:  # If language model didn't work well
                result = self.generate_via_direct_model(image, prompt)
                if len(result) > len(best_result):
                    best_result = result
                    best_prompt = f"{prompt} (direct)"
            
            # Stop if we got a decent result
            if len(best_result) > 100:
                break
        
        print(f"    Best result: {best_prompt} ({len(best_result)} chars)")
        return best_result
    
    def generate_via_language_model(self, image: Image.Image, prompt: str) -> str:
        """Generate using language_model.generate() - PREFERRED"""
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
                    max_new_tokens=512,  # Allow longer text for documents
                    do_sample=True,     # Add some creativity for formatting
                    temperature=0.7,
                    top_p=0.9,
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
        """Generate using model.generate() - FALLBACK"""
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
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
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
        """Clean generated text"""
        if not raw_text:
            return ""
        
        # Remove prompt from output
        cleaned = raw_text.replace(prompt, "").strip()
        
        # Remove XML/HTML tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def convert_to_structured_markdown(self, raw_text: str) -> str:
        """Convert text to structured markdown format"""
        if not raw_text:
            return ""
        
        lines = raw_text.split('\n')
        markdown_lines = []
        
        in_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    markdown_lines.append("")
                    in_list = False
                continue
            
            # Apply markdown formatting heuristics
            if self.looks_like_main_title(line):
                markdown_lines.append(f"# {line}")
                markdown_lines.append("")
                in_list = False
            elif self.looks_like_section_header(line):
                markdown_lines.append(f"## {line}")
                markdown_lines.append("")
                in_list = False
            elif self.looks_like_subsection(line):
                markdown_lines.append(f"### {line}")
                in_list = False
            elif self.looks_like_bullet_point(line):
                clean_bullet = self.clean_bullet_text(line)
                markdown_lines.append(f"- {clean_bullet}")
                in_list = True
            elif self.looks_like_numbered_item(line):
                markdown_lines.append(line)  # Keep as-is
                in_list = True
            else:
                if in_list:
                    markdown_lines.append("")
                    in_list = False
                # Apply inline formatting
                formatted_line = self.apply_inline_formatting(line)
                markdown_lines.append(formatted_line)
        
        # Join and clean up
        result = '\n'.join(markdown_lines)
        
        # Remove excessive blank lines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def looks_like_main_title(self, line: str) -> bool:
        """Check if line should be main title (H1)"""
        return (
            len(line) < 60 and
            (line.isupper() or
             any(word in line.lower() for word in [
                 'invoice', 'report', 'document', 'statement', 'summary', 'quarterly'
             ]) or
             (not ':' in line and len(line.split()) <= 5))
        )
    
    def looks_like_section_header(self, line: str) -> bool:
        """Check if line should be section header (H2)"""
        return (
            len(line) < 50 and
            (line.endswith(':') or
             any(word in line.lower() for word in [
                 'section', 'details', 'information', 'services', 'items', 
                 'total', 'summary', 'contact', 'billing', 'payment'
             ]))
        )
    
    def looks_like_subsection(self, line: str) -> bool:
        """Check if line should be subsection header (H3)"""
        return (
            len(line) < 40 and
            any(word in line.lower() for word in [
                'key', 'main', 'important', 'note', 'terms'
            ])
        )
    
    def looks_like_bullet_point(self, line: str) -> bool:
        """Check if line should be bullet point"""
        return (
            line.startswith(('•', '-', '*', '·')) or
            re.match(r'^[•\-\*·]\s', line) or
            (len(line) < 150 and ' - ' in line)
        )
    
    def looks_like_numbered_item(self, line: str) -> bool:
        """Check if line is already numbered"""
        return re.match(r'^\d+\.', line)
    
    def clean_bullet_text(self, line: str) -> str:
        """Clean bullet point text"""
        cleaned = re.sub(r'^[•\-\*·]\s*', '', line)
        cleaned = re.sub(r'^\s*-\s*', '', cleaned)
        return cleaned.strip()
    
    def apply_inline_formatting(self, line: str) -> str:
        """Apply inline markdown formatting"""
        # Bold for currency amounts
        line = re.sub(r'(\$[\d,]+\.?\d*)', r'**\1**', line)
        
        # Italic for dates
        line = re.sub(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', r'*\1*', line)
        
        # Code formatting for emails
        line = re.sub(r'(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)', r'`\1`', line)
        
        # Code formatting for phone numbers
        line = re.sub(r'(\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4})', r'`\1`', line)
        
        return line
    
    def run_markdown_generation(self, image: Image.Image) -> Dict:
        """Run complete markdown generation process"""
        print(f"\n📝 Running Markdown Generation (Generation-Only Methods)")
        print(f"⚠️  SKIPPING forward pass logits")
        
        start_time = time.time()
        
        # Extract text using multiple prompts
        print("🔍 Extracting text with multiple prompts...")
        raw_text = self.generate_text_with_multiple_prompts(image)
        
        extraction_time = time.time() - start_time
        
        # Convert to markdown if we got text
        if raw_text:
            print("🎨 Converting to structured markdown...")
            markdown_text = self.convert_to_structured_markdown(raw_text)
            
            # Final cleanup
            markdown_text = self.finalize_markdown_formatting(markdown_text)
        else:
            markdown_text = ""
        
        total_time = time.time() - start_time
        
        return {
            "success": len(markdown_text) > 20,
            "raw_text": raw_text,
            "markdown_text": markdown_text,
            "extraction_time": extraction_time,
            "total_time": total_time,
            "text_length": len(markdown_text),
            "approach": "generation_only",
            "skipped_methods": ["forward_pass_logits"]
        }
    
    def finalize_markdown_formatting(self, text: str) -> str:
        """Apply final markdown formatting touches"""
        if not text:
            return ""
        
        # Ensure proper spacing around headers
        text = re.sub(r'(^|\n)(#{1,6})\s*([^\n]+)', r'\1\2 \3\n', text)
        
        # Ensure proper list formatting
        text = re.sub(r'\n-\s+', r'\n- ', text)
        
        # Clean up excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def create_test_business_report_image(self) -> Image.Image:
        """Create a structured business report for testing"""
        image = Image.new('RGB', (700, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 28)
            font_header = ImageFont.truetype("arial.ttf", 20)
            font_text = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_title = ImageFont.load_default()
            font_header = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw structured content
        y = 40
        draw.text((50, y), "Q3 2024 BUSINESS PERFORMANCE REPORT", fill=(0, 0, 0), font=font_title)
        y += 70
        
        draw.text((50, y), "Executive Summary", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "This quarter demonstrated outstanding growth across", fill=(0, 0, 0), font=font_text)
        y += 25
        draw.text((70, y), "all key performance indicators and business metrics.", fill=(0, 0, 0), font=font_text)
        y += 45
        
        draw.text((50, y), "Key Performance Metrics", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "• Total Revenue: $3.2M (+22% YoY)", fill=(0, 0, 0), font=font_text)
        y += 28
        draw.text((70, y), "• New Customer Acquisitions: 1,847", fill=(0, 0, 0), font=font_text)
        y += 28
        draw.text((70, y), "• Customer Retention Rate: 96.2%", fill=(0, 0, 0), font=font_text)
        y += 28
        draw.text((70, y), "• Employee Satisfaction: 91%", fill=(0, 0, 0), font=font_text)
        y += 45
        
        draw.text((50, y), "Strategic Initiatives", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "1. Digital transformation program launch", fill=(0, 0, 0), font=font_text)
        y += 28
        draw.text((70, y), "2. International market expansion", fill=(0, 0, 0), font=font_text)
        y += 28
        draw.text((70, y), "3. AI-powered customer service implementation", fill=(0, 0, 0), font=font_text)
        y += 45
        
        draw.text((50, y), "Contact Information", fill=(0, 0, 0), font=font_header)
        y += 35
        draw.text((70, y), "Email: reports@company.com", fill=(0, 0, 0), font=font_small)
        y += 25
        draw.text((70, y), "Phone: (555) 234-5678", fill=(0, 0, 0), font=font_small)
        y += 25
        draw.text((70, y), "Report Date: September 11, 2024", fill=(0, 0, 0), font=font_small)
        
        return image

def main():
    parser = argparse.ArgumentParser(description="Working Markdown Generation for Quantized Kosmos-2.5")