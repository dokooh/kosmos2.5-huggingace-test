"""
Enhanced Markdown generation with proper text generation for Kosmos-2.5
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

class EnhancedKosmos25Markdown:
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
    
    def prepare_inputs(self, image, prompt="<md>"):
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
    
    def generate_markdown(self, image, max_new_tokens=1024):
        """Generate markdown using the model's generation capabilities"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print("Generating markdown...")
        
        try:
            # Try different markdown prompts
            prompts = ["<md>", "<ocr>"]  # Sometimes OCR works better for structured content
            
            for prompt in prompts:
                print(f"Trying prompt: {prompt}")
                
                # Prepare inputs
                inputs = self.prepare_inputs(image, prompt)
                
                # Get input_ids for generation
                input_ids = inputs.get("input_ids")
                if input_ids is None:
                    continue
                
                # Generation parameters optimized for markdown
                generation_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_beams": 1,
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,
                }
                
                # Add other inputs
                for key, value in inputs.items():
                    if key not in generation_kwargs and key != "input_ids":
                        generation_kwargs[key] = value
                
                start_time = time.time()
                
                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(**generation_kwargs)
                
                generation_time = time.time() - start_time
                
                # Decode the generated text
                input_length = input_ids.shape[1]
                generated_text_ids = generated_ids[:, input_length:]
                
                generated_text = self.processor.tokenizer.decode(
                    generated_text_ids[0], 
                    skip_special_tokens=True
                ).strip()
                
                print(f"Generated text length: {len(generated_text)}")
                
                if generated_text:
                    print(f"✓ Success with prompt: {prompt}")
                    return {
                        "success": True,
                        "generated_text": generated_text,
                        "generation_time": generation_time,
                        "prompt_used": prompt,
                        "output_length": generated_text_ids.shape[1]
                    }
                else:
                    print(f"Empty result with prompt: {prompt}")
            
            # If all prompts failed
            return {
                "success": False,
                "error": "No markdown could be generated with any prompt",
                "generated_text": "",
                "generation_time": 0
            }
            
        except Exception as e:
            print(f"✗ Markdown generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "generated_text": "",
                "generation_time": 0
            }
    
    def post_process_markdown(self, raw_text):
        """Post-process generated text to improve markdown formatting"""
        if not raw_text:
            return ""
        
        # Remove special tokens
        cleaned_text = raw_text.replace("<md>", "").replace("</md>", "")
        cleaned_text = raw_text.replace("<ocr>", "").replace("</ocr>", "")
        cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
        
        # Split into lines and process
        lines = cleaned_text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Apply markdown formatting heuristics
            if self.is_title(line):
                processed_lines.append(f"# {line}")
            elif self.is_header(line):
                processed_lines.append(f"## {line}")
            elif self.is_bullet_point(line):
                processed_lines.append(f"- {self.clean_bullet_point(line)}")
            elif self.is_numbered_item(line):
                processed_lines.append(line)  # Keep as-is
            else:
                processed_lines.append(line)
            
            # Add spacing after headers
            if line.startswith('#'):
                processed_lines.append("")
        
        # Join and clean up
        result = '\n'.join(processed_lines)
        
        # Clean up excessive whitespace
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def is_title(self, line):
        """Check if line should be formatted as title"""
        return (
            len(line) < 50 and
            (line.isupper() or 
             any(word in line.lower() for word in ['invoice', 'receipt', 'document', 'report', 'title']) or
             ':' not in line)
        )
    
    def is_header(self, line):
        """Check if line should be formatted as header"""
        return (
            len(line) < 30 and
            (line.endswith(':') or 
             any(word in line.lower() for word in ['section', 'details', 'information', 'summary']))
        )
    
    def is_bullet_point(self, line):
        """Check if line should be formatted as bullet point"""
        return (
            line.startswith(('•', '-', '*')) or
            re.match(r'^[•\-\*]\s', line) or
            (len(line) < 100 and any(word in line.lower() for word in ['product', 'item', 'service']))
        )
    
    def is_numbered_item(self, line):
        """Check if line is already a numbered item"""
        return re.match(r'^\d+\.', line)
    
    def clean_bullet_point(self, line):
        """Clean bullet point text"""
        # Remove existing bullet characters
        cleaned = re.sub(r'^[•\-\*]\s*', '', line)
        return cleaned.strip()
    
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
            print("Creating default structured document...")
            from PIL import ImageDraw, ImageFont
            
            # Create a structured document for markdown testing
            image = Image.new('RGB', (700, 500), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            try:
                font_title = ImageFont.truetype("arial.ttf", 28)
                font_header = ImageFont.truetype("arial.ttf", 20)
                font_text = ImageFont.truetype("arial.ttf", 16)
            except:
                font_title = ImageFont.load_default()
                font_header = ImageFont.load_default()
                font_text = ImageFont.load_default()
            
            # Draw structured content
            y = 30
            draw.text((50, y), "QUARTERLY REPORT", fill=(0, 0, 0), font=font_title)
            y += 50
            
            draw.text((50, y), "Executive Summary", fill=(0, 0, 0), font=font_header)
            y += 35
            draw.text((70, y), "This quarter showed significant growth across all metrics.", fill=(0, 0, 0), font=font_text)
            y += 25
            draw.text((70, y), "Key achievements include improved efficiency and customer satisfaction.", fill=(0, 0, 0), font=font_text)
            y += 40
            
            draw.text((50, y), "Key Metrics", fill=(0, 0, 0), font=font_header)
            y += 35
            draw.text((70, y), "• Revenue: $2.5M (+15%)", fill=(0, 0, 0), font=font_text)
            y += 25
            draw.text((70, y), "• Customer Growth: 1,200 new customers", fill=(0, 0, 0), font=font_text)
            y += 25
            draw.text((70, y), "• Satisfaction Score: 4.8/5.0", fill=(0, 0, 0), font=font_text)
            y += 40
            
            draw.text((50, y), "Next Steps", fill=(0, 0, 0), font=font_header)
            y += 35
            draw.text((70, y), "1. Expand marketing efforts", fill=(0, 0, 0), font=font_text)
            y += 25
            draw.text((70, y), "2. Improve product features", fill=(0, 0, 0), font=font_text)
            y += 25
            draw.text((70, y), "3. Hire additional staff", fill=(0, 0, 0), font=font_text)
            
            return image
        
        else:
            raise ValueError(f"Invalid image input: {image_input}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced markdown generation using FP4 quantized Kosmos-2.5")
    parser.add_argument("--model_path", type=str, default="./fp4_quantized_model",
                       help="Path to saved FP4 quantized model")
    parser.add_argument("--image", type=str, default="default",
                       help="Image path, URL, or 'default' for test image")
    parser.add_argument("--output", type=str,
                       help="Output markdown file")
    parser.add_argument("--output_json", type=str,
                       help="Output JSON file with full results")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize markdown generator
    markdown_gen = EnhancedKosmos25Markdown(model_path=args.model_path)
    
    try:
        # Load model
        markdown_gen.load_model()
        
        # Load image
        image = markdown_gen.load_image(args.image)
        print(f"Image size: {image.size}")
        
        # Generate markdown
        result = markdown_gen.generate_markdown(image, max_new_tokens=args.max_tokens)
        
        # Display results
        print(f"\n{'='*60}")
        print("ENHANCED MARKDOWN GENERATION RESULTS")
        print('='*60)
        
        if result["success"]:
            raw_text = result.get("generated_text", "")
            processed_markdown = markdown_gen.post_process_markdown(raw_text)
            
            print(f"✓ Markdown generation successful")
            print(f"  Generation time: {result.get('generation_time', 0):.2f}s")
            print(f"  Prompt used: {result.get('prompt_used', 'Unknown')}")
            print(f"  Output tokens: {result.get('output_length', 0)}")
            
            if processed_markdown:
                print(f"\nGenerated Markdown:")
                print("-" * 40)
                print(processed_markdown)
                
                print(f"\nRaw Generated Text:")
                print("-" * 40)
                print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
                
                # Save markdown file
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(processed_markdown)
                    print(f"\nMarkdown saved to: {args.output}")
            else:
                print("\n⚠️  No markdown could be generated from the image")
                print("Possible reasons:")
                print("- The image may not contain structured text")
                print("- Different generation parameters might be needed")
                print("- The image preprocessing may need adjustment")
        else:
            print(f"✗ Markdown generation failed: {result.get('error', 'Unknown error')}")
        
        # Save JSON results
        if args.output_json:
            output_data = {
                "image_source": args.image,
                "model_path": args.model_path,
                "result": result,
                "processed_markdown": markdown_gen.post_process_markdown(result.get("generated_text", "")),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {args.output_json}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
