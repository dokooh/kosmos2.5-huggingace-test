#!/usr/bin/env python3
"""
Optimized Markdown Generation Inference Module for Kosmos-2.5 with FP16 Quantization

This module provides fast markdown generation using FP16 quantized Kosmos-2.5 model.
Features:
- FP16 quantization for faster inference
- Optimized memory usage
- Enhanced markdown post-processing
- Batch processing support
"""

import torch
import requests
import argparse
import sys
import os
import time
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMarkdownInference:
    def __init__(self, model_name="microsoft/kosmos-2.5", device=None, cache_dir=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        
        logger.info(f"Initializing Markdown inference on {self.device} with {self.dtype}")
    
    def load_model(self):
        """Load FP16 quantized model for optimal performance"""
        if self.model is not None:
            return
            
        logger.info("Loading FP16 quantized Kosmos-2.5 model...")
        try:
            # Load with FP16 quantization for speed
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto" if self.device.startswith('cuda') else None,
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Ensure model is in FP16 for CUDA
            if self.device.startswith('cuda'):
                self.model = self.model.half()
                
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_image(self, image_path):
        """Load image from local path or URL with error handling"""
        try:
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Loading image from URL: {image_path}")
                response = requests.get(image_path, stream=True, timeout=30)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                logger.info(f"Loading image from file: {image_path}")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
            
            # Convert to RGB and validate
            image = image.convert('RGB')
            logger.info(f"Image loaded successfully. Size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def post_process_markdown(self, generated_text, prompt="<md>"):
        """Post-process and clean up generated markdown"""
        # Remove the prompt
        markdown = generated_text.replace(prompt, "").strip()
        
        # Clean up common issues
        markdown = self.clean_markdown(markdown)
        
        return markdown
    
    def clean_markdown(self, text):
        """Clean and format markdown text"""
        # Remove extra whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines initially
                cleaned_lines.append(line)
        
        # Join lines back
        text = '\n'.join(cleaned_lines)
        
        # Fix common markdown formatting issues
        # Fix headers
        text = re.sub(r'^#{1,6}\s*', lambda m: m.group(0).rstrip() + ' ', text, flags=re.MULTILINE)
        
        # Ensure proper spacing around headers
        text = re.sub(r'(^#{1,6}.*$)', r'\n\1\n', text, flags=re.MULTILINE)
        
        # Fix list items
        text = re.sub(r'^[\*\-\+]\s+', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', lambda m: m.group(0), text, flags=re.MULTILINE)
        
        # Fix table formatting
        text = re.sub(r'\|([^|]+)\|', lambda m: '|' + m.group(1).strip() + '|', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure text starts and ends cleanly
        text = text.strip()
        
        return text
    
    def generate_markdown(self, image_path, max_tokens=2048, temperature=0.1, save_output=None):
        """Generate markdown from image"""
        if self.model is None:
            self.load_model()
        
        # Load and process image
        image = self.load_image(image_path)
        
        prompt = "<md>"
        start_time = time.time()
        
        try:
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Remove height/width info (not needed for generation)
            inputs.pop("height", None)
            inputs.pop("width", None)
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            # Convert flattened_patches to correct dtype
            if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            
            # Generate markdown
            logger.info("Generating markdown...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process markdown
            markdown_output = self.post_process_markdown(generated_text, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"Markdown generation completed in {inference_time:.2f}s")
            
            # Save output if requested
            if save_output:
                self.save_markdown(markdown_output, save_output)
            
            return {
                'markdown': markdown_output,
                'inference_time': inference_time,
                'raw_output': generated_text,
                'word_count': len(markdown_output.split()),
                'char_count': len(markdown_output)
            }
            
        except Exception as e:
            logger.error(f"Error during markdown generation: {e}")
            raise
    
    def save_markdown(self, markdown_text, output_path):
        """Save markdown to file"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            logger.info(f"Markdown saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving markdown: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=2048, temperature=0.1):
        """Process multiple images in batch"""
        if self.model is None:
            self.load_model()
        
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")
            
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_markdown.md")
                
                # Generate markdown
                result = self.generate_markdown(
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    save_output=output_path
                )
                
                result['input_path'] = image_path
                result['output_path'] = output_path
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e),
                    'inference_time': 0
                })
        
        return results

def get_args():
    parser = argparse.ArgumentParser(description='Optimized Markdown generation using FP16 quantized Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--output', '-o', type=str, default='./output.md',
                       help='Output path for generated markdown')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--max_tokens', '-m', type=int, default=2048,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Sampling temperature (0 for deterministic)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images (image should be a directory)')
    parser.add_argument('--print_output', '-p', action='store_true',
                       help='Print generated markdown to console')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Initialize markdown inference
    md_engine = OptimizedMarkdownInference(
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    try:
        if args.batch and os.path.isdir(args.image):
            # Batch processing
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_paths = [
                os.path.join(args.image, f) for f in os.listdir(args.image)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_paths:
                logger.error(f"No images found in directory: {args.image}")
                sys.exit(1)
            
            logger.info(f"Processing {len(image_paths)} images in batch mode")
            results = md_engine.batch_process(
                image_paths=image_paths,
                output_dir=args.output,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Print summary
            successful = sum(1 for r in results if 'error' not in r)
            total_time = sum(r.get('inference_time', 0) for r in results)
            
            print(f"\n{'='*60}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(results):.2f}s")
            print(f"{'='*60}")
            
        else:
            # Single image processing
            result = md_engine.generate_markdown(
                image_path=args.image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_output=args.output
            )
            
            # Print results summary
            print(f"\n{'='*60}")
            print("MARKDOWN GENERATION SUMMARY")
            print(f"{'='*60}")
            print(f"Processing time: {result['inference_time']:.2f}s")
            print(f"Word count: {result['word_count']}")
            print(f"Character count: {result['char_count']}")
            print(f"Output saved to: {args.output}")
            print(f"{'='*60}")
            
            if args.print_output:
                print("\nGENERATED MARKDOWN:")
                print("=" * 60)
                print(result['markdown'])
                print("=" * 60)
        
    except Exception as e:
        logger.error(f"Markdown generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()