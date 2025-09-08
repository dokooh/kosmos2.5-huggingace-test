#!/usr/bin/env python3
"""
Optimized OCR Inference Module for Kosmos-2.5 with FP16 Quantization

This module provides fast OCR inference using FP16 quantized Kosmos-2.5 model.
Features:
- FP16 quantization for faster inference
- Optimized memory usage
- Batch processing support
- Enhanced error handling
"""

import re
import torch
import requests
import argparse
import sys
import os
import time
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOCRInference:
    def __init__(self, model_name="microsoft/kosmos-2.5", device=None, cache_dir=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        
        logger.info(f"Initializing OCR inference on {self.device} with {self.dtype}")
        
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
    
    def post_process_ocr(self, generated_text, scale_height, scale_width, prompt="<ocr>"):
        """Post-process OCR results to extract bounding boxes and text"""
        text = generated_text.replace(prompt, "").strip()
        
        # Pattern to match bounding boxes
        pattern = r"<bbox><x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)></bbox>"
        bbox_matches = re.finditer(pattern, text)
        
        # Split text by bounding boxes
        text_parts = re.split(pattern, text)
        
        results = []
        bbox_index = 0
        
        for match in re.finditer(pattern, text):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, match.groups())
            
            # Scale coordinates to original image size
            x1_scaled = int(x1 * scale_width)
            y1_scaled = int(y1 * scale_height)
            x2_scaled = int(x2 * scale_width)
            y2_scaled = int(y2 * scale_height)
            
            # Get corresponding text (next non-empty part after bbox)
            text_start = match.end()
            next_bbox = re.search(pattern, text[text_start:])
            
            if next_bbox:
                extracted_text = text[text_start:text_start + next_bbox.start()].strip()
            else:
                extracted_text = text[text_start:].strip()
            
            if extracted_text and not (x1_scaled >= x2_scaled or y1_scaled >= y2_scaled):
                results.append({
                    'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                    'text': extracted_text,
                    'confidence': 1.0  # Kosmos doesn't provide confidence scores
                })
        
        return results
    
    def draw_bounding_boxes(self, image, ocr_results, output_path=None):
        """Draw bounding boxes and text on image"""
        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i, result in enumerate(ocr_results):
            bbox = result['bbox']
            text = result['text']
            
            # Draw bounding box
            draw.rectangle(bbox, outline="red", width=2)
            
            # Draw text label
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text above the bounding box
            text_x = bbox[0]
            text_y = max(0, bbox[1] - text_height - 5)
            
            # Draw background for text
            draw.rectangle(
                [text_x, text_y, text_x + text_width, text_y + text_height],
                fill="red", outline="red"
            )
            draw.text((text_x, text_y), text, fill="white", font=font)
        
        if output_path:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            annotated_image.save(output_path)
            logger.info(f"Annotated image saved to: {output_path}")
        
        return annotated_image
    
    def perform_ocr(self, image_path, max_tokens=1024, save_image=None, save_text=None):
        """Perform OCR on image and return structured results"""
        if self.model is None:
            self.load_model()
        
        # Load and process image
        image = self.load_image(image_path)
        
        prompt = "<ocr>"
        start_time = time.time()
        
        try:
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Extract scaling information
            height = inputs.pop("height")
            width = inputs.pop("width")
            raw_width, raw_height = image.size
            scale_height = raw_height / height
            scale_width = raw_width / width
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            # Convert flattened_patches to correct dtype
            if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            
            # Generate OCR results
            logger.info("Performing OCR inference...")
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process to extract structured data
            ocr_results = self.post_process_ocr(generated_text, scale_height, scale_width, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"OCR completed in {inference_time:.2f}s. Found {len(ocr_results)} text regions.")
            
            # Save outputs if requested
            if save_text:
                self.save_text_results(ocr_results, save_text)
            
            if save_image:
                self.draw_bounding_boxes(image, ocr_results, save_image)
            
            return {
                'results': ocr_results,
                'inference_time': inference_time,
                'raw_output': generated_text
            }
            
        except Exception as e:
            logger.error(f"Error during OCR inference: {e}")
            raise
    
    def save_text_results(self, ocr_results, output_path):
        """Save OCR results as structured text"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("OCR Results\n")
                f.write("=" * 50 + "\n\n")
                
                for i, result in enumerate(ocr_results, 1):
                    bbox = result['bbox']
                    text = result['text']
                    f.write(f"Region {i}:\n")
                    f.write(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n")
                    f.write(f"  Text: {text}\n\n")
            
            logger.info(f"OCR text results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving text results: {e}")

def get_args():
    parser = argparse.ArgumentParser(description='Optimized OCR inference using FP16 quantized Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--output_image', '-o', type=str, default='./ocr_output.png',
                       help='Output path for annotated image')
    parser.add_argument('--output_text', '-t', type=str, default=None,
                       help='Output path for OCR text results')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum tokens to generate')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--no_image_output', action='store_true',
                       help='Skip saving annotated image')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Initialize OCR inference
    ocr_engine = OptimizedOCRInference(
        device=args.device,
        cache_dir=args.cache_dir
    )
    
    try:
        # Perform OCR
        results = ocr_engine.perform_ocr(
            image_path=args.image,
            max_tokens=args.max_tokens,
            save_image=None if args.no_image_output else args.output_image,
            save_text=args.output_text
        )
        
        # Print results summary
        print(f"\n{'='*60}")
        print("OCR RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Processing time: {results['inference_time']:.2f}s")
        print(f"Text regions found: {len(results['results'])}")
        print(f"{'='*60}")
        
        for i, result in enumerate(results['results'], 1):
            print(f"Region {i}: {result['text']}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"OCR inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()