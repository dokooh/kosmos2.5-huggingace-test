#!/usr/bin/env python3
"""
8-bit Mixed Precision OCR Inference for Kosmos-2.5

This module provides fast OCR inference using 8-bit mixed precision quantized Kosmos-2.5 model.
Features:
- 8-bit mixed precision quantization for faster inference and reduced memory usage
- BitsAndBytesConfig for advanced quantization settings
- SafeTensors format support for faster loading
- Support for local model checkpoints and remote models
- Optimized memory usage with gradient checkpointing
- Enhanced OCR post-processing with text region extraction
- Batch processing support with progress tracking
- Comprehensive error handling and fallback mechanisms
"""

import re
import torch
import requests
import argparse
import sys
import os
import time
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EightBitOCRInference:
    def __init__(self, model_checkpoint, device=None, cache_dir=None, use_8bit=True, mixed_precision=True):
        """
        Initialize 8-bit Mixed Precision OCR inference
        
        Args:
            model_checkpoint (str): Path to model checkpoint (local directory or HuggingFace model name)
            device (str): Device to use for inference
            cache_dir (str): Cache directory for downloaded models
            use_8bit (bool): Enable 8-bit quantization
            mixed_precision (bool): Enable mixed precision for critical layers
        """
        self.model_checkpoint = model_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit and torch.cuda.is_available()
        self.mixed_precision = mixed_precision
        self.model = None
        self.processor = None
        
        # Determine model type
        self.is_local_checkpoint = os.path.exists(model_checkpoint)
        
        # Configure precision settings
        self._configure_precision()
        
        logger.info(f"Initializing 8-bit Mixed Precision OCR inference on {self.device}")
        logger.info(f"Model checkpoint: {self.model_checkpoint}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Local checkpoint: {self.is_local_checkpoint}")
    
    def _configure_precision(self):
        """Configure precision and quantization settings"""
        if self.use_8bit:
            # Configure 8-bit quantization with mixed precision
            if self.mixed_precision:
                logger.info("Configuring 8-bit mixed precision quantization")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_threshold=6.0,
                    # Skip critical layers for better OCR accuracy
                    llm_int8_skip_modules=["lm_head", "embed_tokens", "layernorm", "layer_norm", "norm"]
                )
            else:
                logger.info("Configuring standard 8-bit quantization")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0,
                )
            
            # Use float16 for non-quantized operations
            self.dtype = torch.float16
        else:
            self.quantization_config = None
            # Use bfloat16 for better performance on modern hardware
            if self.device.startswith('cuda') and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                logger.info("Using bfloat16 for optimal performance")
            elif self.device.startswith('cuda'):
                self.dtype = torch.float16
                logger.info("Using float16")
            else:
                self.dtype = torch.float32
                logger.info("Using float32 for CPU")
    
    def _validate_checkpoint(self, checkpoint_path):
        """Validate that the checkpoint contains required files"""
        if not os.path.isdir(checkpoint_path):
            return False
        
        # Check for model files (SafeTensors or PyTorch)
        model_files = [f for f in os.listdir(checkpoint_path) 
                      if f.endswith(('.safetensors', '.bin', '.pt'))]
        config_files = [f for f in os.listdir(checkpoint_path) 
                       if f in ['config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json']]
        
        has_model = len(model_files) > 0
        has_config = len(config_files) > 0
        
        if has_model and has_config:
            logger.info(f"✓ Found {len(model_files)} model files in {checkpoint_path}")
            return True
        else:
            logger.warning(f"⚠ Checkpoint missing required files. Model files: {has_model}, Config: {has_config}")
            return False
    
    def load_model(self):
        """Load model with 8-bit mixed precision quantization"""
        if self.model is not None:
            return
            
        logger.info("Loading Kosmos-2.5 model with 8-bit mixed precision...")
        
        # Validate local checkpoint if specified
        if self.is_local_checkpoint:
            if not self._validate_checkpoint(self.model_checkpoint):
                logger.error(f"Invalid checkpoint directory: {self.model_checkpoint}")
                raise ValueError("Checkpoint path does not contain valid model files")
        
        try:
            # Configure loading parameters
            loading_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            # Add quantization config if using 8-bit
            if self.use_8bit:
                loading_kwargs.update({
                    "quantization_config": self.quantization_config,
                    "device_map": "auto",
                    "torch_dtype": self.dtype,
                })
                logger.info("Loading with 8-bit mixed precision quantization...")
            else:
                loading_kwargs.update({
                    "torch_dtype": self.dtype,
                    "device_map": "auto" if torch.cuda.is_available() else None,
                })
            
            # Configure for local vs remote loading
            if self.is_local_checkpoint:
                loading_kwargs.update({
                    "local_files_only": True,
                    "use_safetensors": True,
                })
                logger.info("Loading from local checkpoint...")
            else:
                loading_kwargs.update({
                    "local_files_only": False,
                    "use_safetensors": True,
                    "resume_download": True,
                })
                logger.info("Loading from HuggingFace Hub...")
            
            # Add Flash Attention if available (better performance)
            try:
                if hasattr(torch.nn, 'scaled_dot_product_attention'):
                    loading_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Enabled Flash Attention 2")
            except:
                logger.debug("Flash Attention 2 not available")
            
            # Load the model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_checkpoint,
                **loading_kwargs
            )
            
            # Apply additional optimizations for non-8bit models
            if not self.use_8bit:
                if self.device.startswith('cuda'):
                    if self.dtype == torch.bfloat16:
                        self.model = self.model.bfloat16()
                    else:
                        self.model = self.model.half()
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for memory efficiency")
            
            # Load processor
            processor_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "use_fast": True,
            }
            
            if self.is_local_checkpoint:
                processor_kwargs["local_files_only"] = True
            else:
                processor_kwargs.update({
                    "local_files_only": False,
                    "resume_download": True
                })
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                **processor_kwargs
            )
            
            # Set pad token if not present
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Enable model optimizations
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Enable torch.compile for PyTorch 2.0+ (if available and not using 8-bit)
            if hasattr(torch, 'compile') and self.device.startswith('cuda') and not self.use_8bit:
                try:
                    logger.info("Compiling model with torch.compile for faster inference...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without it: {e}")
            
            logger.info("✓ Model loaded successfully with 8-bit mixed precision optimizations")
            
            # Print model info
            try:
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model parameters: {num_params:,}")
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU memory available: {gpu_memory:.1f} GB")
                    logger.info(f"GPU memory allocated: {allocated_memory:.1f} GB")
                    logger.info(f"Memory efficiency: {(1 - allocated_memory/gpu_memory)*100:.1f}% free")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to basic loading
            logger.info("Attempting fallback model loading without quantization...")
            try:
                fallback_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True
                }
                
                if self.is_local_checkpoint:
                    fallback_kwargs["local_files_only"] = True
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_checkpoint,
                    **fallback_kwargs
                )
                
                processor_fallback_kwargs = {
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True
                }
                
                if self.is_local_checkpoint:
                    processor_fallback_kwargs["local_files_only"] = True
                
                self.processor = AutoProcessor.from_pretrained(
                    self.model_checkpoint,
                    **processor_fallback_kwargs
                )
                
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                
                logger.info("Fallback model loading successful")
                
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {e2}")
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
        """Enhanced post-process OCR results to extract bounding boxes and text"""
        text = generated_text.replace(prompt, "").strip()
        
        # Enhanced pattern to match bounding boxes with better error handling
        pattern = r"<bbox><x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)></bbox>"
        
        results = []
        processed_positions = set()
        
        for match in re.finditer(pattern, text):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, match.groups())
            
            # Skip if coordinates are invalid or already processed
            bbox_key = (x1, y1, x2, y2)
            if bbox_key in processed_positions:
                continue
            
            # Scale coordinates to original image size
            x1_scaled = int(x1 * scale_width)
            y1_scaled = int(y1 * scale_height)
            x2_scaled = int(x2 * scale_width)
            y2_scaled = int(y2 * scale_height)
            
            # Validate bounding box dimensions
            if x1_scaled >= x2_scaled or y1_scaled >= y2_scaled:
                continue
            
            # Get corresponding text (next non-empty part after bbox)
            text_start = match.end()
            next_bbox = re.search(pattern, text[text_start:])
            
            if next_bbox:
                extracted_text = text[text_start:text_start + next_bbox.start()].strip()
            else:
                extracted_text = text[text_start:].strip()
            
            # Enhanced text cleaning
            extracted_text = self._clean_extracted_text(extracted_text)
            
            if extracted_text:
                # Calculate confidence based on text length and bbox size
                bbox_area = (x2_scaled - x1_scaled) * (y2_scaled - y1_scaled)
                confidence = min(1.0, len(extracted_text) / 50.0 + bbox_area / 10000.0)
                
                results.append({
                    'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                    'text': extracted_text,
                    'confidence': confidence,
                    'raw_coords': [x1, y1, x2, y2]  # Keep original coordinates for debugging
                })
                
                processed_positions.add(bbox_key)
        
        return results
    
    def _clean_extracted_text(self, text):
        """Enhanced text cleaning for OCR results"""
        # Remove HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\-.,!?():;"\'/]', '', text)
        
        # Remove very short or meaningless extractions
        if len(text.strip()) < 2:
            return ""
        
        return text
    
    def draw_bounding_boxes(self, image, ocr_results, output_path=None):
        """Enhanced drawing of bounding boxes and text on image"""
        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to load a font with better size handling
        try:
            font_size = max(12, min(24, int(min(image.size) / 50)))  # Adaptive font size
            
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("arial.ttf", font_size)
            else:  # Linux/Mac
                font_paths = [
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/System/Library/Fonts/Arial.ttf"  # macOS
                ]
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Enhanced color palette
        colors = [
            '#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', 
            '#FF0080', '#00FFFF', '#FFFF00', '#FF8080', '#80FF80'
        ]
        
        for i, result in enumerate(ocr_results):
            bbox = result['bbox']
            text = result['text']
            confidence = result.get('confidence', 1.0)
            
            # Choose color based on confidence
            if confidence > 0.8:
                color = colors[i % len(colors)]
            elif confidence > 0.5:
                color = '#FFA500'  # Orange for medium confidence
            else:
                color = '#FF6B6B'  # Light red for low confidence
            
            # Draw bounding box with thickness based on confidence
            thickness = max(1, int(3 * confidence))
            draw.rectangle(bbox, outline=color, width=thickness)
            
            # Prepare text label with confidence
            display_text = text
            if len(display_text) > 30:
                display_text = display_text[:27] + "..."
            
            # Calculate text dimensions
            try:
                text_bbox = draw.textbbox((0, 0), display_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(display_text, font=font)
            
            # Position text above the bounding box, with fallback positioning
            text_x = bbox[0]
            text_y = max(0, bbox[1] - text_height - 5)
            
            # If text would go off the top, put it inside the box
            if text_y < 0:
                text_y = bbox[1] + 5
            
            # Draw background for text with semi-transparency effect
            background_bbox = [
                text_x - 2, text_y - 2, 
                text_x + text_width + 4, text_y + text_height + 2
            ]
            draw.rectangle(background_bbox, fill=color, outline=color)
            
            # Draw text
            draw.text((text_x, text_y), display_text, fill="white", font=font)
            
            # Draw confidence indicator (small circle)
            if confidence < 1.0:
                circle_size = 8
                circle_x = bbox[2] - circle_size
                circle_y = bbox[1]
                circle_color = '#00FF00' if confidence > 0.8 else '#FFFF00' if confidence > 0.5 else '#FF0000'
                draw.ellipse([circle_x, circle_y, circle_x + circle_size, circle_y + circle_size], 
                           fill=circle_color, outline='white')
        
        if output_path:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            annotated_image.save(output_path, quality=95, optimize=True)
            logger.info(f"Annotated image saved to: {output_path}")
        
        return annotated_image
    
    def perform_ocr(self, image_path, max_tokens=1024, save_image=None, save_text=None):
        """Perform OCR on image using 8-bit mixed precision"""
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
            
            # Move inputs to device and convert to correct dtype if not using 8-bit
            if not self.use_8bit:
                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
                
                # Convert flattened_patches to correct dtype
                if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                    inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            else:
                # For 8-bit models, just move to device (quantization handles dtype)
                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            # Generate OCR results
            logger.info("Performing OCR inference with 8-bit mixed precision...")
            with torch.no_grad():
                # Clear GPU cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use mixed precision if enabled and not using 8-bit quantization
                if self.mixed_precision and not self.use_8bit and self.device.startswith('cuda'):
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=1,  # Faster than beam search for OCR
                            early_stopping=True,
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        early_stopping=True,
                    )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process to extract structured data
            ocr_results = self.post_process_ocr(generated_text, scale_height, scale_width, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"OCR completed in {inference_time:.2f}s. Found {len(ocr_results)} text regions.")
            
            # Calculate additional statistics
            total_text_length = sum(len(result['text']) for result in ocr_results)
            avg_confidence = sum(result['confidence'] for result in ocr_results) / len(ocr_results) if ocr_results else 0
            
            # Save outputs if requested
            if save_text:
                self.save_text_results(ocr_results, save_text)
            
            if save_image:
                self.draw_bounding_boxes(image, ocr_results, save_image)
            
            return {
                'results': ocr_results,
                'inference_time': inference_time,
                'raw_output': generated_text,
                'statistics': {
                    'total_regions': len(ocr_results),
                    'total_text_length': total_text_length,
                    'avg_confidence': avg_confidence,
                    'image_size': image.size
                }
            }
            
        except Exception as e:
            logger.error(f"Error during OCR inference: {e}")
            raise
    
    def save_text_results(self, ocr_results, output_path):
        """Save OCR results as structured text with enhanced formatting"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("8-bit Mixed Precision OCR Results\n")
                f.write("=" * 60 + "\n")
                f.write(f"Model: {self.model_checkpoint}\n")
                f.write(f"Quantization: {'8-bit' if self.use_8bit else 'FP16/BF16'}\n")
                f.write(f"Mixed Precision: {self.mixed_precision}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, result in enumerate(ocr_results, 1):
                    bbox = result['bbox']
                    text = result['text']
                    confidence = result['confidence']
                    
                    f.write(f"Region {i}:\n")
                    f.write(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n")
                    f.write(f"  Confidence: {confidence:.3f}\n")
                    f.write(f"  Text: {text}\n\n")
                
                # Add summary
                total_regions = len(ocr_results)
                avg_confidence = sum(r['confidence'] for r in ocr_results) / total_regions if total_regions > 0 else 0
                
                f.write("=" * 60 + "\n")
                f.write("SUMMARY:\n")
                f.write(f"Total regions: {total_regions}\n")
                f.write(f"Average confidence: {avg_confidence:.3f}\n")
                f.write("=" * 60 + "\n")
            
            logger.info(f"OCR text results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving text results: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=1024):
        """Process multiple images in batch with progress tracking"""
        if self.model is None:
            self.load_model()
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting batch OCR processing of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Generate output filenames
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image = os.path.join(output_dir, f"{base_name}_ocr.png")
                output_text = os.path.join(output_dir, f"{base_name}_ocr.txt")
                
                # Perform OCR
                result = self.perform_ocr(
                    image_path=image_path,
                    max_tokens=max_tokens,
                    save_image=output_image,
                    save_text=output_text
                )
                
                result['input_path'] = image_path
                result['output_image'] = output_image
                result['output_text'] = output_text
                results.append(result)
                
                # Progress update
                progress = (i / len(image_paths)) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{len(image_paths)})")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e),
                    'inference_time': 0,
                    'statistics': {}
                })
        
        logger.info("Batch OCR processing completed!")
        return results

def get_args():
    parser = argparse.ArgumentParser(description='8-bit Mixed Precision OCR inference using Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--model_checkpoint', '-m', type=str, required=True,
                       help='Path to model checkpoint (local directory or HuggingFace model name)')
    parser.add_argument('--output_image', '-o', type=str, default='./ocr_output.png',
                       help='Output path for annotated image')
    parser.add_argument('--output_text', '-t', type=str, default=None,
                       help='Output path for OCR text results')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='Maximum tokens to generate')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--no_8bit', action='store_true',
                       help='Disable 8-bit quantization')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision for critical layers')
    parser.add_argument('--no_image_output', action='store_true',
                       help='Skip saving annotated image')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images (image should be a directory)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output with statistics')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize 8-bit mixed precision OCR inference
    ocr_engine = EightBitOCRInference(
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        use_8bit=not args.no_8bit,
        mixed_precision=not args.no_mixed_precision
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
            results = ocr_engine.batch_process(
                image_paths=image_paths,
                output_dir=args.output_image,  # Use as output directory
                max_tokens=args.max_tokens
            )
            
            # Calculate batch statistics
            successful = sum(1 for r in results if 'error' not in r)
            total_time = sum(r.get('inference_time', 0) for r in results)
            total_regions = sum(r.get('statistics', {}).get('total_regions', 0) for r in results if 'error' not in r)
            total_text_length = sum(r.get('statistics', {}).get('total_text_length', 0) for r in results if 'error' not in r)
            
            print(f"\n{'='*80}")
            print("BATCH OCR PROCESSING SUMMARY (8-BIT MIXED PRECISION)")
            print(f"{'='*80}")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total text regions found: {total_regions}")
            print(f"Total text characters: {total_text_length:,}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(results):.2f}s")
            print(f"Average regions per image: {total_regions/successful:.1f}" if successful > 0 else "N/A")
            print(f"Model checkpoint: {args.model_checkpoint}")
            print(f"Quantization: {'8-bit' if not args.no_8bit else 'FP16/BF16'}")
            print(f"Mixed precision: {'Enabled' if not args.no_mixed_precision else 'Disabled'}")
            print(f"Output directory: {args.output_image}")
            print(f"{'='*80}")
            
        else:
            # Single image processing
            results = ocr_engine.perform_ocr(
                image_path=args.image,
                max_tokens=args.max_tokens,
                save_image=None if args.no_image_output else args.output_image,
                save_text=args.output_text
            )
            
            stats = results['statistics']
            
            # Print results summary
            print(f"\n{'='*80}")
            print("OCR RESULTS SUMMARY (8-BIT MIXED PRECISION)")
            print(f"{'='*80}")
            print(f"Processing time: {results['inference_time']:.2f}s")
            print(f"Text regions found: {stats['total_regions']}")
            print(f"Total text characters: {stats['total_text_length']:,}")
            print(f"Average confidence: {stats['avg_confidence']:.3f}")
            print(f"Image size: {stats['image_size'][0]}x{stats['image_size'][1]}")
            print(f"Model checkpoint: {args.model_checkpoint}")
            print(f"Quantization: {'8-bit' if not args.no_8bit else 'FP16/BF16'}")
            print(f"Mixed precision: {'Enabled' if not args.no_mixed_precision else 'Disabled'}")
            if not args.no_image_output:
                print(f"Annotated image: {args.output_image}")
            if args.output_text:
                print(f"Text results: {args.output_text}")
            print(f"{'='*80}")
            
            if args.verbose:
                print("\nDETAILED RESULTS:")
                print("-" * 60)
                for i, result in enumerate(results['results'], 1):
                    bbox = result['bbox']
                    print(f"Region {i} (confidence: {result['confidence']:.3f}): {result['text']}")
                    print(f"  BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                print("-" * 60)
        
    except Exception as e:
        logger.error(f"OCR inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
