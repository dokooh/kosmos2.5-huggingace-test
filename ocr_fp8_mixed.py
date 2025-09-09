#!/usr/bin/env python3
"""
8-bit Mixed Precision OCR Inference for Kosmos-2.5 with Enhanced Debugging

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
- EXTENSIVE DEBUGGING OUTPUT TO IDENTIFY STOPPING POINTS
"""

import re
import torch
import requests
import argparse
import sys
import os
import time
import traceback
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
import logging

# Enhanced logging setup with more detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ocr_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def debug_checkpoint(message, checkpoint_id=None):
    """Debug checkpoint function to track execution flow"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    if checkpoint_id:
        logger.debug(f"🔍 CHECKPOINT [{checkpoint_id}]: {message}")
        print(f"[{timestamp}] 🔍 CHECKPOINT [{checkpoint_id}]: {message}", flush=True)
    else:
        logger.debug(f"🔍 DEBUG: {message}")
        print(f"[{timestamp}] 🔍 DEBUG: {message}", flush=True)

def debug_memory_status():
    """Debug memory status"""
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            debug_checkpoint(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        except Exception as e:
            debug_checkpoint(f"Failed to get GPU memory status: {e}")
    else:
        debug_checkpoint("CUDA not available")

def safe_execute(func, description, *args, **kwargs):
    """Safely execute a function with detailed error reporting"""
    debug_checkpoint(f"Starting: {description}")
    try:
        result = func(*args, **kwargs)
        debug_checkpoint(f"Completed: {description}")
        return result
    except Exception as e:
        debug_checkpoint(f"FAILED: {description} - Error: {str(e)}")
        logger.error(f"Exception in {description}: {traceback.format_exc()}")
        raise

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
        debug_checkpoint("Initializing EightBitOCRInference", "INIT_START")
        
        self.model_checkpoint = model_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit and torch.cuda.is_available()
        self.mixed_precision = mixed_precision
        self.model = None
        self.processor = None
        
        debug_checkpoint(f"Parameters - Model: {model_checkpoint}, Device: {self.device}, 8bit: {self.use_8bit}")
        
        # Determine model type
        self.is_local_checkpoint = os.path.exists(model_checkpoint)
        debug_checkpoint(f"Local checkpoint: {self.is_local_checkpoint}")
        
        # Configure precision settings
        safe_execute(self._configure_precision, "Configure precision settings")
        
        logger.info(f"Initializing 8-bit Mixed Precision OCR inference on {self.device}")
        logger.info(f"Model checkpoint: {self.model_checkpoint}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Local checkpoint: {self.is_local_checkpoint}")
        
        debug_checkpoint("EightBitOCRInference initialization completed", "INIT_END")
    
    def _configure_precision(self):
        """Configure precision and quantization settings"""
        debug_checkpoint("Configuring precision settings", "PRECISION_START")
        
        if self.use_8bit:
            debug_checkpoint("Setting up 8-bit quantization")
            # Configure 8-bit quantization with mixed precision
            if self.mixed_precision:
                logger.info("Configuring 8-bit mixed precision quantization")
                debug_checkpoint("Creating BitsAndBytesConfig with mixed precision")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_threshold=6.0,
                    # Skip critical layers for better OCR accuracy
                    llm_int8_skip_modules=["lm_head", "embed_tokens", "layernorm", "layer_norm", "norm"]
                )
                debug_checkpoint("BitsAndBytesConfig created successfully")
            else:
                logger.info("Configuring standard 8-bit quantization")
                debug_checkpoint("Creating standard BitsAndBytesConfig")
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0,
                )
            
            # Use float16 for non-quantized operations
            self.dtype = torch.float16
            debug_checkpoint("Set dtype to float16 for 8-bit mode")
        else:
            debug_checkpoint("Setting up non-8bit precision")
            self.quantization_config = None
            # Use bfloat16 for better performance on modern hardware
            if self.device.startswith('cuda') and torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                logger.info("Using bfloat16 for optimal performance")
                debug_checkpoint("Set dtype to bfloat16")
            elif self.device.startswith('cuda'):
                self.dtype = torch.float16
                logger.info("Using float16")
                debug_checkpoint("Set dtype to float16")
            else:
                self.dtype = torch.float32
                logger.info("Using float32 for CPU")
                debug_checkpoint("Set dtype to float32 for CPU")
        
        debug_checkpoint("Precision configuration completed", "PRECISION_END")
    
    def _validate_checkpoint(self, checkpoint_path):
        """Validate that the checkpoint contains required files"""
        debug_checkpoint(f"Validating checkpoint: {checkpoint_path}", "VALIDATE_START")
        
        if not os.path.isdir(checkpoint_path):
            debug_checkpoint("Checkpoint path is not a directory")
            return False
        
        debug_checkpoint("Scanning checkpoint directory for files")
        try:
            files = os.listdir(checkpoint_path)
            debug_checkpoint(f"Found {len(files)} files in checkpoint directory")
            
            # Check for model files (SafeTensors or PyTorch)
            model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt'))]
            config_files = [f for f in files if f in ['config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json']]
            
            debug_checkpoint(f"Model files: {model_files}")
            debug_checkpoint(f"Config files: {config_files}")
            
            has_model = len(model_files) > 0
            has_config = len(config_files) > 0
            
            if has_model and has_config:
                logger.info(f"✓ Found {len(model_files)} model files in {checkpoint_path}")
                debug_checkpoint("Checkpoint validation successful", "VALIDATE_END")
                return True
            else:
                logger.warning(f"⚠ Checkpoint missing required files. Model files: {has_model}, Config: {has_config}")
                debug_checkpoint("Checkpoint validation failed", "VALIDATE_END")
                return False
        except Exception as e:
            debug_checkpoint(f"Error during checkpoint validation: {e}")
            return False
    
    def load_model(self):
        """Load model with 8-bit mixed precision quantization"""
        debug_checkpoint("Starting model loading", "LOAD_MODEL_START")
        
        if self.model is not None:
            debug_checkpoint("Model already loaded, skipping")
            return
            
        logger.info("Loading Kosmos-2.5 model with 8-bit mixed precision...")
        debug_memory_status()
        
        # Validate local checkpoint if specified
        if self.is_local_checkpoint:
            debug_checkpoint("Validating local checkpoint")
            if not safe_execute(self._validate_checkpoint, "Validate checkpoint", self.model_checkpoint):
                logger.error(f"Invalid checkpoint directory: {self.model_checkpoint}")
                raise ValueError("Checkpoint path does not contain valid model files")
        
        try:
            debug_checkpoint("Configuring loading parameters")
            # Configure loading parameters
            loading_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            debug_checkpoint(f"Base loading kwargs: {loading_kwargs}")
            
            # Add quantization config if using 8-bit
            if self.use_8bit:
                debug_checkpoint("Adding 8-bit quantization config")
                loading_kwargs.update({
                    "quantization_config": self.quantization_config,
                    "device_map": "auto",
                    "torch_dtype": self.dtype,
                })
                logger.info("Loading with 8-bit mixed precision quantization...")
                debug_checkpoint("8-bit config added to loading kwargs")
            else:
                debug_checkpoint("Adding non-8bit config")
                loading_kwargs.update({
                    "torch_dtype": self.dtype,
                    "device_map": "auto" if torch.cuda.is_available() else None,
                })
            
            # Configure for local vs remote loading
            if self.is_local_checkpoint:
                debug_checkpoint("Configuring for local checkpoint loading")
                loading_kwargs.update({
                    "local_files_only": True,
                    "use_safetensors": True,
                })
                logger.info("Loading from local checkpoint...")
            else:
                debug_checkpoint("Configuring for remote checkpoint loading")
                loading_kwargs.update({
                    "local_files_only": False,
                    "use_safetensors": True,
                    "resume_download": True,
                })
                logger.info("Loading from HuggingFace Hub...")
            
            debug_checkpoint(f"Final loading kwargs: {loading_kwargs}")
            
            # Add Flash Attention if available (better performance)
            debug_checkpoint("Checking Flash Attention availability")
            try:
                if hasattr(torch.nn, 'scaled_dot_product_attention'):
                    loading_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Enabled Flash Attention 2")
                    debug_checkpoint("Flash Attention 2 enabled")
                else:
                    debug_checkpoint("Flash Attention 2 not available - no scaled_dot_product_attention")
            except Exception as e:
                debug_checkpoint(f"Flash Attention check failed: {e}")
                logger.debug("Flash Attention 2 not available")
            
            # Load the model
            debug_checkpoint("Starting model loading from pretrained", "MODEL_LOAD_START")
            debug_memory_status()
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_checkpoint,
                **loading_kwargs
            )
            
            debug_checkpoint("Model loaded successfully", "MODEL_LOAD_END")
            debug_memory_status()
            
            # Apply additional optimizations for non-8bit models
            if not self.use_8bit:
                debug_checkpoint("Applying non-8bit optimizations")
                if self.device.startswith('cuda'):
                    if self.dtype == torch.bfloat16:
                        debug_checkpoint("Converting model to bfloat16")
                        self.model = self.model.bfloat16()
                    else:
                        debug_checkpoint("Converting model to half precision")
                        self.model = self.model.half()
                debug_checkpoint("Non-8bit optimizations applied")
            
            # Enable gradient checkpointing for memory efficiency
            debug_checkpoint("Checking gradient checkpointing")
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                debug_checkpoint("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for memory efficiency")
            else:
                debug_checkpoint("Gradient checkpointing not available")
            
            # Load processor
            debug_checkpoint("Starting processor loading", "PROCESSOR_LOAD_START")
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
            
            debug_checkpoint(f"Processor kwargs: {processor_kwargs}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                **processor_kwargs
            )
            
            debug_checkpoint("Processor loaded successfully", "PROCESSOR_LOAD_END")
            
            # Set pad token if not present
            debug_checkpoint("Checking pad token")
            if self.processor.tokenizer.pad_token is None:
                debug_checkpoint("Setting pad token to eos token")
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            else:
                debug_checkpoint("Pad token already set")
            
            # Enable model optimizations
            debug_checkpoint("Setting model to eval mode")
            if hasattr(self.model, 'eval'):
                self.model.eval()
                debug_checkpoint("Model set to eval mode")
            else:
                debug_checkpoint("Model does not have eval method")
            
            # Enable torch.compile for PyTorch 2.0+ (if available and not using 8-bit)
            debug_checkpoint("Checking torch.compile availability")
            if hasattr(torch, 'compile') and self.device.startswith('cuda') and not self.use_8bit:
                try:
                    debug_checkpoint("Attempting torch.compile")
                    logger.info("Compiling model with torch.compile for faster inference...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    debug_checkpoint("torch.compile successful")
                except Exception as e:
                    debug_checkpoint(f"torch.compile failed: {e}")
                    logger.warning(f"torch.compile failed, continuing without it: {e}")
            else:
                debug_checkpoint("torch.compile not available or not applicable")
            
            logger.info("✓ Model loaded successfully with 8-bit mixed precision optimizations")
            debug_checkpoint("Model loading completed successfully", "LOAD_MODEL_END")
            debug_memory_status()
            
            # Print model info
            debug_checkpoint("Gathering model information")
            try:
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model parameters: {num_params:,}")
                debug_checkpoint(f"Model has {num_params:,} parameters")
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"GPU memory available: {gpu_memory:.1f} GB")
                    logger.info(f"GPU memory allocated: {allocated_memory:.1f} GB")
                    logger.info(f"Memory efficiency: {(1 - allocated_memory/gpu_memory)*100:.1f}% free")
                    debug_checkpoint(f"GPU memory - Total: {gpu_memory:.1f}GB, Allocated: {allocated_memory:.1f}GB")
            except Exception as e:
                debug_checkpoint(f"Failed to gather model info: {e}")
            
        except Exception as e:
            debug_checkpoint(f"Model loading failed with error: {str(e)}", "LOAD_MODEL_FAILED")
            logger.error(f"Failed to load model: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to basic loading
            debug_checkpoint("Attempting fallback model loading", "FALLBACK_START")
            logger.info("Attempting fallback model loading without quantization...")
            try:
                fallback_kwargs = {
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True
                }
                
                if self.is_local_checkpoint:
                    fallback_kwargs["local_files_only"] = True
                
                debug_checkpoint(f"Fallback kwargs: {fallback_kwargs}")
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_checkpoint,
                    **fallback_kwargs
                )
                debug_checkpoint("Fallback model loaded")
                
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
                debug_checkpoint("Fallback processor loaded")
                
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                
                logger.info("Fallback model loading successful")
                debug_checkpoint("Fallback loading successful", "FALLBACK_END")
                
            except Exception as e2:
                debug_checkpoint(f"Fallback loading also failed: {str(e2)}", "FALLBACK_FAILED")
                logger.error(f"Fallback loading also failed: {e2}")
                logger.error(f"Fallback traceback: {traceback.format_exc()}")
                raise
    
    def load_image(self, image_path):
        """Load image from local path or URL with error handling"""
        debug_checkpoint(f"Loading image: {image_path}", "IMAGE_LOAD_START")
        
        try:
            if image_path.startswith(('http://', 'https://')):
                debug_checkpoint("Loading image from URL")
                logger.info(f"Loading image from URL: {image_path}")
                response = requests.get(image_path, stream=True, timeout=30)
                response.raise_for_status()
                image = Image.open(response.raw)
                debug_checkpoint("URL image loaded successfully")
            else:
                debug_checkpoint("Loading image from file")
                logger.info(f"Loading image from file: {image_path}")
                if not os.path.exists(image_path):
                    debug_checkpoint("Image file not found")
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
                debug_checkpoint("File image loaded successfully")
            
            # Convert to RGB and validate
            debug_checkpoint("Converting image to RGB")
            image = image.convert('RGB')
            logger.info(f"Image loaded successfully. Size: {image.size}")
            debug_checkpoint(f"Image conversion completed. Size: {image.size}", "IMAGE_LOAD_END")
            return image
            
        except Exception as e:
            debug_checkpoint(f"Image loading failed: {str(e)}", "IMAGE_LOAD_FAILED")
            logger.error(f"Error loading image: {e}")
            raise
    
    def post_process_ocr(self, generated_text, scale_height, scale_width, prompt="<ocr>"):
        """Enhanced post-process OCR results to extract bounding boxes and text"""
        debug_checkpoint("Starting OCR post-processing", "POSTPROCESS_START")
        
        text = generated_text.replace(prompt, "").strip()
        debug_checkpoint(f"Cleaned text length: {len(text)}")
        
        # Enhanced pattern to match bounding boxes with better error handling
        pattern = r"<bbox><x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)></bbox>"
        debug_checkpoint(f"Using pattern: {pattern}")
        
        results = []
        processed_positions = set()
        
        debug_checkpoint("Searching for bounding box matches")
        matches = list(re.finditer(pattern, text))
        debug_checkpoint(f"Found {len(matches)} potential matches")
        
        for i, match in enumerate(matches):
            debug_checkpoint(f"Processing match {i+1}/{len(matches)}")
            
            # Extract coordinates
            x1, y1, x2, y2 = map(int, match.groups())
            debug_checkpoint(f"Raw coordinates: ({x1}, {y1}, {x2}, {y2})")
            
            # Skip if coordinates are invalid or already processed
            bbox_key = (x1, y1, x2, y2)
            if bbox_key in processed_positions:
                debug_checkpoint("Skipping duplicate bbox")
                continue
            
            # Scale coordinates to original image size
            x1_scaled = int(x1 * scale_width)
            y1_scaled = int(y1 * scale_height)
            x2_scaled = int(x2 * scale_width)
            y2_scaled = int(y2 * scale_height)
            debug_checkpoint(f"Scaled coordinates: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled})")
            
            # Validate bounding box dimensions
            if x1_scaled >= x2_scaled or y1_scaled >= y2_scaled:
                debug_checkpoint("Skipping invalid bbox dimensions")
                continue
            
            # Get corresponding text (next non-empty part after bbox)
            text_start = match.end()
            next_bbox = re.search(pattern, text[text_start:])
            
            if next_bbox:
                extracted_text = text[text_start:text_start + next_bbox.start()].strip()
            else:
                extracted_text = text[text_start:].strip()
            
            debug_checkpoint(f"Raw extracted text: '{extracted_text[:50]}...'")
            
            # Enhanced text cleaning
            extracted_text = safe_execute(self._clean_extracted_text, "Clean extracted text", extracted_text)
            debug_checkpoint(f"Cleaned text: '{extracted_text[:50]}...'")
            
            if extracted_text:
                # Calculate confidence based on text length and bbox size
                bbox_area = (x2_scaled - x1_scaled) * (y2_scaled - y1_scaled)
                confidence = min(1.0, len(extracted_text) / 50.0 + bbox_area / 10000.0)
                debug_checkpoint(f"Calculated confidence: {confidence:.3f}")
                
                result = {
                    'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                    'text': extracted_text,
                    'confidence': confidence,
                    'raw_coords': [x1, y1, x2, y2]  # Keep original coordinates for debugging
                }
                results.append(result)
                processed_positions.add(bbox_key)
                debug_checkpoint(f"Added result {len(results)}")
        
        debug_checkpoint(f"Post-processing completed. Found {len(results)} valid regions", "POSTPROCESS_END")
        return results
    
    def _clean_extracted_text(self, text):
        """Enhanced text cleaning for OCR results"""
        debug_checkpoint("Cleaning extracted text")
        
        # Remove HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\-.,!?():;"\'/]', '', text)
        
        # Remove very short or meaningless extractions
        if len(text.strip()) < 2:
            debug_checkpoint("Text too short, returning empty")
            return ""
        
        debug_checkpoint(f"Text cleaning completed: '{text[:30]}...'")
        return text
    
    def draw_bounding_boxes(self, image, ocr_results, output_path=None):
        """Enhanced drawing of bounding boxes and text on image"""
        debug_checkpoint("Starting bounding box drawing", "DRAW_START")
        
        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        debug_checkpoint("Created image copy and draw object")
        
        # Try to load a font with better size handling
        debug_checkpoint("Loading font")
        try:
            font_size = max(12, min(24, int(min(image.size) / 50)))  # Adaptive font size
            debug_checkpoint(f"Calculated font size: {font_size}")
            
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("arial.ttf", font_size)
                debug_checkpoint("Loaded Arial font on Windows")
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
                        debug_checkpoint(f"Loaded font from: {font_path}")
                        break
                    except:
                        continue
                
                if font is None:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    debug_checkpoint("Loaded DejaVuSans font")
        except Exception as e:
            debug_checkpoint(f"Font loading failed, using default: {e}")
            font = ImageFont.load_default()
        
        # Enhanced color palette
        colors = [
            '#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', 
            '#FF0080', '#00FFFF', '#FFFF00', '#FF8080', '#80FF80'
        ]
        debug_checkpoint(f"Drawing {len(ocr_results)} bounding boxes")
        
        for i, result in enumerate(ocr_results):
            debug_checkpoint(f"Drawing box {i+1}/{len(ocr_results)}")
            
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
            debug_checkpoint(f"Saving annotated image to: {output_path}")
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            annotated_image.save(output_path, quality=95, optimize=True)
            logger.info(f"Annotated image saved to: {output_path}")
        
        debug_checkpoint("Bounding box drawing completed", "DRAW_END")
        return annotated_image
    
    def perform_ocr(self, image_path, max_tokens=1024, save_image=None, save_text=None):
        """Perform OCR on image using 8-bit mixed precision"""
        debug_checkpoint("Starting OCR performance", "OCR_START")
        
        if self.model is None:
            debug_checkpoint("Model not loaded, loading now")
            safe_execute(self.load_model, "Load model")
        
        # Load and process image
        image = safe_execute(self.load_image, "Load image", image_path)
        
        prompt = "<ocr>"
        start_time = time.time()
        debug_checkpoint(f"Starting inference with prompt: {prompt}")
        
        try:
            # Process inputs
            debug_checkpoint("Processing inputs with processor", "PROCESS_INPUTS_START")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            debug_checkpoint(f"Processor returned keys: {list(inputs.keys())}")
            
            # Extract scaling information
            height = inputs.pop("height")
            width = inputs.pop("width")
            raw_width, raw_height = image.size
            scale_height = raw_height / height
            scale_width = raw_width / width
            debug_checkpoint(f"Image scaling - Raw: {raw_width}x{raw_height}, Processed: {width}x{height}")
            debug_checkpoint(f"Scale factors - Height: {scale_height:.3f}, Width: {scale_width:.3f}")
            debug_checkpoint("Input processing completed", "PROCESS_INPUTS_END")
            
            # Move inputs to device and convert to correct dtype if not using 8-bit
            debug_checkpoint("Moving inputs to device", "DEVICE_MOVE_START")
            debug_memory_status()
            
            if not self.use_8bit:
                debug_checkpoint("Non-8bit mode: moving inputs to device manually")
                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
                
                # Convert flattened_patches to correct dtype
                if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                    debug_checkpoint("Converting flattened_patches to correct dtype")
                    inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            else:
                debug_checkpoint("8-bit mode: moving inputs to device (quantization handles dtype)")
                # For 8-bit models, just move to device (quantization handles dtype)
                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            debug_checkpoint("Device move completed", "DEVICE_MOVE_END")
            debug_memory_status()
            
            # Generate OCR results
            debug_checkpoint("Starting OCR inference generation", "GENERATION_START")
            logger.info("Performing OCR inference with 8-bit mixed precision...")
            
            with torch.no_grad():
                debug_checkpoint("Entered torch.no_grad() context")
                
                # Clear GPU cache before generation
                if torch.cuda.is_available():
                    debug_checkpoint("Clearing CUDA cache")
                    torch.cuda.empty_cache()
                    debug_memory_status()
                
                # Use mixed precision if enabled and not using 8-bit quantization
                if self.mixed_precision and not self.use_8bit and self.device.startswith('cuda'):
                    debug_checkpoint("Using mixed precision autocast")
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        debug_checkpoint("Inside autocast context, calling model.generate")
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=1,  # Faster than beam search for OCR
                            early_stopping=True,
                        )
                        debug_checkpoint("model.generate completed with autocast")
                else:
                    debug_checkpoint("Using standard generation (no autocast)")
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        early_stopping=True,
                    )
                    debug_checkpoint("model.generate completed without autocast")
            
            debug_checkpoint("Generation completed", "GENERATION_END")
            debug_memory_status()
            
            # Decode results
            debug_checkpoint("Decoding generated results", "DECODE_START")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            debug_checkpoint(f"Decoded text length: {len(generated_text)}")
            debug_checkpoint(f"Generated text preview: '{generated_text[:100]}...'")
            debug_checkpoint("Decoding completed", "DECODE_END")
            
            # Post-process to extract structured data
            ocr_results = safe_execute(self.post_process_ocr, "Post-process OCR", generated_text, scale_height, scale_width, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"OCR completed in {inference_time:.2f}s. Found {len(ocr_results)} text regions.")
            debug_checkpoint(f"OCR inference completed in {inference_time:.2f}s")
            
            # Calculate additional statistics
            debug_checkpoint("Calculating statistics")
            total_text_length = sum(len(result['text']) for result in ocr_results)
            avg_confidence = sum(result['confidence'] for result in ocr_results) / len(ocr_results) if ocr_results else 0
            debug_checkpoint(f"Statistics - Total text length: {total_text_length}, Avg confidence: {avg_confidence:.3f}")
            
            # Save outputs if requested
            if save_text:
                debug_checkpoint("Saving text results")
                safe_execute(self.save_text_results, "Save text results", ocr_results, save_text)
            
            if save_image:
                debug_checkpoint("Saving annotated image")
                safe_execute(self.draw_bounding_boxes, "Draw bounding boxes", image, ocr_results, save_image)
            
            result = {
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
            
            debug_checkpoint("OCR performance completed successfully", "OCR_END")
            return result
            
        except Exception as e:
            debug_checkpoint(f"OCR inference failed with error: {str(e)}", "OCR_FAILED")
            logger.error(f"Error during OCR inference: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def save_text_results(self, ocr_results, output_path):
        """Save OCR results as structured text with enhanced formatting"""
        debug_checkpoint(f"Saving text results to: {output_path}", "SAVE_TEXT_START")
        
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
            debug_checkpoint("Text results saved successfully", "SAVE_TEXT_END")
            
        except Exception as e:
            debug_checkpoint(f"Failed to save text results: {str(e)}", "SAVE_TEXT_FAILED")
            logger.error(f"Error saving text results: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=1024):
        """Process multiple images in batch with progress tracking"""
        debug_checkpoint(f"Starting batch processing of {len(image_paths)} images", "BATCH_START")
        
        if self.model is None:
            debug_checkpoint("Loading model for batch processing")
            safe_execute(self.load_model, "Load model for batch")
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting batch OCR processing of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            debug_checkpoint(f"Processing batch image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Generate output filenames
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image = os.path.join(output_dir, f"{base_name}_ocr.png")
                output_text = os.path.join(output_dir, f"{base_name}_ocr.txt")
                
                # Perform OCR
                result = safe_execute(self.perform_ocr, f"OCR for {base_name}",
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
                debug_checkpoint(f"Batch progress: {progress:.1f}% completed")
                
            except Exception as e:
                debug_checkpoint(f"Failed to process batch image {image_path}: {str(e)}")
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e),
                    'inference_time': 0,
                    'statistics': {}
                })
        
        logger.info("Batch OCR processing completed!")
        debug_checkpoint("Batch processing completed", "BATCH_END")
        return results

def get_args():
    parser = argparse.ArgumentParser(description='8-bit Mixed Precision OCR inference using Kosmos-2.5 with Enhanced Debugging')
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable maximum debug output')
    
    return parser.parse_args()

def main():
    debug_checkpoint("Application starting", "MAIN_START")
    
    args = get_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        debug_checkpoint("Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        debug_checkpoint("Verbose mode enabled")
    
    debug_checkpoint(f"Arguments: {vars(args)}")
    
    # Initialize 8-bit mixed precision OCR inference
    debug_checkpoint("Initializing OCR engine")
    ocr_engine = safe_execute(EightBitOCRInference, "Initialize OCR engine",
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        use_8bit=not args.no_8bit,
        mixed_precision=not args.no_mixed_precision
    )
    
    try:
        if args.batch and os.path.isdir(args.image):
            debug_checkpoint("Starting batch processing mode")
            # Batch processing
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_paths = [
                os.path.join(args.image, f) for f in os.listdir(args.image)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_paths:
                debug_checkpoint("No images found in batch directory")
                logger.error(f"No images found in directory: {args.image}")
                sys.exit(1)
            
            debug_checkpoint(f"Found {len(image_paths)} images for batch processing")
            logger.info(f"Processing {len(image_paths)} images in batch mode")
            
            results = safe_execute(ocr_engine.batch_process, "Batch process images",
                image_paths=image_paths,
                output_dir=args.output_image,  # Use as output directory
                max_tokens=args.max_tokens
            )
            
            # Calculate batch statistics
            debug_checkpoint("Calculating batch statistics")
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
            debug_checkpoint("Starting single image processing mode")
            # Single image processing
            results = safe_execute(ocr_engine.perform_ocr, "Perform OCR on single image",
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
        
        debug_checkpoint("Application completed successfully", "MAIN_END")
        
    except Exception as e:
        debug_checkpoint(f"Application failed with error: {str(e)}", "MAIN_FAILED")
        logger.error(f"OCR inference failed: {e}")
        if args.verbose or args.debug:
            logger.error(f"Full traceback: {traceback.format_exc()}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
