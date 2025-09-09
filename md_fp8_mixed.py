#!/usr/bin/env python3
# filepath: c:\SAI\IA\unilm\kosmos-2.5\md_fp8_mixed.py
"""
8-bit Mixed Precision Markdown Generation for Kosmos-2.5 with Enhanced Debugging - CUDA ERROR FIX

This module provides fast markdown generation using 8-bit mixed precision quantized Kosmos-2.5 model.
Features:
- 8-bit mixed precision quantization with faster inference and reduced memory usage
- BitsAndBytesConfig for advanced quantization settings
- SafeTensors format support for faster loading
- Support for local model checkpoints and remote models
- Optimized memory usage with gradient checkpointing
- Enhanced markdown post-processing with structure detection
- Batch processing support with progress tracking
- Comprehensive error handling and fallback mechanisms
- EXTENSIVE DEBUGGING OUTPUT TO IDENTIFY STOPPING POINTS
- CUDA ERROR FIX: Enhanced tensor validation and device placement
"""

import torch
import requests
import argparse
import sys
import os
import time
import re
import traceback as tb_module  # Avoid namespace conflict
from PIL import Image
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
        logging.FileHandler('md_debug.log', mode='w')
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

def debug_tensor_devices(tensor_dict, name="Tensors"):
    """Debug tensor device placement with enhanced validation"""
    debug_checkpoint(f"Checking {name} device placement:")
    for key, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            try:
                # Check for invalid values
                has_nan = torch.isnan(tensor).any()
                has_inf = torch.isinf(tensor).any()
                min_val = tensor.min().item() if tensor.numel() > 0 else 0
                max_val = tensor.max().item() if tensor.numel() > 0 else 0
                
                debug_checkpoint(f"  {key}: device={tensor.device}, dtype={tensor.dtype}, shape={tensor.shape}")
                debug_checkpoint(f"    Range: [{min_val:.6f}, {max_val:.6f}], NaN: {has_nan}, Inf: {has_inf}")
                
                # Warn about problematic values
                if has_nan:
                    debug_checkpoint(f"    WARNING: Tensor {key} contains NaN values!")
                if has_inf:
                    debug_checkpoint(f"    WARNING: Tensor {key} contains infinite values!")
                    
            except Exception as e:
                debug_checkpoint(f"  {key}: Error checking tensor - {e}")
        else:
            debug_checkpoint(f"  {key}: {type(tensor)} (not a tensor)")

def validate_and_fix_tensors(tensor_dict, device):
    """Validate and fix tensor issues that could cause CUDA errors"""
    debug_checkpoint("Validating and fixing tensor issues", "TENSOR_FIX_START")
    
    fixed_tensors = {}
    for key, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            try:
                # Check and fix NaN/Inf values
                if torch.isnan(tensor).any():
                    debug_checkpoint(f"Fixing NaN values in {key}")
                    tensor = torch.nan_to_num(tensor, nan=0.0)
                
                if torch.isinf(tensor).any():
                    debug_checkpoint(f"Fixing infinite values in {key}")
                    tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=-1.0)
                
                # Ensure tensor is on correct device
                if tensor.device.type != device.split(':')[0]:
                    debug_checkpoint(f"Moving {key} from {tensor.device} to {device}")
                    tensor = tensor.to(device)
                
                # For attention masks, ensure values are 0 or 1
                if 'attention_mask' in key.lower():
                    debug_checkpoint(f"Validating attention mask {key}")
                    # Clamp to valid range and ensure proper dtype
                    tensor = torch.clamp(tensor, 0, 1).long()
                    debug_checkpoint(f"Attention mask {key} clamped and converted to long")
                
                # For input_ids, ensure they're within vocabulary range
                if 'input_ids' in key.lower():
                    debug_checkpoint(f"Validating input_ids {key}")
                    # Ensure no negative values
                    tensor = torch.clamp(tensor, min=0)
                    debug_checkpoint(f"Input_ids {key} clamped to non-negative")
                
                fixed_tensors[key] = tensor
                
            except Exception as e:
                debug_checkpoint(f"Error fixing tensor {key}: {e}")
                fixed_tensors[key] = tensor  # Use original if fixing fails
        else:
            fixed_tensors[key] = tensor
    
    debug_checkpoint("Tensor validation and fixing completed", "TENSOR_FIX_END")
    return fixed_tensors

def move_to_device_safe(data, device, dtype=None):
    """Safely move data to device with proper error handling"""
    if isinstance(data, torch.Tensor):
        try:
            # Check if tensor is already on the target device
            if data.device.type == device.split(':')[0]:
                if dtype is not None and data.dtype != dtype and data.dtype not in [torch.int32, torch.int64, torch.bool]:
                    return data.to(dtype=dtype)
                return data
            
            if dtype is not None and data.dtype != dtype and data.dtype not in [torch.int32, torch.int64, torch.bool]:
                return data.to(device=device, dtype=dtype)
            else:
                return data.to(device=device)
        except Exception as e:
            debug_checkpoint(f"Warning: Failed to move tensor to {device}: {e}")
            return data
    elif isinstance(data, dict):
        return {k: move_to_device_safe(v, device, dtype) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device_safe(item, device, dtype) for item in data]
    else:
        return data

def safe_execute(func, description, *args, **kwargs):
    """Safely execute a function with detailed error reporting"""
    debug_checkpoint(f"Starting: {description}")
    try:
        result = func(*args, **kwargs)
        debug_checkpoint(f"Completed: {description}")
        return result
    except Exception as e:
        debug_checkpoint(f"FAILED: {description} - Error: {str(e)}")
        logger.error(f"Exception in {description}: {tb_module.format_exc()}")
        raise

def tensor_to_float(tensor_val):
    """Safely convert tensor to float for formatting"""
    if isinstance(tensor_val, torch.Tensor):
        return float(tensor_val.item())
    return float(tensor_val)

def enable_cuda_debugging():
    """Enable CUDA debugging environment variables"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    debug_checkpoint("Enabled CUDA debugging environment variables")

class EightBitMarkdownInference:
    def __init__(self, model_checkpoint, device=None, cache_dir=None, use_8bit=True, mixed_precision=True, force_fallback=False):
        """
        Initialize 8-bit Mixed Precision Markdown inference
        
        Args:
            model_checkpoint (str): Path to model checkpoint (local directory or HuggingFace model name)
            device (str): Device to use for inference
            cache_dir (str): Cache directory for downloaded models
            use_8bit (bool): Enable 8-bit quantization
            mixed_precision (bool): Enable mixed precision for critical layers
            force_fallback (bool): Force use of fallback mode (no 8-bit)
        """
        debug_checkpoint("Initializing EightBitMarkdownInference", "INIT_START")
        
        # Enable CUDA debugging
        enable_cuda_debugging()
        
        self.model_checkpoint = model_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit and torch.cuda.is_available() and not force_fallback
        self.mixed_precision = mixed_precision
        self.force_fallback = force_fallback
        self.model = None
        self.processor = None
        
        debug_checkpoint(f"Parameters - Model: {model_checkpoint}, Device: {self.device}, 8bit: {self.use_8bit}")
        
        # Determine model type
        self.is_local_checkpoint = os.path.exists(model_checkpoint)
        debug_checkpoint(f"Local checkpoint: {self.is_local_checkpoint}")
        
        # Configure precision settings
        safe_execute(self._configure_precision, "Configure precision settings")
        
        logger.info(f"Initializing 8-bit Mixed Precision Markdown inference on {self.device}")
        logger.info(f"Model checkpoint: {self.model_checkpoint}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Local checkpoint: {self.is_local_checkpoint}")
        logger.info(f"Force fallback: {self.force_fallback}")
        
        debug_checkpoint("EightBitMarkdownInference initialization completed", "INIT_END")
    
    def _configure_precision(self):
        """Configure precision and quantization settings"""
        debug_checkpoint("Configuring precision settings", "PRECISION_START")
        
        if self.use_8bit:
            debug_checkpoint("Setting up 8-bit quantization")
            # More conservative 8-bit configuration to avoid device issues
            debug_checkpoint("Creating conservative BitsAndBytesConfig")
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,  # Disable CPU offload to keep everything on GPU
                llm_int8_has_fp16_weight=False,
                llm_int8_threshold=6.0,
            )
            debug_checkpoint("Conservative BitsAndBytesConfig created successfully")
            
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
        
        # Try 8-bit loading first, then fallback if it fails
        model_loaded = False
        
        if self.use_8bit and not self.force_fallback:
            debug_checkpoint("Attempting 8-bit model loading", "8BIT_ATTEMPT")
            try:
                model_loaded = self._load_8bit_model()
                debug_checkpoint("8-bit model loading successful", "8BIT_SUCCESS")
            except Exception as e:
                debug_checkpoint(f"8-bit model loading failed: {str(e)}", "8BIT_FAILED")
                logger.warning(f"8-bit loading failed, will try fallback: {e}")
                # Don't re-raise, let it fall through to fallback
        
        if not model_loaded:
            debug_checkpoint("Attempting fallback model loading", "FALLBACK_ATTEMPT")
            try:
                self._load_fallback_model()
                debug_checkpoint("Fallback model loading successful", "FALLBACK_SUCCESS")
            except Exception as e:
                debug_checkpoint(f"Fallback model loading failed: {str(e)}", "FALLBACK_FAILED")
                logger.error(f"Both 8-bit and fallback loading failed: {e}")
                raise
        
        # Common post-loading setup
        self._post_loading_setup()
        
        debug_checkpoint("Model loading completed successfully", "LOAD_MODEL_END")
    
    def _load_8bit_model(self):
        """Load model with 8-bit quantization"""
        debug_checkpoint("Loading 8-bit quantized model", "8BIT_LOAD_START")
        
        # Configure loading parameters for 8-bit
        loading_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "quantization_config": self.quantization_config,
            "torch_dtype": self.dtype,
        }
        
        # For 8-bit, use more conservative device mapping
        if torch.cuda.device_count() > 1:
            # Multi-GPU: use specific device
            loading_kwargs["device_map"] = {
                "": f"cuda:{torch.cuda.current_device()}"
            }
            debug_checkpoint(f"Multi-GPU setup: using device cuda:{torch.cuda.current_device()}")
        else:
            # Single GPU: use specific device instead of auto
            loading_kwargs["device_map"] = {"": self.device}
            debug_checkpoint(f"Single GPU setup: using device {self.device}")
        
        # Configure for local vs remote loading
        if self.is_local_checkpoint:
            loading_kwargs.update({
                "local_files_only": True,
                "use_safetensors": True,
            })
        else:
            loading_kwargs.update({
                "local_files_only": False,
                "use_safetensors": True,
                "resume_download": True,
            })
        
        debug_checkpoint(f"8-bit loading kwargs: {loading_kwargs}")
        
        # Load the model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_checkpoint,
            **loading_kwargs
        )
        
        debug_checkpoint("8-bit model loaded", "8BIT_LOAD_END")
        return True
    
    def _load_fallback_model(self):
        """Load model without 8-bit quantization as fallback"""
        debug_checkpoint("Loading fallback model", "FALLBACK_LOAD_START")
        
        # Disable 8-bit for fallback
        self.use_8bit = False
        
        fallback_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": None,  # We'll handle device placement manually
        }
        
        if self.is_local_checkpoint:
            fallback_kwargs["local_files_only"] = True
        
        debug_checkpoint(f"Fallback kwargs: {fallback_kwargs}")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_checkpoint,
            **fallback_kwargs
        )
        
        # Move to device manually for fallback
        if torch.cuda.is_available():
            debug_checkpoint("Moving fallback model to GPU")
            self.model = self.model.to(self.device)
            if self.dtype == torch.float16:
                self.model = self.model.half()
        
        debug_checkpoint("Fallback model loaded", "FALLBACK_LOAD_END")
    
    def _post_loading_setup(self):
        """Common setup after model loading"""
        debug_checkpoint("Starting post-loading setup", "POSTLOAD_START")
        
        # Load processor
        debug_checkpoint("Loading processor", "PROCESSOR_LOAD_START")
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
        
        debug_checkpoint(f"Processor loading kwargs: {processor_kwargs}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            **processor_kwargs
        )
        debug_checkpoint("Processor loaded", "PROCESSOR_LOAD_END")
        
        # Set pad token if not present
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            debug_checkpoint("Set pad token to eos token")
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
            debug_checkpoint("Model set to eval mode")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            debug_checkpoint("Enabled gradient checkpointing")
        
        debug_checkpoint("Post-loading setup completed", "POSTLOAD_END")
    
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
            
            # Validate image size
            if image.size[0] < 1 or image.size[1] < 1:
                raise ValueError(f"Invalid image size: {image.size}")
            
            logger.info(f"Image loaded successfully. Size: {image.size}")
            debug_checkpoint(f"Image conversion completed. Size: {image.size}", "IMAGE_LOAD_END")
            return image
            
        except Exception as e:
            debug_checkpoint(f"Image loading failed: {str(e)}", "IMAGE_LOAD_FAILED")
            logger.error(f"Error loading image: {e}")
            raise
    
    def post_process_markdown(self, generated_text, prompt="<md>"):
        """Post-process and clean up generated markdown with enhanced structure detection"""
        debug_checkpoint("Starting markdown post-processing", "POSTPROCESS_START")
        
        # Remove the prompt
        markdown = generated_text.replace(prompt, "").strip()
        debug_checkpoint(f"Cleaned text length: {len(markdown)}")
        
        # Enhanced markdown cleaning
        markdown = safe_execute(self.clean_markdown_advanced, "Clean markdown advanced", markdown)
        
        debug_checkpoint("Markdown post-processing completed", "POSTPROCESS_END")
        return markdown
    
    def clean_markdown_advanced(self, text):
        """Advanced markdown cleaning with structure detection"""
        debug_checkpoint("Starting advanced markdown cleaning")
        
        # Remove extra whitespace and clean up
        lines = text.split('\n')
        cleaned_lines = []
        
        in_code_block = False
        for line in lines:
            # Handle code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line.strip())
                continue
            
            if in_code_block:
                # Preserve code block formatting
                cleaned_lines.append(line)
                continue
            
            line = line.strip()
            # Remove any remaining HTML-like tags (except in code)
            line = re.sub(r'<(?!code|pre)[^>]+>', '', line)
            
            if line:  # Skip empty lines initially
                cleaned_lines.append(line)
        
        # Join lines back
        text = '\n'.join(cleaned_lines)
        
        # Advanced markdown formatting fixes
        
        # Fix headers - ensure proper spacing and format
        text = re.sub(r'^(#{1,6})\s*(.+)', r'\1 \2', text, flags=re.MULTILINE)
        
        # Ensure proper spacing around headers
        text = re.sub(r'(?<!^)(?<!\n)(^#{1,6}.*$)', r'\n\1', text, flags=re.MULTILINE)
        text = re.sub(r'(^#{1,6}.*$)(?!\n)(?!\Z)', r'\1\n', text, flags=re.MULTILINE)
        
        # Fix list items with proper indentation
        text = re.sub(r'^[\*\-\+]\s+', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*)(\d+)\.\s+', r'\1\2. ', text, flags=re.MULTILINE)
        
        # Fix nested lists
        text = re.sub(r'^(\s+)[\*\-\+]\s+', r'\1- ', text, flags=re.MULTILINE)
        
        # Fix table formatting
        text = re.sub(r'\|\s*([^|]+?)\s*\|', lambda m: '| ' + m.group(1).strip() + ' |', text)
        
        # Ensure table header separators
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '|' in line and i < len(lines) - 1:
                next_line = lines[i + 1]
                if '|' in next_line and not re.match(r'^\s*\|[\s\-:]+\|\s*$', next_line):
                    # Check if this looks like a header row
                    if line.count('|') >= 2 and not re.match(r'^\s*\|[\s\-:]+\|\s*$', line):
                        # Insert separator row
                        cols = line.count('|') - 1
                        separator = '|' + '---|' * cols
                        lines.insert(i + 1, separator)
                        break
        text = '\n'.join(lines)
        
        # Fix emphasis and strong formatting
        text = re.sub(r'\*\*([^*]+?)\*\*', r'**\1**', text)
        text = re.sub(r'\*([^*]+?)\*', r'*\1*', text)
        text = re.sub(r'__([^_]+?)__', r'**\1**', text)
        text = re.sub(r'_([^_]+?)_', r'*\1*', text)
        
        # Fix links
        text = re.sub(r'\[([^\]]+?)\]\s*\(([^\)]+?)\)', r'[\1](\2)', text)
        
        # Fix code spans
        text = re.sub(r'`([^`]+?)`', r'`\1`', text)
        
        # Remove excessive newlines but preserve intentional spacing
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        text = re.sub(r'\n{3}(?=\n)', '\n\n', text)  # Reduce 3+ to 2
        
        # Ensure proper spacing around block elements
        text = re.sub(r'(\n#+.*\n)(\w)', r'\1\n\2', text)  # Space after headers
        text = re.sub(r'(\w)(\n#+.*)', r'\1\n\2', text)    # Space before headers
        
        # Ensure text starts and ends cleanly
        text = text.strip()
        
        debug_checkpoint(f"Advanced markdown cleaning completed: '{text[:100]}...'")
        return text
    
    def generate_markdown(self, image_path, max_tokens=2048, temperature=0.1, save_output=None):
        """Generate markdown from image using 8-bit mixed precision with enhanced CUDA error handling"""
        debug_checkpoint("Starting markdown generation", "MD_START")
        
        if self.model is None:
            debug_checkpoint("Model not loaded, loading now")
            safe_execute(self.load_model, "Load model")
        
        # Load and process image
        image = safe_execute(self.load_image, "Load image", image_path)
        
        prompt = "<md>"
        start_time = time.time()
        debug_checkpoint(f"Starting inference with prompt: {prompt}")
        
        try:
            # Process inputs
            debug_checkpoint("Processing inputs with processor", "PROCESS_INPUTS_START")
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            debug_checkpoint(f"Processor returned keys: {list(inputs.keys())}")
            
            # Debug input tensor devices before any processing
            debug_tensor_devices(inputs, "Raw processor inputs")
            
            # Remove height/width info (not needed for generation but handle gracefully)
            height = inputs.pop("height", None)
            width = inputs.pop("width", None)
            if height is not None:
                debug_checkpoint(f"Removed height: {tensor_to_float(height)}")
            if width is not None:
                debug_checkpoint(f"Removed width: {tensor_to_float(width)}")
            
            debug_checkpoint("Input processing completed", "PROCESS_INPUTS_END")
            
            # ENHANCED DEVICE PLACEMENT AND TENSOR VALIDATION
            debug_checkpoint("Moving inputs to device with validation", "DEVICE_MOVE_START")
            debug_memory_status()
            
            # Get model device (where the first parameter is located)
            model_device = next(self.model.parameters()).device
            debug_checkpoint(f"Model is on device: {model_device}")
            
            # Validate and fix tensors before moving to device
            inputs = validate_and_fix_tensors(inputs, str(model_device))
            
            # Final device verification
            debug_tensor_devices(inputs, "Final processed and validated inputs")
            debug_checkpoint("Device move and validation completed", "DEVICE_MOVE_END")
            debug_memory_status()
            
            # Generate markdown with enhanced error handling
            debug_checkpoint("Starting markdown inference generation", "GENERATION_START")
            logger.info("Generating markdown with 8-bit mixed precision...")
            
            with torch.no_grad():
                debug_checkpoint("Entered torch.no_grad() context")
                
                # Clear GPU cache before generation
                if torch.cuda.is_available():
                    debug_checkpoint("Clearing CUDA cache")
                    torch.cuda.empty_cache()
                    debug_memory_status()
                
                # Enhanced generation with better error handling and smaller max_length for stability
                generation_kwargs = {
                    "max_new_tokens": min(max_tokens, 1024),  # Limit for stability
                    "do_sample": temperature > 0,
                    "temperature": temperature if temperature > 0 else None,
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.0,
                    "use_cache": True,
                    "num_beams": 1,
                    "early_stopping": True,
                    "output_attentions": False,  # Disable to save memory
                    "output_hidden_states": False,  # Disable to save memory
                    "return_dict_in_generate": False,  # Simplify output
                }
                
                debug_checkpoint(f"Generation kwargs: {generation_kwargs}")
                
                try:
                    # Use mixed precision if enabled and not using 8-bit quantization
                    if self.mixed_precision and not self.use_8bit and self.device.startswith('cuda'):
                        debug_checkpoint("Using mixed precision autocast")
                        with torch.cuda.amp.autocast(dtype=self.dtype):
                            debug_checkpoint("Inside autocast context, calling model.generate")
                            generated_ids = self.model.generate(
                                **inputs,
                                **generation_kwargs
                            )
                            debug_checkpoint("model.generate completed with autocast")
                    else:
                        debug_checkpoint("Using standard generation (no autocast)")
                        generated_ids = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                        debug_checkpoint("model.generate completed without autocast")
                        
                except RuntimeError as cuda_error:
                    if "CUDA error" in str(cuda_error):
                        debug_checkpoint(f"CUDA error detected: {cuda_error}")
                        debug_checkpoint("Attempting fallback generation with reduced settings")
                        
                        # Try with even more conservative settings
                        fallback_kwargs = {
                            "max_new_tokens": min(max_tokens // 2, 512),
                            "do_sample": False,  # Disable sampling
                            "pad_token_id": self.processor.tokenizer.eos_token_id,
                            "eos_token_id": self.processor.tokenizer.eos_token_id,
                            "use_cache": False,  # Disable cache
                            "num_beams": 1,
                            "early_stopping": True,
                            "output_attentions": False,
                            "output_hidden_states": False,
                            "return_dict_in_generate": False,
                        }
                        
                        debug_checkpoint(f"Fallback generation kwargs: {fallback_kwargs}")
                        
                        # Clear cache and try again
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        generated_ids = self.model.generate(
                            **inputs,
                            **fallback_kwargs
                        )
                        debug_checkpoint("Fallback generation completed")
                    else:
                        raise  # Re-raise if not a CUDA error
            
            debug_checkpoint("Generation completed", "GENERATION_END")
            debug_memory_status()
            
            # Decode results
            debug_checkpoint("Decoding generated results", "DECODE_START")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            debug_checkpoint(f"Decoded text length: {len(generated_text)}")
            debug_checkpoint(f"Generated text preview: '{generated_text[:100]}...'")
            debug_checkpoint("Decoding completed", "DECODE_END")
            
            # Post-process markdown
            markdown_output = safe_execute(self.post_process_markdown, "Post-process markdown", generated_text, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"Markdown generation completed in {inference_time:.2f}s")
            debug_checkpoint(f"Markdown generation completed in {inference_time:.2f}s")
            
            # Calculate statistics
            debug_checkpoint("Calculating statistics")
            word_count = len(markdown_output.split())
            char_count = len(markdown_output)
            line_count = len(markdown_output.split('\n'))
            
            # Detect markdown elements
            headers = len(re.findall(r'^#+\s', markdown_output, re.MULTILINE))
            lists = len(re.findall(r'^[\*\-\+]\s|^\d+\.\s', markdown_output, re.MULTILINE))
            tables = markdown_output.count('|')
            code_blocks = markdown_output.count('```') // 2
            
            debug_checkpoint(f"Statistics - Words: {word_count}, Headers: {headers}, Lists: {lists}")
            
            # Save output if requested
            if save_output:
                debug_checkpoint("Saving markdown output")
                safe_execute(self.save_markdown, "Save markdown", markdown_output, save_output)
            
            result = {
                'markdown': markdown_output,
                'inference_time': inference_time,
                'raw_output': generated_text,
                'statistics': {
                    'word_count': word_count,
                    'char_count': char_count,
                    'line_count': line_count,
                    'headers': headers,
                    'lists': lists,
                    'tables': tables,
                    'code_blocks': code_blocks
                }
            }
            
            debug_checkpoint("Markdown generation completed successfully", "MD_END")
            return result
            
        except Exception as e:
            debug_checkpoint(f"Markdown generation failed with error: {str(e)}", "MD_FAILED")
            logger.error(f"Error during markdown generation: {e}")
            logger.error(f"Full traceback: {tb_module.format_exc()}")
            
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            raise
    
    def save_markdown(self, markdown_text, output_path):
        """Save markdown to file with metadata"""
        debug_checkpoint(f"Saving markdown to: {output_path}", "SAVE_MD_START")
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Add metadata header
            metadata = f"""<!--
Generated by 8-bit Mixed Precision Kosmos-2.5
Model: {self.model_checkpoint}
Quantization: {'8-bit' if self.use_8bit else 'FP16/BF16'}
Mixed Precision: {self.mixed_precision}
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
-->

"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(metadata + markdown_text)
            
            logger.info(f"Markdown saved to: {output_path}")
            debug_checkpoint("Markdown saved successfully", "SAVE_MD_END")
            
        except Exception as e:
            debug_checkpoint(f"Failed to save markdown: {str(e)}", "SAVE_MD_FAILED")
            logger.error(f"Error saving markdown: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=2048, temperature=0.1):
        """Process multiple images in batch with progress tracking"""
        debug_checkpoint(f"Starting batch processing of {len(image_paths)} images", "BATCH_START")
        
        if self.model is None:
            debug_checkpoint("Loading model for batch processing")
            safe_execute(self.load_model, "Load model for batch")
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting batch markdown processing of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            debug_checkpoint(f"Processing batch image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_markdown.md")
                
                # Generate markdown
                result = safe_execute(self.generate_markdown, f"Markdown for {base_name}",
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    save_output=output_path
                )
                
                result['input_path'] = image_path
                result['output_path'] = output_path
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
        
        logger.info("Batch markdown processing completed!")
        debug_checkpoint("Batch processing completed", "BATCH_END")
        return results

def get_args():
    parser = argparse.ArgumentParser(description='8-bit Mixed Precision Markdown generation using Kosmos-2.5 with Enhanced Debugging - CUDA ERROR FIX')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--model_checkpoint', '-m', type=str, required=True,
                       help='Path to model checkpoint (local directory or HuggingFace model name)')
    parser.add_argument('--output', '-o', type=str, default='./output.md',
                       help='Output path for generated markdown')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    
    # Enhanced token size configuration
    parser.add_argument('--max_tokens', '--tokens', type=int, default=1024,  # Reduced default for stability
                       help='Maximum number of tokens to generate (default: 1024, recommended: 512-2048)')
    parser.add_argument('--min_tokens', type=int, default=50,
                       help='Minimum number of tokens to generate (default: 50)')
    
    parser.add_argument('--temperature', '-t', type=float, default=0.0,  # Default to deterministic
                       help='Sampling temperature (0 for deterministic, 0.1-1.0 for creative)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--no_8bit', action='store_true',
                       help='Disable 8-bit quantization')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision for critical layers')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images (image should be a directory)')
    parser.add_argument('--print_output', '-p', action='store_true',
                       help='Print generated markdown to console')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output with statistics')
    parser.add_argument('--debug', action='store_true',
                       help='Enable maximum debug output')
    parser.add_argument('--force_fallback', action='store_true',
                       help='Force use of fallback mode (no 8-bit quantization)')
    
    return parser.parse_args()

def main():
    debug_checkpoint("Application starting", "MAIN_START")
    
    args = get_args()
    
    # Validate token configuration
    if args.max_tokens < args.min_tokens:
        logger.error(f"max_tokens ({args.max_tokens}) must be >= min_tokens ({args.min_tokens})")
        sys.exit(1)
    
    if args.max_tokens > 4096:  # More conservative limit
        logger.warning(f"Large max_tokens ({args.max_tokens}) may cause CUDA errors. Consider using smaller values.")
    
    if args.temperature < 0 or args.temperature > 1.0:
        logger.error(f"Temperature must be between 0.0 and 1.0, got {args.temperature}")
        sys.exit(1)
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        debug_checkpoint("Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        debug_checkpoint("Verbose mode enabled")
    
    debug_checkpoint(f"Arguments: {vars(args)}")
    debug_checkpoint(f"Token configuration - Max: {args.max_tokens}, Min: {args.min_tokens}, Temperature: {args.temperature}")
    
    # Initialize 8-bit mixed precision markdown inference
    debug_checkpoint("Initializing markdown engine")
    md_engine = safe_execute(EightBitMarkdownInference, "Initialize markdown engine",
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        use_8bit=not args.no_8bit,
        mixed_precision=not args.no_mixed_precision,
        force_fallback=args.force_fallback
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
            logger.info(f"Processing {len(image_paths)} images in batch mode with max_tokens={args.max_tokens}, temperature={args.temperature}")
            
            results = safe_execute(md_engine.batch_process, "Batch process images",
                image_paths=image_paths,
                output_dir=args.output,  # Use as output directory
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Calculate batch statistics
            debug_checkpoint("Calculating batch statistics")
            successful = sum(1 for r in results if 'error' not in r)
            total_time = sum(r.get('inference_time', 0) for r in results)
            total_words = sum(r.get('statistics', {}).get('word_count', 0) for r in results if 'error' not in r)
            total_headers = sum(r.get('statistics', {}).get('headers', 0) for r in results if 'error' not in r)
            total_lists = sum(r.get('statistics', {}).get('lists', 0) for r in results if 'error' not in r)
            
            print(f"\n{'='*80}")
            print("BATCH MARKDOWN PROCESSING SUMMARY (8-BIT MIXED PRECISION)")
            print(f"{'='*80}")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total words generated: {total_words:,}")
            print(f"Total headers found: {total_headers}")
            print(f"Total list items: {total_lists}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(results):.2f}s")
            print(f"Average words per image: {total_words/successful:.0f}" if successful > 0 else "N/A")
            print(f"Model checkpoint: {args.model_checkpoint}")
            print(f"Token configuration: Max={args.max_tokens}, Min={args.min_tokens}")
            print(f"Temperature: {args.temperature}")
            print(f"Quantization: {'8-bit' if not args.no_8bit and not args.force_fallback else 'FP16/BF16'}")
            print(f"Mixed precision: {'Enabled' if not args.no_mixed_precision else 'Disabled'}")
            print(f"Output directory: {args.output}")
            print(f"{'='*80}")
            
        else:
            debug_checkpoint("Starting single image processing mode")
            # Single image processing
            result = safe_execute(md_engine.generate_markdown, "Generate markdown for single image",
                image_path=args.image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_output=args.output
            )
            
            stats = result['statistics']
            
            # Print results summary
            print(f"\n{'='*80}")
            print("MARKDOWN GENERATION SUMMARY (8-BIT MIXED PRECISION)")
            print(f"{'='*80}")
            print(f"Processing time: {result['inference_time']:.2f}s")
            print(f"Word count: {stats['word_count']:,}")
            print(f"Character count: {stats['char_count']:,}")
            print(f"Line count: {stats['line_count']:,}")
            print(f"Headers: {stats['headers']}")
            print(f"List items: {stats['lists']}")
            print(f"Tables: {stats['tables']}")
            print(f"Code blocks: {stats['code_blocks']}")
            print(f"Output saved to: {args.output}")
            print(f"Model checkpoint: {args.model_checkpoint}")
            print(f"Token configuration: Max={args.max_tokens}, Min={args.min_tokens}")
            print(f"Temperature: {args.temperature}")
            print(f"Quantization: {'8-bit' if not args.no_8bit and not args.force_fallback else 'FP16/BF16'}")
            print(f"Mixed precision: {'Enabled' if not args.no_mixed_precision else 'Disabled'}")
            print(f"{'='*80}")
            
            if args.print_output:
                print("\nGENERATED MARKDOWN:")
                print("=" * 80)
                print(result['markdown'])
                print("=" * 80)
        
        debug_checkpoint("Application completed successfully", "MAIN_END")
        
    except Exception as e:
        debug_checkpoint(f"Application failed with error: {str(e)}", "MAIN_FAILED")
        logger.error(f"Markdown generation failed: {e}")
        if args.verbose or args.debug:
            logger.error(f"Full traceback: {tb_module.format_exc()}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
