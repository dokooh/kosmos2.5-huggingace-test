#!/usr/bin/env python3
"""
8-bit Mixed Precision Markdown Generation for Kosmos-2.5

This module provides fast markdown generation using 8-bit mixed precision quantized Kosmos-2.5 model.
Features:
- 8-bit mixed precision quantization for faster inference and reduced memory usage
- BitsAndBytesConfig for advanced quantization settings
- SafeTensors format support for faster loading
- Support for local model checkpoints and remote models
- Optimized memory usage with gradient checkpointing
- Enhanced markdown post-processing with structure detection
- Batch processing support with progress tracking
- Comprehensive error handling and fallback mechanisms
"""

import torch
import requests
import argparse
import sys
import os
import time
import re
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EightBitMarkdownInference:
    def __init__(self, model_checkpoint, device=None, cache_dir=None, use_8bit=True, mixed_precision=True):
        """
        Initialize 8-bit Mixed Precision Markdown inference
        
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
        
        logger.info(f"Initializing 8-bit Mixed Precision Markdown inference on {self.device}")
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
                    # Skip critical layers for better quality
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
    
    def post_process_markdown(self, generated_text, prompt="<md>"):
        """Post-process and clean up generated markdown with enhanced structure detection"""
        # Remove the prompt
        markdown = generated_text.replace(prompt, "").strip()
        
        # Enhanced markdown cleaning
        markdown = self.clean_markdown_advanced(markdown)
        
        return markdown
    
    def clean_markdown_advanced(self, text):
        """Advanced markdown cleaning with structure detection"""
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
        
        return text
    
    def generate_markdown(self, image_path, max_tokens=2048, temperature=0.1, save_output=None):
        """Generate markdown from image using 8-bit mixed precision"""
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
            
            # Move inputs to device and convert to correct dtype if not using 8-bit
            if not self.use_8bit:
                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
                
                # Convert flattened_patches to correct dtype
                if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                    inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            else:
                # For 8-bit models, just move to device (quantization handles dtype)
                inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            # Generate markdown
            logger.info("Generating markdown with 8-bit mixed precision...")
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
                            do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else None,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            length_penalty=1.0,
                            use_cache=True,
                            num_beams=1 if temperature > 0 else 1,
                            early_stopping=True,
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                        use_cache=True,
                        num_beams=1,
                        early_stopping=True,
                    )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process markdown
            markdown_output = self.post_process_markdown(generated_text, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"Markdown generation completed in {inference_time:.2f}s")
            
            # Calculate statistics
            word_count = len(markdown_output.split())
            char_count = len(markdown_output)
            line_count = len(markdown_output.split('\n'))
            
            # Detect markdown elements
            headers = len(re.findall(r'^#+\s', markdown_output, re.MULTILINE))
            lists = len(re.findall(r'^[\*\-\+]\s|^\d+\.\s', markdown_output, re.MULTILINE))
            tables = markdown_output.count('|')
            code_blocks = markdown_output.count('```') // 2
            
            # Save output if requested
            if save_output:
                self.save_markdown(markdown_output, save_output)
            
            return {
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
            
        except Exception as e:
            logger.error(f"Error during markdown generation: {e}")
            raise
    
    def save_markdown(self, markdown_text, output_path):
        """Save markdown to file with metadata"""
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
            
        except Exception as e:
            logger.error(f"Error saving markdown: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=2048, temperature=0.1):
        """Process multiple images in batch with progress tracking"""
        if self.model is None:
            self.load_model()
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting batch processing of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
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
        
        logger.info("Batch processing completed!")
        return results

def get_args():
    parser = argparse.ArgumentParser(description='8-bit Mixed Precision Markdown generation using Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--model_checkpoint', '-m', type=str, required=True,
                       help='Path to model checkpoint (local directory or HuggingFace model name)')
    parser.add_argument('--output', '-o', type=str, default='./output.md',
                       help='Output path for generated markdown')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Sampling temperature (0 for deterministic)')
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
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize 8-bit mixed precision markdown inference
    md_engine = EightBitMarkdownInference(
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
            results = md_engine.batch_process(
                image_paths=image_paths,
                output_dir=args.output,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Calculate batch statistics
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
            print(f"Quantization: {'8-bit' if not args.no_8bit else 'FP16/BF16'}")
            print(f"Mixed precision: {'Enabled' if not args.no_mixed_precision else 'Disabled'}")
            print(f"Output directory: {args.output}")
            print(f"{'='*80}")
            
        else:
            # Single image processing
            result = md_engine.generate_markdown(
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
            print(f"Quantization: {'8-bit' if not args.no_8bit else 'FP16/BF16'}")
            print(f"Mixed precision: {'Enabled' if not args.no_mixed_precision else 'Disabled'}")
            print(f"{'='*80}")
            
            if args.print_output:
                print("\nGENERATED MARKDOWN:")
                print("=" * 80)
                print(result['markdown'])
                print("=" * 80)
        
    except Exception as e:
        logger.error(f"Markdown generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()