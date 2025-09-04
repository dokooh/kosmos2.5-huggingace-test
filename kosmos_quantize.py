#!/usr/bin/env python3
"""
Kosmos-2.5 Model Quantization Script

This script provides multiple quantization approaches for the Microsoft Kosmos-2.5 model:
1. BitsAndBytes (8-bit/4-bit) - Easy to use, good for inference
2. GPTQ (4-bit) - Better compression with minimal accuracy loss
3. AWQ (4-bit) - Fastest inference with activation-aware quantization

Choose the method based on your requirements:
- BitsAndBytes: Quick setup, good memory reduction
- GPTQ: Best balance of size/accuracy for general use
- AWQ: Best for inference speed and throughput
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    BitsAndBytesConfig,
    GPTQConfig
)

# Handle different model import possibilities
try:
    from transformers import Kosmos2_5ForConditionalGeneration
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as Kosmos2_5ForConditionalGeneration
        logger.warning("Using AutoModelForVision2Seq as fallback for Kosmos2_5ForConditionalGeneration")
    except ImportError:
        logger.error("Could not import Kosmos2_5ForConditionalGeneration or AutoModelForVision2Seq")
        raise

from optimum.gptq import GPTQQuantizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KosmosQuantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.processor = None
        self.model = None
        
    def load_tokenizer_and_processor(self):
        """Load tokenizer and processor"""
        logger.info("Loading tokenizer and processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir
        )
        
    def quantize_bitsandbytes(self, bits=8, save_path="./kosmos2.5-bnb-quantized"):
        """
        Quantize using BitsAndBytes (8-bit or 4-bit)
        
        Args:
            bits (int): 4 or 8 bit quantization
            save_path (str): Path to save quantized model
        """
        logger.info(f"Starting BitsAndBytes {bits}-bit quantization...")
        
        if bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        elif bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Normal Float 4
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization
            )
        else:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
            
        # Load model with quantization config
        self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )
        
        # Save quantized model
        self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        if self.processor:
            self.processor.save_pretrained(save_path)
            
        logger.info(f"BitsAndBytes {bits}-bit quantized model saved to {save_path}")
        return self.model
        
    def quantize_gptq(self, bits=4, save_path="./kosmos2.5-gptq-quantized", dataset_path=None):
        """
        Quantize using GPTQ (4-bit recommended)
        
        Args:
            bits (int): Number of bits (4 recommended)
            save_path (str): Path to save quantized model
            dataset_path (str): Path to calibration dataset (optional)
        """
        logger.info(f"Starting GPTQ {bits}-bit quantization...")
        
        # Load unquantized model first
        self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir
        )
        
        # Configure GPTQ
        gptq_config = GPTQConfig(
            bits=bits,
            group_size=128,
            desc_act=False,  # Disable activation ordering for stability
        )
        
        # Create quantizer
        quantizer = GPTQQuantizer(
            bits=bits,
            group_size=128,
            desc_act=False,
        )
        
        # Prepare calibration data (you can replace this with domain-specific data)
        if dataset_path is None:
            calibration_data = self._prepare_calibration_data()
        else:
            calibration_data = self._load_custom_dataset(dataset_path)
        
        # Quantize the model
        quantized_model = quantizer.quantize_model(
            self.model, 
            tokenizer=self.tokenizer,
            calibration_dataset=calibration_data
        )
        
        # Save quantized model
        quantized_model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        if self.processor:
            self.processor.save_pretrained(save_path)
            
        logger.info(f"GPTQ {bits}-bit quantized model saved to {save_path}")
        return quantized_model
    
    def quantize_awq(self, save_path="./kosmos2.5-awq-quantized"):
        """
        Quantize using AWQ (Activation-aware Weight Quantization)
        Note: This requires the awq library to be installed
        """
        try:
            from awq import AutoAWQForCausalLM
            from awq.quantize.quantizer import AwqQuantizer
        except ImportError:
            raise ImportError("AWQ library not installed. Install with: pip install autoawq")
            
        logger.info("Starting AWQ 4-bit quantization...")
        
        # AWQ quantization config
        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }
        
        # Load model for AWQ quantization
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto",
            cache_dir=self.cache_dir
        )
        
        # Prepare calibration data
        calibration_data = self._prepare_calibration_data()
        
        # Quantize
        model.quantize(
            tokenizer=self.tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data
        )
        
        # Save
        model.save_quantized(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        if self.processor:
            self.processor.save_pretrained(save_path)
            
        logger.info(f"AWQ quantized model saved to {save_path}")
        return model
    
    def _prepare_calibration_data(self):
        """Prepare calibration data for quantization"""
        # Create sample prompts for multimodal tasks
        calibration_prompts = [
            "<ocr>",
            "<md>", 
            "What is in this image?",
            "Describe the contents of this document.",
            "Extract the text from this image.",
        ]
        
        if not self.tokenizer or not self.processor:
            self.load_tokenizer_and_processor()
            
        # Prepare calibration data with dummy images
        calibration_data = []
        for prompt in calibration_prompts:
            # Create a dummy image for each prompt
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            
            try:
                # Use processor for multimodal inputs
                inputs = self.processor(
                    text=prompt,
                    images=dummy_image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Remove problematic keys
                filtered_inputs = {k: v for k, v in inputs.items() 
                                 if k not in ['token_type_ids']}
                
                calibration_data.append(filtered_inputs)
                
            except Exception as e:
                logger.warning(f"Failed to process calibration prompt '{prompt}': {e}")
                # Fallback to text-only tokenization
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                # Remove problematic keys
                filtered_inputs = {k: v for k, v in inputs.items() 
                                 if k not in ['token_type_ids']}
                calibration_data.append(filtered_inputs)
                
        return calibration_data
    
    def _load_custom_dataset(self, dataset_path):
        """Load custom calibration dataset"""
        # Implement custom dataset loading logic here
        logger.info(f"Loading custom dataset from {dataset_path}")
        # This is a placeholder - implement based on your dataset format
        return self._prepare_calibration_data()
    
    def benchmark_model(self, model, test_iterations=10):
        """Benchmark model inference speed and memory usage"""
        logger.info("Benchmarking model performance...")
        
        # For Kosmos-2.5, we need both text and image inputs
        # Create a dummy image for benchmarking
        from PIL import Image
        import numpy as np
        
        # Create a small dummy image (RGB)
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Prepare inputs using the processor
        dummy_text = "<ocr>"  # or "<md>" for markdown
        
        if self.processor is None:
            self.load_tokenizer_and_processor()
        
        # Use processor instead of tokenizer for multimodal inputs
        inputs = self.processor(
            text=dummy_text,
            images=dummy_image,
            return_tensors="pt"
        )
        
        # Remove any unused parameters that might cause issues
        # Keep only the essential inputs for generation
        essential_inputs = {}
        for key in ['input_ids', 'attention_mask', 'pixel_values', 'image_embeds']:
            if key in inputs:
                essential_inputs[key] = inputs[key]
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        essential_inputs = {k: v.to(device) if v is not None else v 
                           for k, v in essential_inputs.items()}
        
        # Warm up
        logger.info("Warming up model...")
        with torch.no_grad():
            try:
                for _ in range(3):
                    _ = model.generate(
                        **essential_inputs,
                        max_new_tokens=50,  # Use max_new_tokens instead of max_length
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")
                # Fallback to simpler generation
                for _ in range(3):
                    _ = model.generate(
                        input_ids=essential_inputs.get('input_ids'),
                        max_new_tokens=20,
                        do_sample=False
                    )
        
        # Benchmark
        logger.info(f"Running benchmark with {test_iterations} iterations...")
        import time
        start_time = time.time()
        
        successful_runs = 0
        with torch.no_grad():
            for i in range(test_iterations):
                try:
                    outputs = model.generate(
                        **essential_inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    successful_runs += 1
                    
                    if i == 0:
                        # Decode and show sample output
                        try:
                            decoded_output = self.processor.batch_decode(
                                outputs, skip_special_tokens=True
                            )[0]
                            logger.info(f"Sample output: {decoded_output}")
                        except Exception as e:
                            logger.info(f"Could not decode output: {e}")
                        
                except Exception as e:
                    logger.warning(f"Generation failed on iteration {i}: {e}")
                    continue
                
        end_time = time.time()
        
        if successful_runs == 0:
            logger.error("All benchmark iterations failed!")
            return None
            
        avg_time = (end_time - start_time) / successful_runs
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"GPU Memory used: {memory_used:.2f} GB")
            
        logger.info(f"Successful runs: {successful_runs}/{test_iterations}")
        logger.info(f"Average inference time: {avg_time:.3f} seconds")
        return avg_time

def main():
    parser = argparse.ArgumentParser(description='Quantize Kosmos-2.5 model')
    parser.add_argument('--method', choices=['bnb', 'gptq', 'awq'], required=True,
                       help='Quantization method')
    parser.add_argument('--bits', type=int, choices=[4, 8], default=4,
                       help='Number of bits for quantization')
    parser.add_argument('--model_name', default='microsoft/kosmos-2.5',
                       help='Model name or path')
    parser.add_argument('--save_path', required=True,
                       help='Path to save quantized model')
    parser.add_argument('--cache_dir', default=None,
                       help='Cache directory for model downloads')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after quantization')
    parser.add_argument('--dataset_path', default=None,
                       help='Path to calibration dataset (for GPTQ)')
    
    args = parser.parse_args()
    
    # Initialize quantizer
    quantizer = KosmosQuantizer(args.model_name, args.cache_dir)
    quantizer.load_tokenizer_and_processor()
    
    # Perform quantization
    if args.method == 'bnb':
        model = quantizer.quantize_bitsandbytes(args.bits, args.save_path)
    elif args.method == 'gptq':
        model = quantizer.quantize_gptq(args.bits, args.save_path, args.dataset_path)
    elif args.method == 'awq':
        model = quantizer.quantize_awq(args.save_path)
    
    # Benchmark if requested
    if args.benchmark:
        quantizer.benchmark_model(model)
    
    logger.info("Quantization completed successfully!")

if __name__ == "__main__":
    # Example usage:
    # python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit
    # python kosmos_quantize.py --method gptq --bits 4 --save_path ./kosmos2.5-gptq4bit
    # python kosmos_quantize.py --method awq --save_path ./kosmos2.5-awq4bit
    
    main()
