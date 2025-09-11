"""
Markdown inference script for quantized Kosmos-2.5 models
Supports NF4_DoubleQuant and FP4_BF16 quantized models
"""

import re
import torch
import requests
import argparse
import sys
import os
import json
import time
import psutil
import gc
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration, BitsAndBytesConfig

def get_args():
    parser = argparse.ArgumentParser(description='Generate markdown from image using quantized Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True, 
                       help='Path to input image file or URL')
    parser.add_argument('--model_path', '-mp', type=str, required=True,
                       help='Path to quantized model directory (e.g., ./quantized_models/NF4_DoubleQuant)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for markdown (optional, prints to console if not specified)')
    parser.add_argument('--device', '-d', type=str, default='auto', 
                       help='Device to use (default: auto, options: auto, cpu, cuda:0)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum number of tokens to generate (default: 1024)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run performance benchmark')
    return parser.parse_args()

def load_quantized_model(model_path, device='auto', verbose=False):
    """Load quantized model with appropriate configuration"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load metadata to determine quantization method
    metadata_path = model_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        method = metadata.get('method', 'unknown')
        if verbose:
            print(f"Loading quantized model: {method}")
            print(f"Model size: {metadata.get('size_mb', 'unknown')} MB")
            print(f"Quantization time: {metadata.get('quantization_time', 'unknown')}s")
    else:
        method = model_path.name
        if verbose:
            print(f"Loading model from: {model_path}")
    
    try:
        # Configure quantization based on method
        if 'NF4' in method:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            if verbose:
                print("Using NF4 quantization with double quantization")
        elif 'FP4' in method:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4"
            )
            if verbose:
                print("Using FP4 quantization with BF16 compute")
        elif 'INT8' in method:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            if verbose:
                print("Using INT8 quantization")
        else:
            # Fallback - try to detect from model files
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            if verbose:
                print("Using fallback NF4 quantization")
        
        # Load model
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        if verbose:
            print("Model and processor loaded successfully")
            
        return model, processor, method
        
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        print("Trying fallback loading method...")
        
        # Fallback: try loading as regular model
        try:
            model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                str(model_path),
                device_map=device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            return model, processor, method
        except Exception as fallback_e:
            raise Exception(f"Both loading methods failed. Original: {e}, Fallback: {fallback_e}")

def load_image(image_path, verbose=False):
    """Load image from local path or URL"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load from URL
            if verbose:
                print(f"Loading image from URL: {image_path}")
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            # Load from local file
            if verbose:
                print(f"Loading image from file: {image_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path)
        
        return image.convert('RGB')  # Ensure RGB format
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def post_process_markdown(generated_text, prompt="<md>"):
    """Post-process markdown output"""
    # Remove the prompt from the beginning
    result = generated_text.replace(prompt, "").strip()
    
    # Clean up any artifact tokens
    result = re.sub(r'<[^>]+>', '', result)  # Remove any remaining special tokens
    
    # Fix common markdown formatting issues
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Remove excessive newlines
    result = re.sub(r'^\s+', '', result, flags=re.MULTILINE)  # Remove leading whitespace
    
    return result

def benchmark_inference(model, processor, inputs, max_tokens, verbose=False):
    """Benchmark inference time and memory usage"""
    if verbose:
        print("Starting inference benchmark...")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Record initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    if torch.cuda.is_available():
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        torch.cuda.synchronize()
    
    # Run inference
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic for consistency
            pad_token_id=processor.tokenizer.eos_token_id,
            temperature=1.0,
            top_p=1.0
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = time.time() - start_time
    
    # Record final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_increase = final_gpu_memory - initial_gpu_memory
    else:
        gpu_memory_increase = 0
    
    if verbose:
        print(f"Inference time: {inference_time:.2f}s")
        print(f"CPU memory increase: {memory_increase:.1f}MB")
        if torch.cuda.is_available():
            print(f"GPU memory increase: {gpu_memory_increase:.1f}MB")
            print(f"Tokens/second: {max_tokens/inference_time:.1f}")
    
    return generated_ids, {
        'inference_time': inference_time,
        'cpu_memory_increase': memory_increase,
        'gpu_memory_increase': gpu_memory_increase,
        'tokens_per_second': max_tokens / inference_time if inference_time > 0 else 0
    }

def save_markdown_output(text, output_path):
    """Save markdown text to file"""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Markdown saved to: {output_path}")
    except Exception as e:
        print(f"Error saving markdown output: {e}")

def format_output(text, method, benchmark_results=None):
    """Format the output with metadata"""
    output = f"# Generated Markdown\n\n"
    output += f"*Generated by Kosmos-2.5 ({method})*\n\n"
    
    if benchmark_results:
        output += f"*Generation time: {benchmark_results['inference_time']:.2f}s*\n\n"
    
    output += "---\n\n"
    output += text
    
    return output

def main():
    args = get_args()
    
    # Load quantized model
    print("Loading quantized model...")
    try:
        model, processor, method = load_quantized_model(
            args.model_path, 
            args.device, 
            args.verbose
        )
        print(f"Quantized model loaded successfully: {method}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load image
    image = load_image(args.image, args.verbose)
    if args.verbose:
        print(f"Image loaded successfully. Size: {image.size}")
    
    # Process image
    prompt = "<md>"
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        height, width = inputs.pop("height"), inputs.pop("width")
        raw_width, raw_height = image.size
        scale_height = raw_height / height
        scale_width = raw_width / width
        
        # Move inputs to appropriate device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
        
        # Ensure proper dtype for quantized models
        if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
            if 'FP4_BF16' in method:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(torch.bfloat16)
            else:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(torch.float16)
        
    except Exception as e:
        print(f"Error processing inputs: {e}")
        sys.exit(1)
    
    # Generate markdown
    print("Generating markdown...")
    try:
        if args.benchmark:
            generated_ids, benchmark_results = benchmark_inference(
                model, processor, inputs, args.max_tokens, args.verbose
            )
        else:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            benchmark_results = None
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract and clean the generated markdown
        result = post_process_markdown(generated_text[0], prompt)
        
        print("\n" + "="*50)
        print("GENERATED MARKDOWN:")
        print("="*50)
        print(result)
        print("="*50)
        
        if args.verbose and benchmark_results:
            print("\n" + "="*50)
            print("PERFORMANCE METRICS:")
            print("="*50)
            print(f"Model: {method}")
            print(f"Inference time: {benchmark_results['inference_time']:.2f}s")
            print(f"Tokens/second: {benchmark_results['tokens_per_second']:.1f}")
            print(f"CPU memory increase: {benchmark_results['cpu_memory_increase']:.1f}MB")
            if benchmark_results['gpu_memory_increase'] > 0:
                print(f"GPU memory increase: {benchmark_results['gpu_memory_increase']:.1f}MB")
            print("="*50)
        
        # Save output if specified
        if args.output:
            formatted_output = format_output(result, method, benchmark_results)
            save_markdown_output(formatted_output, args.output)
            
            # Also save raw output
            raw_output_path = args.output.replace('.md', '_raw.md')
            save_markdown_output(result, raw_output_path)
        
    except Exception as e:
        print(f"Error during markdown generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
