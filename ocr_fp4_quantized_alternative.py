"""
OCR inference script for quantized Kosmos-2.5 models
Supports NF4_DoubleQuant and FP4_BF16 quantized models
"""

import re
import torch
import requests
import argparse
import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration, BitsAndBytesConfig

def get_args():
    parser = argparse.ArgumentParser(description='Perform OCR on image using quantized Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True, 
                       help='Path to input image file or URL')
    parser.add_argument('--model_path', '-mp', type=str, required=True,
                       help='Path to quantized model directory (e.g., ./quantized_models/NF4_DoubleQuant)')
    parser.add_argument('--output', '-o', type=str, default='./output_quantized.png',
                       help='Output path for the annotated image (default: ./output_quantized.png)')
    parser.add_argument('--text_output', '-t', type=str, default=None,
                       help='Output path for the OCR text results (optional)')
    parser.add_argument('--device', '-d', type=str, default='auto', 
                       help='Device to use (default: auto, options: auto, cpu, cuda:0)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum number of tokens to generate (default: 1024)')
    parser.add_argument('--no_bbox', action='store_true',
                       help='Skip drawing bounding boxes on the output image')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
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
        elif 'FP4' in method:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4"
            )
        elif 'INT8' in method:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # Fallback - try to detect from model files
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
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

def post_process(y, scale_height, scale_width, prompt="<ocr>"):
    """Post-process OCR output to extract bounding boxes and text"""
    y = y.replace(prompt, "")
    if "<md>" in prompt:
        return y
    
    pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
    bboxs_raw = re.findall(pattern, y)
    lines = re.split(pattern, y)[1:]
    bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
    bboxs = [[int(j) for j in i] for i in bboxs]
    
    info = ""
    for i in range(len(lines)):
        if i < len(bboxs):
            box = bboxs[i]
            if len(box) >= 4:
                x0, y0, x1, y1 = box[:4]
                if not (x0 >= x1 or y0 >= y1):
                    x0 = int(x0 * scale_width)
                    y0 = int(y0 * scale_height)
                    x1 = int(x1 * scale_width)
                    y1 = int(y1 * scale_height)
                    info += f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{lines[i].strip()}\n"
    
    return info

def draw_bounding_boxes(image, output_text):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    lines = output_text.strip().split("\n")
    
    for line in lines:
        if not line.strip():
            continue
            
        # Draw the bounding box
        parts = line.split(",")
        if len(parts) < 8:
            continue
            
        try:
            coords = list(map(int, parts[:8]))
            draw.polygon(coords, outline="red", width=2)
        except (ValueError, IndexError):
            # Skip lines that don't have valid coordinates
            continue
    
    return image

def save_text_output(text, output_path):
    """Save OCR text results to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"OCR text saved to: {output_path}")
    except Exception as e:
        print(f"Error saving text output: {e}")

def benchmark_inference(model, processor, inputs, max_tokens, verbose=False):
    """Benchmark inference time and memory usage"""
    import time
    import psutil
    import gc
    
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
    
    # Run inference
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic for consistency
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
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
    
    return generated_ids, {
        'inference_time': inference_time,
        'cpu_memory_increase': memory_increase,
        'gpu_memory_increase': gpu_memory_increase
    }

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
    prompt = "<ocr>"
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
            inputs["flattened_patches"] = inputs["flattened_patches"].to(torch.float16)
        
    except Exception as e:
        print(f"Error processing inputs: {e}")
        sys.exit(1)
    
    # Generate OCR results with benchmarking
    print("Performing OCR...")
    try:
        generated_ids, benchmark_results = benchmark_inference(
            model, processor, inputs, args.max_tokens, args.verbose
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = post_process(generated_text[0], scale_height, scale_width, prompt)
        
        print("\n" + "="*50)
        print("OCR RESULTS:")
        print("="*50)
        print(output_text)
        print("="*50)
        
        if args.verbose:
            print("\n" + "="*50)
            print("PERFORMANCE METRICS:")
            print("="*50)
            print(f"Model: {method}")
            print(f"Inference time: {benchmark_results['inference_time']:.2f}s")
            print(f"CPU memory increase: {benchmark_results['cpu_memory_increase']:.1f}MB")
            if benchmark_results['gpu_memory_increase'] > 0:
                print(f"GPU memory increase: {benchmark_results['gpu_memory_increase']:.1f}MB")
            print("="*50)
        
        # Save text output if specified
        if args.text_output:
            save_text_output(output_text, args.text_output)
        
        # Create output image with bounding boxes (unless disabled)
        if not args.no_bbox:
            # Create a copy of the original image for annotation
            annotated_image = image.copy()
            annotated_image = draw_bounding_boxes(annotated_image, output_text)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the annotated image
            annotated_image.save(args.output)
            print(f"Annotated image saved to: {args.output}")
        else:
            print("Bounding box drawing skipped (--no_bbox flag used)")
        
    except Exception as e:
        print(f"Error during OCR generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
