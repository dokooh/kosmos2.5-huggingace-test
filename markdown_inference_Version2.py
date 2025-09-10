"""
Markdown inference script using saved FP4 quantized Kosmos-2.5 model
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
from pathlib import Path

class Kosmos25MarkdownInference:
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
        
        # Check for metadata
        metadata_path = os.path.join(self.model_path, "quantization_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Model info:")
            print(f"  Quantization: {metadata.get('quantization_type', 'unknown')}")
            print(f"  Size: {metadata.get('size_mb', 0):.1f} MB")
            print(f"  Parameters: {metadata.get('parameter_count', 0):,}")
        
        try:
            start_time = time.time()
            
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
            
            load_time = time.time() - start_time
            model_dtype = next(self.model.parameters()).dtype
            
            print(f"✓ Model loaded in {load_time:.2f}s")
            print(f"  Model dtype: {model_dtype}")
            print(f"  Device: {next(self.model.parameters()).device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def load_image(self, image_input):
        """Load image from various sources"""
        if image_input.startswith(('http://', 'https://')):
            # Load from URL
            print(f"Loading image from URL: {image_input}")
            try:
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from URL: {e}")
        
        elif os.path.exists(image_input):
            # Load from local file
            print(f"Loading image from file: {image_input}")
            try:
                image = Image.open(image_input)
                return image.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from file: {e}")
        
        elif image_input == "default":
            # Create default test image with some structure
            print("Using default test image (document-like)")
            from PIL import ImageDraw, ImageFont
            
            image = Image.new('RGB', (800, 600), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            # Try to use a basic font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 24)
                small_font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw sample document content
            draw.text((50, 50), "Document Title", fill=(0, 0, 0), font=font)
            draw.text((50, 100), "This is a sample document with multiple paragraphs.", fill=(0, 0, 0), font=small_font)
            draw.text((50, 130), "It contains structured text that can be converted to markdown.", fill=(0, 0, 0), font=small_font)
            draw.text((50, 180), "• Bullet point 1", fill=(0, 0, 0), font=small_font)
            draw.text((50, 210), "• Bullet point 2", fill=(0, 0, 0), font=small_font)
            draw.text((50, 260), "Section Header", fill=(0, 0, 0), font=font)
            draw.text((50, 310), "More content here with different formatting.", fill=(0, 0, 0), font=small_font)
            
            return image
        
        else:
            raise ValueError(f"Invalid image input: {image_input}")
    
    def run_markdown_generation(self, image, extract_markdown_only=True):
        """Run markdown generation inference on the image"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            print("Running markdown generation inference...")
            start_time = time.time()
            
            # Prepare inputs for markdown generation
            model_dtype = next(self.model.parameters()).dtype
            inputs = self.processor(text="<md>", images=image, return_tensors="pt")
            
            # Convert inputs to correct dtype and device
            processed_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype.is_floating_point:
                        processed_inputs[key] = value.to(dtype=model_dtype, device=self.device)
                    else:
                        processed_inputs[key] = value.to(device=self.device)
                else:
                    processed_inputs[key] = value
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**processed_inputs)
            
            inference_time = time.time() - start_time
            print(f"✓ Markdown generation completed in {inference_time:.2f}s")
            
            # Process outputs
            result = self.process_markdown_output(outputs, extract_markdown_only)
            
            return {
                "success": True,
                "inference_time": inference_time,
                "result": result,
                "output_type": str(type(outputs))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "inference_time": 0,
                "result": None
            }
    
    def process_markdown_output(self, outputs, extract_markdown_only=True):
        """Process the model output to extract markdown results"""
        try:
            # Analyze output structure
            output_info = {
                "type": str(type(outputs)),
                "attributes": []
            }
            
            # Get available attributes
            for attr in dir(outputs):
                if not attr.startswith('_'):
                    try:
                        value = getattr(outputs, attr)
                        if isinstance(value, torch.Tensor):
                            output_info["attributes"].append({
                                "name": attr,
                                "shape": list(value.shape),
                                "dtype": str(value.dtype)
                            })
                    except:
                        pass
            
            if extract_markdown_only:
                # Try to extract markdown from common attributes
                markdown_candidates = []
                
                # Check for logits or similar attributes that might contain markdown
                for attr_info in output_info["attributes"]:
                    attr_name = attr_info["name"]
                    if "logits" in attr_name.lower() or "output" in attr_name.lower():
                        try:
                            tensor = getattr(outputs, attr_name)
                            if tensor.dim() >= 2:
                                # Try to decode as markdown
                                predicted_ids = torch.argmax(tensor, dim=-1)
                                if predicted_ids.dim() > 1:
                                    sample_ids = predicted_ids[0][:200]  # More tokens for markdown
                                else:
                                    sample_ids = predicted_ids[:200]
                                
                                try:
                                    decoded = self.processor.tokenizer.decode(sample_ids, skip_special_tokens=True)
                                    if decoded.strip():
                                        # Post-process to make it more markdown-like
                                        processed_markdown = self.post_process_markdown(decoded.strip())
                                        markdown_candidates.append({
                                            "source": attr_name,
                                            "raw_output": decoded.strip(),
                                            "processed_markdown": processed_markdown
                                        })
                                except:
                                    pass
                        except:
                            pass
                
                if markdown_candidates:
                    return {
                        "generated_markdown": markdown_candidates[0]["processed_markdown"],
                        "raw_output": markdown_candidates[0]["raw_output"],
                        "all_candidates": markdown_candidates,
                        "output_structure": output_info
                    }
                else:
                    return {
                        "generated_markdown": "No markdown could be generated",
                        "raw_output": "",
                        "all_candidates": [],
                        "output_structure": output_info
                    }
            else:
                return output_info
                
        except Exception as e:
            return {"error": f"Failed to process output: {e}"}
    
    def post_process_markdown(self, raw_text):
        """Post-process raw output to improve markdown formatting"""
        try:
            lines = raw_text.split('\n')
            processed_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    processed_lines.append("")
                    continue
                
                # Simple heuristics to improve markdown formatting
                if line.isupper() and len(line) < 50:
                    # Likely a header
                    processed_lines.append(f"# {line.title()}")
                elif line.startswith('-') or line.startswith('•'):
                    # Bullet point
                    processed_lines.append(f"- {line[1:].strip()}")
                elif line.endswith(':') and len(line) < 30:
                    # Likely a section header
                    processed_lines.append(f"## {line[:-1]}")
                else:
                    # Regular text
                    processed_lines.append(line)
            
            return '\n'.join(processed_lines)
            
        except Exception as e:
            return raw_text  # Return original if processing fails
    
    def batch_markdown_generation(self, image_paths, output_dir=None):
        """Run markdown generation on multiple images"""
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                image = self.load_image(image_path)
                result = self.run_markdown_generation(image)
                result["image_path"] = image_path
                results.append(result)
                
                if result["success"]:
                    print(f"✓ Markdown generation successful")
                    
                    # Save individual markdown file if output directory specified
                    if output_dir and "generated_markdown" in result["result"]:
                        image_name = Path(image_path).stem if image_path != "default" else f"image_{i+1}"
                        markdown_file = os.path.join(output_dir, f"{image_name}.md")
                        
                        with open(markdown_file, 'w', encoding='utf-8') as f:
                            f.write(result["result"]["generated_markdown"])
                        print(f"  Markdown saved to: {markdown_file}")
                        result["markdown_file"] = markdown_file
                else:
                    print(f"✗ Markdown generation failed: {result['error']}")
                    
            except Exception as e:
                print(f"✗ Failed to process {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": str(e),
                    "result": None
                })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Markdown generation using FP4 quantized Kosmos-2.5")
    parser.add_argument("--model_path", type=str, default="./fp4_quantized_model",
                       help="Path to saved FP4 quantized model")
    parser.add_argument("--image", type=str, default="default",
                       help="Image path, URL, or 'default' for test image")
    parser.add_argument("--batch", type=str, nargs='+',
                       help="Multiple image paths for batch processing")
    parser.add_argument("--output", type=str,
                       help="Output file to save results (JSON format)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory to save individual markdown files")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output structure")
    
    args = parser.parse_args()
    
    # Initialize markdown inference
    markdown_gen = Kosmos25MarkdownInference(model_path=args.model_path)
    
    try:
        # Load model
        markdown_gen.load_model()
        
        if args.batch:
            # Batch processing
            results = markdown_gen.batch_markdown_generation(args.batch, args.output_dir)
            
            # Print summary
            successful = sum(1 for r in results if r["success"])
            print(f"\n{'='*50}")
            print("BATCH MARKDOWN GENERATION SUMMARY")
            print('='*50)
            print(f"Total images: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            
            # Save batch results
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"Batch results saved to {args.output}")
            
        else:
            # Single image processing
            image = markdown_gen.load_image(args.image)
            result = markdown_gen.run_markdown_generation(image, extract_markdown_only=not args.verbose)
            
            print(f"\n{'='*50}")
            print("MARKDOWN GENERATION RESULTS")
            print('='*50)
            
            if result["success"]:
                print(f"✓ Markdown generation successful (inference time: {result['inference_time']:.2f}s)")
                
                if "generated_markdown" in result["result"]:
                    print(f"\nGenerated Markdown:")
                    print("-" * 30)
                    print(result["result"]["generated_markdown"])
                    
                    if args.verbose and "raw_output" in result["result"]:
                        print(f"\nRaw Output:")
                        print("-" * 30)
                        print(result["result"]["raw_output"])
                
                if args.verbose:
                    print(f"\nOutput structure:")
                    output_structure = result["result"].get("output_structure", {})
                    print(f"  Type: {output_structure.get('type', 'Unknown')}")
                    print(f"  Attributes: {len(output_structure.get('attributes', []))}")
                    for attr in output_structure.get('attributes', []):
                        print(f"    {attr['name']}: {attr['shape']} ({attr['dtype']})")
                
                # Save markdown to file
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    markdown_file = os.path.join(args.output_dir, "generated.md")
                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(result["result"]["generated_markdown"])
                    print(f"\nMarkdown saved to: {markdown_file}")
                    
            else:
                print(f"✗ Markdown generation failed: {result['error']}")
            
            # Save single result if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nResult saved to {args.output}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()