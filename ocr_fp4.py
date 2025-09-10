"""
OCR inference script using saved FP4 quantized Kosmos-2.5 model
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

class Kosmos25OCRInference:
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
            # Create default test image
            print("Using default test image")
            return Image.new('RGB', (300, 400), color=(255, 255, 255))
        
        else:
            raise ValueError(f"Invalid image input: {image_input}")
    
    def run_ocr(self, image, extract_text_only=True):
        """Run OCR inference on the image"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            print("Running OCR inference...")
            start_time = time.time()
            
            # Prepare inputs
            model_dtype = next(self.model.parameters()).dtype
            inputs = self.processor(text="<ocr>", images=image, return_tensors="pt")
            
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
            print(f"✓ OCR completed in {inference_time:.2f}s")
            
            # Process outputs
            result = self.process_ocr_output(outputs, extract_text_only)
            
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
    
    def process_ocr_output(self, outputs, extract_text_only=True):
        """Process the model output to extract OCR results"""
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
            
            if extract_text_only:
                # Try to extract text from common attributes
                text_candidates = []
                
                # Check for logits or similar attributes that might contain text
                for attr_info in output_info["attributes"]:
                    attr_name = attr_info["name"]
                    if "logits" in attr_name.lower() or "output" in attr_name.lower():
                        try:
                            tensor = getattr(outputs, attr_name)
                            if tensor.dim() >= 2:
                                # Try to decode as text
                                predicted_ids = torch.argmax(tensor, dim=-1)
                                if predicted_ids.dim() > 1:
                                    sample_ids = predicted_ids[0][:100]  # First 100 tokens
                                else:
                                    sample_ids = predicted_ids[:100]
                                
                                try:
                                    decoded = self.processor.tokenizer.decode(sample_ids, skip_special_tokens=True)
                                    if decoded.strip():
                                        text_candidates.append({
                                            "source": attr_name,
                                            "text": decoded.strip()
                                        })
                                except:
                                    pass
                        except:
                            pass
                
                if text_candidates:
                    return {
                        "extracted_text": text_candidates[0]["text"],
                        "all_candidates": text_candidates,
                        "output_structure": output_info
                    }
                else:
                    return {
                        "extracted_text": "No text could be extracted",
                        "all_candidates": [],
                        "output_structure": output_info
                    }
            else:
                return output_info
                
        except Exception as e:
            return {"error": f"Failed to process output: {e}"}
    
    def batch_ocr(self, image_paths, output_file=None):
        """Run OCR on multiple images"""
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                image = self.load_image(image_path)
                result = self.run_ocr(image)
                result["image_path"] = image_path
                results.append(result)
                
                if result["success"]:
                    print(f"✓ OCR successful")
                    if "extracted_text" in result["result"]:
                        preview = result["result"]["extracted_text"][:100]
                        print(f"  Text preview: {preview}...")
                else:
                    print(f"✗ OCR failed: {result['error']}")
                    
            except Exception as e:
                print(f"✗ Failed to process {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "success": False,
                    "error": str(e),
                    "result": None
                })
        
        # Save results if output file specified
        if output_file:
            print(f"\nSaving results to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="OCR inference using FP4 quantized Kosmos-2.5")
    parser.add_argument("--model_path", type=str, default="./fp4_quantized_model",
                       help="Path to saved FP4 quantized model")
    parser.add_argument("--image", type=str, default="default",
                       help="Image path, URL, or 'default' for test image")
    parser.add_argument("--batch", type=str, nargs='+',
                       help="Multiple image paths for batch processing")
    parser.add_argument("--output", type=str,
                       help="Output file to save results (JSON format)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output structure")
    
    args = parser.parse_args()
    
    # Initialize OCR inference
    ocr = Kosmos25OCRInference(model_path=args.model_path)
    
    try:
        # Load model
        ocr.load_model()
        
        if args.batch:
            # Batch processing
            results = ocr.batch_ocr(args.batch, args.output)
            
            # Print summary
            successful = sum(1 for r in results if r["success"])
            print(f"\n{'='*50}")
            print("BATCH OCR SUMMARY")
            print('='*50)
            print(f"Total images: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            
        else:
            # Single image processing
            image = ocr.load_image(args.image)
            result = ocr.run_ocr(image, extract_text_only=not args.verbose)
            
            print(f"\n{'='*50}")
            print("OCR RESULTS")
            print('='*50)
            
            if result["success"]:
                print(f"✓ OCR successful (inference time: {result['inference_time']:.2f}s)")
                
                if "extracted_text" in result["result"]:
                    print(f"\nExtracted Text:")
                    print("-" * 30)
                    print(result["result"]["extracted_text"])
                    
                    if args.verbose and "all_candidates" in result["result"]:
                        print(f"\nAll text candidates:")
                        for candidate in result["result"]["all_candidates"]:
                            print(f"  {candidate['source']}: {candidate['text'][:100]}...")
                
                if args.verbose:
                    print(f"\nOutput structure:")
                    output_structure = result["result"].get("output_structure", {})
                    print(f"  Type: {output_structure.get('type', 'Unknown')}")
                    print(f"  Attributes: {len(output_structure.get('attributes', []))}")
                    for attr in output_structure.get('attributes', []):
                        print(f"    {attr['name']}: {attr['shape']} ({attr['dtype']})")
            else:
                print(f"✗ OCR failed: {result['error']}")
            
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
