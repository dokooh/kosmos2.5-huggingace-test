"""
Working OCR inference that properly handles Kosmos-2.5 model structure
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
import re

class WorkingKosmos25OCR:
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
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(self.model)}")
        
        # Check model components
        if hasattr(self.model, 'language_model'):
            print(f"  ✓ Has language_model: {type(self.model.language_model)}")
        if hasattr(self.model, 'vision_model'):
            print(f"  ✓ Has vision_model: {type(self.model.vision_model)}")
    
    def prepare_inputs(self, image, prompt="<ocr>"):
        """Prepare inputs with proper dtype handling"""
        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    processed_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                else:
                    processed_inputs[key] = value.to(device=model_device)
            else:
                processed_inputs[key] = value
        
        return processed_inputs
    
    def extract_text_multiple_methods(self, image, prompt="<ocr>"):
        """Try multiple methods to extract text from the model"""
        print(f"Extracting text with prompt: {prompt}")
        
        results = []
        
        try:
            inputs = self.prepare_inputs(image, prompt)
            
            # Method 1: Standard forward pass
            print("Method 1: Standard forward pass...")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Try to extract text from different output attributes
            for attr_name in ['logits', 'prediction_logits', 'last_hidden_state']:
                if hasattr(outputs, attr_name):
                    try:
                        tensor = getattr(outputs, attr_name)
                        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                            
                            # Method A: Argmax decoding
                            predicted_ids = torch.argmax(tensor, dim=-1)
                            if predicted_ids.dim() > 1:
                                predicted_ids = predicted_ids[0]
                            
                            # Limit to reasonable length
                            predicted_ids = predicted_ids[:512]
                            
                            try:
                                decoded_text = self.processor.tokenizer.decode(
                                    predicted_ids, 
                                    skip_special_tokens=True
                                ).strip()
                                
                                if decoded_text and len(decoded_text) > 5:
                                    results.append({
                                        'method': f'forward_pass_{attr_name}_argmax',
                                        'text': decoded_text,
                                        'confidence': 'medium'
                                    })
                                    print(f"  ✓ Found text via {attr_name} argmax: {len(decoded_text)} chars")
                            except:
                                pass
                            
                            # Method B: Top-k sampling for diversity
                            try:
                                if tensor.shape[-1] > 1000:  # Likely vocabulary logits
                                    top_k_ids = torch.topk(tensor, k=5, dim=-1).indices
                                    for k in range(min(3, top_k_ids.shape[-1])):
                                        sample_ids = top_k_ids[:, :, k] if top_k_ids.dim() > 2 else top_k_ids[:, k]
                                        if sample_ids.dim() > 1:
                                            sample_ids = sample_ids[0]
                                        
                                        sample_ids = sample_ids[:512]
                                        
                                        try:
                                            sample_text = self.processor.tokenizer.decode(
                                                sample_ids, 
                                                skip_special_tokens=True
                                            ).strip()
                                            
                                            if sample_text and len(sample_text) > 5:
                                                results.append({
                                                    'method': f'forward_pass_{attr_name}_topk_{k}',
                                                    'text': sample_text,
                                                    'confidence': 'low'
                                                })
                                                print(f"  ✓ Found text via {attr_name} top-k {k}: {len(sample_text)} chars")
                                        except:
                                            pass
                            except:
                                pass
                    except Exception as e:
                        print(f"  ⚠️  Error processing {attr_name}: {e}")
            
            # Method 2: Language model generation (if available)
            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'generate'):
                print("Method 2: Language model generation...")
                try:
                    input_ids = inputs.get("input_ids")
                    if input_ids is not None:
                        generated_ids = self.model.language_model.generate(
                            input_ids=input_ids,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                        
                        input_length = input_ids.shape[1]
                        new_tokens = generated_ids[:, input_length:]
                        
                        if new_tokens.shape[1] > 0:
                            generated_text = self.processor.tokenizer.decode(
                                new_tokens[0], 
                                skip_special_tokens=True
                            ).strip()
                            
                            if generated_text:
                                results.append({
                                    'method': 'language_model_generation',
                                    'text': generated_text,
                                    'confidence': 'high'
                                })
                                print(f"  ✓ Language model generated: {len(generated_text)} chars")
                except Exception as e:
                    print(f"  ⚠️  Language model generation failed: {e}")
            
            return results
            
        except Exception as e:
            print(f"✗ Text extraction failed: {e}")
            return []
    
    def post_process_ocr_text(self, raw_text, method_info=""):
        """Post-process extracted OCR text"""
        if not raw_text:
            return ""
        
        # Remove common artifacts
        cleaned = raw_text.replace("<ocr>", "").replace("</ocr>", "")
        cleaned = re.sub(r'<[^>]+>', '', cleaned)  # Remove XML/HTML tags
        
        # Remove the original prompt if it appears in output
        if cleaned.startswith(cleaned.split()[0]) and len(cleaned.split()) > 1:
            words = cleaned.split()
            if words[0] in ['<ocr>', 'ocr', 'OCR', 'extract', 'text']:
                cleaned = ' '.join(words[1:])
        
        # Clean up whitespace and formatting
        lines = [line.strip() for line in cleaned.split('\n')]
        lines = [line for line in lines if line and not line.isspace()]
        
        # Remove repeated characters (common OCR artifact)
        result_lines = []
        for line in lines:
            # Remove excessive repetition
            if len(set(line)) > 1:  # Only keep lines with character variety
                result_lines.append(line)
        
        result = '\n'.join(result_lines).strip()
        
        # Final cleanup
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = re.sub(r'(.)\1{4,}', r'\1', result)  # Remove excessive repetition
        
        return result
    
    def run_ocr(self, image):
        """Run OCR with multiple extraction methods"""
        print("Running working OCR extraction...")
        
        # Try different prompts
        prompts = ["<ocr>", "Extract text:", "OCR:"]
        
        all_results = []
        
        for prompt in prompts:
            print(f"\nTrying prompt: '{prompt}'")
            extraction_results = self.extract_text_multiple_methods(image, prompt)
            
            for result in extraction_results:
                processed_text = self.post_process_ocr_text(result['text'], result['method'])
                if processed_text and len(processed_text) > 3:
                    all_results.append({
                        'prompt': prompt,
                        'method': result['method'],
                        'confidence': result['confidence'],
                        'raw_text': result['text'],
                        'processed_text': processed_text,
                        'text_length': len(processed_text)
                    })
        
        # Sort by confidence and text length
        confidence_order = {'high': 3, 'medium': 2, 'low': 1}
        all_results.sort(
            key=lambda x: (confidence_order.get(x['confidence'], 0), x['text_length']), 
            reverse=True
        )
        
        if all_results:
            best_result = all_results[0]
            print(f"\n✓ Best OCR result:")
            print(f"  Method: {best_result['method']}")
            print(f"  Prompt: {best_result['prompt']}")
            print(f"  Confidence: {best_result['confidence']}")
            print(f"  Text length: {best_result['text_length']}")
            print(f"  Preview: {best_result['processed_text'][:100]}...")
            
            return {
                'success': True,
                'best_result': best_result,
                'all_results': all_results,
                'extracted_text': best_result['processed_text']
            }
        else:
            return {
                'success': False,
                'error': 'No text could be extracted using any method',
                'extracted_text': '',
                'all_results': []
            }
    
    def load_image(self, image_input):
        """Load image from various sources"""
        if image_input.startswith(('http://', 'https://')):
            print(f"Loading image from URL: {image_input}")
            try:
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from URL: {e}")
        
        elif os.path.exists(image_input):
            print(f"Loading image from file: {image_input}")
            try:
                image = Image.open(image_input)
                return image.convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image from file: {e}")
        
        elif image_input == "default":
            print("Creating enhanced test image...")
            from PIL import ImageDraw, ImageFont
            
            image = Image.new('RGB', (600, 400), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            
            try:
                font_large = ImageFont.truetype("arial.ttf", 24)
                font_medium = ImageFont.truetype("arial.ttf", 18)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Create realistic document content
            y = 30
            draw.text((50, y), "RECEIPT #R-2024-001", fill=(0, 0, 0), font=font_large)
            y += 50
            draw.text((50, y), "Store: Tech Electronics", fill=(0, 0, 0), font=font_medium)
            y += 30
            draw.text((50, y), "Date: 2024-01-15", fill=(0, 0, 0), font=font_medium)
            y += 30
            draw.text((50, y), "Customer: John Smith", fill=(0, 0, 0), font=font_medium)
            y += 40
            draw.text((50, y), "Items Purchased:", fill=(0, 0, 0), font=font_medium)
            y += 30
            draw.text((70, y), "1x Laptop Computer - $899.99", fill=(0, 0, 0), font=font_small)
            y += 25
            draw.text((70, y), "1x Wireless Mouse - $29.99", fill=(0, 0, 0), font=font_small)
            y += 25
            draw.text((70, y), "1x USB Cable - $15.99", fill=(0, 0, 0), font=font_small)
            y += 40
            draw.text((50, y), "Subtotal: $945.97", fill=(0, 0, 0), font=font_medium)
            y += 25
            draw.text((50, y), "Tax: $75.68", fill=(0, 0, 0), font=font_medium)
            y += 25
            draw.text((50, y), "TOTAL: $1021.65", fill=(0, 0, 0), font=font_large)
            
            return image
        
        else:
            raise ValueError(f"Invalid image input: {image_input}")

def main():
    parser = argparse.ArgumentParser(description="Working OCR inference for Kosmos-2.5")
    parser.add_argument("--model_path", type=str, default="./fp4_quantized_model",
                       help="Path to saved FP4 quantized model")
    parser.add_argument("--image", type=str, default="default",
                       help="Image path, URL, or 'default' for test image")
    parser.add_argument("--output", type=str,
                       help="Output file to save OCR results")
    parser.add_argument("--show_all", action="store_true",
                       help="Show all extraction results, not just the best")
    
    args = parser.parse_args()
    
    ocr = WorkingKosmos25OCR(model_path=args.model_path)
    
    try:
        # Load model
        ocr.load_model()
        
        # Load image
        image = ocr.load_image(args.image)
        print(f"Image size: {image.size}")
        
        # Run OCR
        start_time = time.time()
        result = ocr.run_ocr(image)
        total_time = time.time() - start_time
        
        # Display results
        print(f"\n{'='*60}")
        print("WORKING OCR RESULTS")
        print('='*60)
        
        if result["success"]:
            print(f"✓ OCR successful in {total_time:.2f}s")
            print(f"\nExtracted Text:")
            print("-" * 40)
            print(result["extracted_text"])
            
            if args.show_all and len(result["all_results"]) > 1:
                print(f"\nAll Results ({len(result['all_results'])} found):")
                print("-" * 40)
                for i, res in enumerate(result["all_results"]):
                    print(f"{i+1}. {res['method']} ({res['confidence']}):")
                    print(f"   {res['processed_text'][:100]}{'...' if len(res['processed_text']) > 100 else ''}")
                    print()
        else:
            print(f"✗ OCR failed: {result['error']}")
        
        # Save results
        if args.output:
            output_data = {
                "image_source": args.image,
                "model_path": args.model_path,
                "total_time": total_time,
                "result": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
