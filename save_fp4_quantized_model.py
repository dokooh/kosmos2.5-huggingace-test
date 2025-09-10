"""
Script to quantize Kosmos-2.5 model with FP4 and save weights to disk
"""

import torch
import os
import json
import time
import argparse
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO

class Kosmos25FP4Saver:
    def __init__(self, output_dir="./fp4_quantized_model"):
        self.model_id = "microsoft/kosmos-2.5"
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Output directory created: {self.output_dir}")
    
    def quantize_and_save_model(self):
        """Quantize model with FP4 and save to disk"""
        print("Kosmos-2.5 FP4 Quantization and Saving")
        print("=" * 50)
        
        try:
            # Configure FP4 quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4"
            )
            
            print("Loading and quantizing model...")
            start_time = time.time()
            
            # Load model with FP4 quantization
            model = AutoModel.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            print(f"✓ Model loaded and quantized in {load_time:.2f}s")
            
            # Get model info
            model_dtype = next(model.parameters()).dtype
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            size_mb = (param_size + buffer_size) / 1024 / 1024
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"Model info:")
            print(f"  Dtype: {model_dtype}")
            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Parameters: {param_count:,}")
            
            # Create output directory
            self.create_output_directory()
            
            # Save model and processor
            print("Saving quantized model...")
            save_start = time.time()
            
            model.save_pretrained(self.output_dir)
            processor.save_pretrained(self.output_dir)
            
            save_time = time.time() - save_start
            print(f"✓ Model and processor saved in {save_time:.2f}s")
            
            # Save metadata
            metadata = {
                "model_id": self.model_id,
                "quantization_type": "fp4",
                "model_dtype": str(model_dtype),
                "size_mb": size_mb,
                "parameter_count": param_count,
                "quantization_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "fp4"
                },
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device_used": self.device
            }
            
            metadata_path = os.path.join(self.output_dir, "quantization_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Metadata saved to {metadata_path}")
            
            # Test the saved model
            print("\nTesting saved model...")
            self.test_saved_model()
            
            print(f"\n{'='*50}")
            print("SUCCESS!")
            print("="*50)
            print(f"✓ FP4 quantized model saved to: {self.output_dir}")
            print(f"✓ Model size: {size_mb:.1f} MB")
            print(f"✓ Ready for inference with OCR and markdown scripts")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during quantization and saving: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_saved_model(self):
        """Test the saved quantized model"""
        try:
            # Load saved model
            model = AutoModel.from_pretrained(
                self.output_dir,
                device_map="auto",
                trust_remote_code=True
            )
            
            processor = AutoProcessor.from_pretrained(
                self.output_dir,
                trust_remote_code=True
            )
            
            # Create test image
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
            
            # Test inference
            model_dtype = next(model.parameters()).dtype
            inputs = processor(text="<ocr>", images=image, return_tensors="pt")
            
            # Convert inputs to correct dtype
            processed_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype.is_floating_point:
                        processed_inputs[key] = value.to(dtype=model_dtype)
                    else:
                        processed_inputs[key] = value
                else:
                    processed_inputs[key] = value
            
            with torch.no_grad():
                outputs = model(**processed_inputs)
            
            print("✓ Saved model test successful")
            return True
            
        except Exception as e:
            print(f"✗ Saved model test failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Quantize Kosmos-2.5 with FP4 and save weights")
    parser.add_argument("--output_dir", type=str, default="./fp4_quantized_model",
                       help="Directory to save quantized model (default: ./fp4_quantized_model)")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test existing saved model without re-quantizing")
    
    args = parser.parse_args()
    
    saver = Kosmos25FP4Saver(output_dir=args.output_dir)
    
    if args.test_only:
        print("Testing existing saved model...")
        if os.path.exists(args.output_dir):
            success = saver.test_saved_model()
        else:
            print(f"✗ Model directory {args.output_dir} does not exist")
            success = False
    else:
        success = saver.quantize_and_save_model()
    
    if success:
        print(f"\n🎉 Operation completed successfully!")
        print(f"You can now use the OCR and markdown inference scripts with:")
        print(f"  python ocr_inference.py --model_path {args.output_dir}")
        print(f"  python markdown_inference.py --model_path {args.output_dir}")
    else:
        print(f"\n❌ Operation failed. Check error messages above.")

if __name__ == "__main__":
    main()
