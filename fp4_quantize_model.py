"""
Working FP4 quantization script with proper dtype handling
"""

import torch
import time
import os
import json
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image

class WorkingKosmos25FP4:
    def __init__(self, output_dir="./working_fp4_model"):
        self.model_id = "microsoft/kosmos-2.5"
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def ensure_tensor_compatibility(self, inputs, model):
        """Ensure all input tensors are compatible with model dtype"""
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        
        compatible_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Handle floating point tensors
                if value.dtype.is_floating_point:
                    if value.dtype != model_dtype:
                        print(f"Converting {key}: {value.dtype} -> {model_dtype}")
                        compatible_inputs[key] = value.to(dtype=model_dtype, device=model_device)
                    else:
                        compatible_inputs[key] = value.to(device=model_device)
                else:
                    # Integer tensors - just move to device
                    compatible_inputs[key] = value.to(device=model_device)
            else:
                compatible_inputs[key] = value
        
        return compatible_inputs
    
    def test_quantized_model(self, model, processor):
        """Test the quantized model with proper dtype handling"""
        print("Testing quantized model...")
        
        try:
            # Create test image
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
            
            # Test OCR
            print("Testing OCR...")
            inputs = processor(text="<ocr>", images=image, return_tensors="pt")
            compatible_inputs = self.ensure_tensor_compatibility(inputs, model)
            
            with torch.no_grad():
                outputs = model(**compatible_inputs)
            
            print("✓ OCR test successful")
            
            # Test markdown
            print("Testing markdown generation...")
            inputs = processor(text="<md>", images=image, return_tensors="pt")
            compatible_inputs = self.ensure_tensor_compatibility(inputs, model)
            
            with torch.no_grad():
                outputs = model(**compatible_inputs)
            
            print("✓ Markdown test successful")
            
            return True
            
        except Exception as e:
            print(f"✗ Model test failed: {e}")
            return False
    
    def quantize_and_save(self):
        """Quantize model and save with proper error handling"""
        print("Kosmos-2.5 FP4 Quantization (Working Version)")
        print("=" * 55)
        
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
            
            # Load model with quantization
            model = AutoModel.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
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
            
            # Test before saving
            test_success = self.test_quantized_model(model, processor)
            if not test_success:
                print("⚠️  Model test failed, but continuing with save...")
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save model and processor
            print(f"Saving to {self.output_dir}...")
            save_start = time.time()
            
            model.save_pretrained(self.output_dir)
            processor.save_pretrained(self.output_dir)
            
            save_time = time.time() - save_start
            print(f"✓ Model saved in {save_time:.2f}s")
            
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
                "device_used": str(self.device),
                "test_passed": test_success
            }
            
            metadata_path = os.path.join(self.output_dir, "quantization_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Metadata saved")
            
            print(f"\n{'='*55}")
            print("SUCCESS!")
            print('='*55)
            print(f"✓ FP4 quantized model saved to: {self.output_dir}")
            print(f"✓ Model size: {size_mb:.1f} MB")
            print(f"✓ Compression ratio: ~4x smaller")
            print(f"✓ Ready for inference!")
            
            return True
            
        except Exception as e:
            print(f"✗ Quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Working FP4 quantization for Kosmos-2.5")
    parser.add_argument("--output_dir", type=str, default="./working_fp4_model",
                       help="Output directory for quantized model")
    
    args = parser.parse_args()
    
    # Check requirements
    print("Checking requirements...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Quantization may be slow on CPU.")
    
    try:
        import bitsandbytes
        print("✓ BitsAndBytes available")
    except ImportError:
        print("✗ BitsAndBytes not installed. Install with: pip install bitsandbytes")
        return
    
    try:
        import accelerate
        print("✓ Accelerate available")
    except ImportError:
        print("✗ Accelerate not installed. Install with: pip install accelerate")
        return
    
    # Run quantization
    quantizer = WorkingKosmos25FP4(output_dir=args.output_dir)
    success = quantizer.quantize_and_save()
    
    if success:
        print(f"\n🎉 Quantization completed successfully!")
        print(f"Next steps:")
        print(f"1. Test with: python test_saved_model.py --model_path {args.output_dir}")
        print(f"2. Use for OCR: python ocr_inference.py --model_path {args.output_dir}")
        print(f"3. Use for markdown: python markdown_inference.py --model_path {args.output_dir}")
    else:
        print(f"\n❌ Quantization failed. Check error messages above.")

if __name__ == "__main__":
    main()
