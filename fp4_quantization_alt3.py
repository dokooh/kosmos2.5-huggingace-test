"""
FP4 Quantization Suite for Kosmos-2.5
Uses only HuggingFace BitsAndBytesConfig supported methods
Skips forward pass logits - focuses on working generation methods
"""

import torch
import os
import json
import time
import gc
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
import argparse

from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image

@dataclass
class QuantizationResult:
    method: str
    success: bool
    model_size_mb: float
    quantization_time: float
    compression_ratio: float
    generation_test_passed: bool
    model_dtype: str
    error_message: str = ""

class HuggingFaceFP4Quantizer:
    """FP4 quantization using only HuggingFace BitsAndBytesConfig"""
    
    def __init__(self, model_id="microsoft/kosmos-2.5", output_dir="./quantized_models"):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def cleanup_memory(self):
        """Clean up GPU memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def test_generation_capability(self, model, processor) -> bool:
        """Test if quantized model can actually generate text"""
        try:
            # Create minimal test
            test_image = Image.new('RGB', (200, 100), color=(255, 255, 255))
            
            # Check available generation methods
            has_language_model_gen = (hasattr(model, 'language_model') and 
                                     hasattr(model.language_model, 'generate'))
            has_direct_gen = hasattr(model, 'generate')
            
            if not (has_language_model_gen or has_direct_gen):
                return False
            
            # Prepare inputs
            inputs = processor(text="<ocr>", images=test_image, return_tensors="pt")
            
            # Convert to model's dtype and device
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            input_ids = inputs["input_ids"].to(device=model_device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    dtype=model_dtype, device=model_device
                )
            
            # Test generation (prefer language_model)
            with torch.no_grad():
                if has_language_model_gen:
                    generated = model.language_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                else:
                    generated = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=5,
                        do_sample=False,
                        **{k: v for k, v in inputs.items() if k != "input_ids"}
                    )
                
                # Check if new tokens were generated
                return generated.shape[1] > input_ids.shape[1]
        
        except Exception as e:
            print(f"    Generation test error: {e}")
            return False
    
    def quantize_nf4_double_quant(self) -> QuantizationResult:
        """NF4 with double quantization - Best balance"""
        print("\n🔧 NF4 + Double Quantization")
        
        try:
            start_time = time.time()
            
            # HuggingFace BitsAndBytesConfig for NF4
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load quantized model
            model = AutoModel.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Test generation capability
            generation_works = self.test_generation_capability(model, processor)
            print(f"    Generation test: {'✅' if generation_works else '❌'}")
            
            # Calculate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            # Save model
            output_path = self.output_dir / "NF4_DoubleQuant"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            # Save metadata
            metadata = {
                "quantization_method": "NF4_DoubleQuant",
                "config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                },
                "model_size_mb": model_size_mb,
                "generation_test_passed": generation_works,
                "has_language_model": hasattr(model, 'language_model'),
                "has_generate": hasattr(model, 'generate'),
                "model_dtype": str(next(model.parameters()).dtype)
            }
            
            with open(output_path / "quantization_info.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                method="NF4_DoubleQuant",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=4.0,  # Approximate for 4-bit
                generation_test_passed=generation_works,
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB in {quantization_time:.1f}s")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.cleanup_memory()
            return QuantizationResult(
                method="NF4_DoubleQuant",
                success=False,
                model_size_mb=0,
                quantization_time=0,
                compression_ratio=0,
                generation_test_passed=False,
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_fp4_bf16(self) -> QuantizationResult:
        """FP4 with BFloat16 - For newer GPUs"""
        print("\n🔧 FP4 + BFloat16")
        
        try:
            start_time = time.time()
            
            # HuggingFace BitsAndBytesConfig for FP4 + BF16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4"
            )
            
            model = AutoModel.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Test generation
            generation_works = self.test_generation_capability(model, processor)
            print(f"    Generation test: {'✅' if generation_works else '❌'}")
            
            # Calculate size and save
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            output_path = self.output_dir / "FP4_BF16"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "FP4_BF16",
                "config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "bfloat16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "fp4"
                },
                "model_size_mb": model_size_mb,
                "generation_test_passed": generation_works,
                "model_dtype": str(next(model.parameters()).dtype)
            }
            
            with open(output_path / "quantization_info.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                method="FP4_BF16",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=4.0,
                generation_test_passed=generation_works,
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB in {quantization_time:.1f}s")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.cleanup_memory()
            return QuantizationResult(
                method="FP4_BF16",
                success=False,
                model_size_mb=0,
                quantization_time=0,
                compression_ratio=0,
                generation_test_passed=False,
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_fp4_standard(self) -> QuantizationResult:
        """Standard FP4 quantization"""
        print("\n🔧 Standard FP4")
        
        try:
            start_time = time.time()
            
            # Standard FP4 configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,  # No double quantization
                bnb_4bit_quant_type="fp4"
            )
            
            model = AutoModel.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            generation_works = self.test_generation_capability(model, processor)
            print(f"    Generation test: {'✅' if generation_works else '❌'}")
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            output_path = self.output_dir / "FP4_Standard"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "FP4_Standard",
                "config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": False,
                    "bnb_4bit_quant_type": "fp4"
                },
                "model_size_mb": model_size_mb,
                "generation_test_passed": generation_works,
                "model_dtype": str(next(model.parameters()).dtype)
            }
            
            with open(output_path / "quantization_info.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                method="FP4_Standard",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=4.0,
                generation_test_passed=generation_works,
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB in {quantization_time:.1f}s")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.cleanup_memory()
            return QuantizationResult(
                method="FP4_Standard",
                success=False,
                model_size_mb=0,
                quantization_time=0,
                compression_ratio=0,
                generation_test_passed=False,
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_int8_fallback(self) -> QuantizationResult:
        """INT8 quantization fallback"""
        print("\n🔧 INT8 Fallback")
        
        try:
            start_time = time.time()
            
            # INT8 quantization
            model = AutoModel.from_pretrained(
                self.model_id,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            generation_works = self.test_generation_capability(model, processor)
            print(f"    Generation test: {'✅' if generation_works else '❌'}")
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            output_path = self.output_dir / "INT8_Fallback"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "INT8_Fallback",
                "config": {
                    "load_in_8bit": True
                },
                "model_size_mb": model_size_mb,
                "generation_test_passed": generation_works,
                "model_dtype": str(next(model.parameters()).dtype)
            }
            
            with open(output_path / "quantization_info.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                method="INT8_Fallback",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=2.0,  # Approximate for 8-bit
                generation_test_passed=generation_works,
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB in {quantization_time:.1f}s")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.cleanup_memory()
            return QuantizationResult(
                method="INT8_Fallback",
                success=False,
                model_size_mb=0,
                quantization_time=0,
                compression_ratio=0,
                generation_test_passed=False,
                model_dtype="",
                error_message=str(e)
            )
    
    def run_complete_quantization(self) -> List[QuantizationResult]:
        """Run all quantization methods"""
        print("🚀 KOSMOS-2.5 FP4 QUANTIZATION SUITE")
        print("=" * 60)
        print("📋 HuggingFace BitsAndBytesConfig Methods Only")
        print("⚠️  Generation-Only Testing (No Forward Pass Logits)")
        print("=" * 60)
        
        methods = [
            self.quantize_nf4_double_quant,
            self.quantize_fp4_bf16,
            self.quantize_fp4_standard,
            self.quantize_int8_fallback
        ]
        
        results = []
        
        for method in methods:
            try:
                result = method()
                results.append(result)
                time.sleep(1)  # Brief pause between methods
            except Exception as e:
                print(f"Method completely failed: {e}")
                method_name = method.__name__.replace('quantize_', '').upper()
                results.append(QuantizationResult(
                    method=method_name,
                    success=False,
                    model_size_mb=0,
                    quantization_time=0,
                    compression_ratio=0,
                    generation_test_passed=False,
                    model_dtype="",
                    error_message=str(e)
                ))
        
        # Generate report
        self.generate_report(results)
        
        return results
    
    def generate_report(self, results: List[QuantizationResult]):
        """Generate comprehensive quantization report"""
        successful = [r for r in results if r.success]
        working_gen = [r for r in successful if r.generation_test_passed]
        failed = [r for r in results if not r.success]
        
        print(f"\n{'='*60}")
        print("QUANTIZATION RESULTS SUMMARY")
        print('='*60)
        
        print(f"\n📊 Overall Results:")
        print(f"  ✅ Successful: {len(successful)}/{len(results)}")
        print(f"  🎯 Working Generation: {len(working_gen)}/{len(results)}")
        print(f"  ❌ Failed: {len(failed)}")
        
        if working_gen:
            print(f"\n🎯 WORKING METHODS:")
            print(f"{'Method':<20} {'Size (MB)':<12} {'Time (s)':<10} {'Dtype':<15}")
            print("-" * 65)
            
            for r in working_gen:
                print(f"{r.method:<20} {r.model_size_mb:<12.1f} "
                      f"{r.quantization_time:<10.1f} {r.model_dtype:<15}")
            
            # Recommendations
            best_size = min(working_gen, key=lambda x: x.model_size_mb)
            fastest = min(working_gen, key=lambda x: x.quantization_time)
            
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"  🎯 Best Compression: {best_size.method} ({best_size.model_size_mb:.1f}MB)")
            print(f"  ⚡ Fastest Setup: {fastest.method} ({fastest.quantization_time:.1f}s)")
        
        if failed:
            print(f"\n❌ FAILED METHODS:")
            for r in failed:
                print(f"  {r.method}: {r.error_message}")
        
        # Save detailed report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": self.model_id,
            "device": self.device,
            "pytorch_version": torch.__version__,
            "successful_methods": len(successful),
            "working_generation_methods": len(working_gen),
            "failed_methods": len(failed),
            "results": [
                {
                    "method": r.method,
                    "success": r.success,
                    "model_size_mb": r.model_size_mb,
                    "quantization_time": r.quantization_time,
                    "compression_ratio": r.compression_ratio,
                    "generation_test_passed": r.generation_test_passed,
                    "model_dtype": r.model_dtype,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }
        
        report_path = self.output_dir / "quantization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📋 Full report: {report_path}")
        
        if working_gen:
            print(f"\n🔧 Next Steps:")
            for r in working_gen:
                model_path = self.output_dir / r.method
                print(f"python working_ocr_inference.py --model_path {model_path}")
        else:
            print(f"\n⚠️  No working quantized models found.")
            print(f"Consider using the original model or check your environment.")

def main():
    parser = argparse.ArgumentParser(description="HuggingFace FP4 Quantization Suite for Kosmos-2.5")
    parser.add_argument("--model_id", default="microsoft/kosmos-2.5",
                       help="Model ID to quantize")
    parser.add_argument("--output_dir", default="./quantized_models",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Starting FP4 quantization...")
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    quantizer = HuggingFaceFP4Quantizer(
        model_id=args.model_id,
        output_dir=args.output_dir
    )
    
    results = quantizer.run_complete_quantization()
    
    working_methods = [r for r in results if r.success and r.generation_test_passed]
    
    if working_methods:
        print(f"\n🎉 Quantization completed!")
        print(f"✅ {len(working_methods)} working methods available")
    else:
        print(f"\n❌ No working quantization methods found")

if __name__ == "__main__":
    main()