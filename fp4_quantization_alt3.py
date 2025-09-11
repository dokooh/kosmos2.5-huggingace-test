"""
HuggingFace FP4 Quantization Suite for Kosmos-2.5
Uses only BitsAndBytesConfig supported quantization methods
Focuses on generation methods that actually work
"""

import torch
import os
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

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

class HuggingFaceQuantizer:
    """Quantizer using only HuggingFace BitsAndBytesConfig methods"""
    
    def __init__(self, model_id="microsoft/kosmos-2.5", output_dir="./quantized_models"):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def cleanup_memory(self):
        """Clean up GPU memory between quantization attempts"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def test_generation_capability(self, model, processor) -> bool:
        """Test if quantized model can actually generate text (NO forward pass logits)"""
        try:
            # Create minimal test image
            test_image = Image.new('RGB', (100, 50), color=(255, 255, 255))
            
            # Check available generation methods
            has_language_model = hasattr(model, 'language_model')
            has_lm_generate = has_language_model and hasattr(model.language_model, 'generate')
            has_direct_generate = hasattr(model, 'generate')
            
            if not (has_lm_generate or has_direct_generate):
                print(f"        ❌ No generation methods available")
                return False
            
            # Prepare inputs with proper dtype conversion
            inputs = processor(text="<ocr>", images=test_image, return_tensors="pt")
            
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            # Convert inputs to model's device/dtype
            input_ids = inputs["input_ids"].to(device=model_device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    dtype=model_dtype, device=model_device
                )
            
            # Test generation (prefer language_model.generate)
            with torch.no_grad():
                if has_lm_generate:
                    print(f"        🔍 Testing language_model.generate()...")
                    generated = model.language_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=3,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                elif has_direct_generate:
                    print(f"        🔍 Testing model.generate()...")
                    generated = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=3,
                        do_sample=False,
                        **{k: v for k, v in inputs.items() if k != "input_ids"}
                    )
                else:
                    return False
                
                # Check if new tokens were generated
                new_tokens_generated = generated.shape[1] > input_ids.shape[1]
                print(f"        {'✅' if new_tokens_generated else '❌'} Generation test result")
                return new_tokens_generated
        
        except Exception as e:
            print(f"        ❌ Generation test failed: {e}")
            return False
    
    def quantize_nf4_double_quant(self) -> QuantizationResult:
        """NF4 with double quantization - Best compression/quality balance"""
        print("\n🔧 Quantizing with NF4 + Double Quantization...")
        
        try:
            start_time = time.time()
            
            # HuggingFace BitsAndBytesConfig for NF4 + Double Quantization
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
            
            print(f"    ✅ Model loaded: {type(model).__name__}")
            print(f"    📊 Model dtype: {next(model.parameters()).dtype}")
            
            # Test generation capability
            generation_works = self.test_generation_capability(model, processor)
            
            # Calculate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            # Save quantized model
            output_path = self.output_dir / "NF4_DoubleQuant"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            # Save metadata
            metadata = {
                "quantization_method": "NF4_DoubleQuant",
                "huggingface_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "torch.float16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                },
                "model_size_mb": model_size_mb,
                "generation_test_passed": generation_works,
                "model_dtype": str(next(model.parameters()).dtype),
                "has_language_model": hasattr(model, 'language_model'),
                "has_generate_method": hasattr(model, 'generate')
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
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Generation test: {'✅ PASSED' if generation_works else '❌ FAILED'}")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ NF4 quantization failed: {e}")
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
    
    def quantize_fp4_bfloat16(self) -> QuantizationResult:
        """FP4 with BFloat16 - Optimized for newer GPUs"""
        print("\n🔧 Quantizing with FP4 + BFloat16...")
        
        try:
            start_time = time.time()
            
            # FP4 + BFloat16 configuration
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
            
            print(f"    ✅ Model loaded with BF16 dtype")
            
            generation_works = self.test_generation_capability(model, processor)
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            output_path = self.output_dir / "FP4_BFloat16"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "FP4_BFloat16",
                "huggingface_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "torch.bfloat16",
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
                method="FP4_BFloat16",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=4.0,
                generation_test_passed=generation_works,
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Generation test: {'✅ PASSED' if generation_works else '❌ FAILED'}")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ FP4 BFloat16 failed: {e}")
            self.cleanup_memory()
            return QuantizationResult(
                method="FP4_BFloat16",
                success=False,
                model_size_mb=0,
                quantization_time=0,
                compression_ratio=0,
                generation_test_passed=False,
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_fp4_standard(self) -> QuantizationResult:
        """Standard FP4 quantization - Basic 4-bit"""
        print("\n🔧 Quantizing with Standard FP4...")
        
        try:
            start_time = time.time()
            
            # Standard FP4 configuration (no double quantization)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
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
            
            print(f"    ✅ Standard FP4 model loaded")
            
            generation_works = self.test_generation_capability(model, processor)
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            output_path = self.output_dir / "FP4_Standard"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "FP4_Standard",
                "huggingface_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "torch.float16",
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
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Generation test: {'✅ PASSED' if generation_works else '❌ FAILED'}")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ Standard FP4 failed: {e}")
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
        """INT8 quantization - Maximum compatibility fallback"""
        print("\n🔧 Quantizing with INT8 (Fallback)...")
        
        try:
            start_time = time.time()
            
            # INT8 quantization (more compatible)
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
            
            print(f"    ✅ INT8 model loaded")
            
            generation_works = self.test_generation_capability(model, processor)
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            output_path = self.output_dir / "INT8_Fallback"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "INT8_Fallback",
                "huggingface_config": {
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
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Generation test: {'✅ PASSED' if generation_works else '❌ FAILED'}")
            
            del model, processor
            self.cleanup_memory()
            
            return result
            
        except Exception as e:
            print(f"    ❌ INT8 quantization failed: {e}")
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
    
    def run_complete_quantization_suite(self) -> List[QuantizationResult]:
        """Run all HuggingFace quantization methods"""
        print("🚀 KOSMOS-2.5 FP4 QUANTIZATION SUITE")
        print("=" * 70)
        print("📋 Using only HuggingFace BitsAndBytesConfig methods")
        print("⚠️  Testing generation methods only (NO forward pass logits)")
        print("=" * 70)
        
        # Define quantization methods to test
        quantization_methods = [
            self.quantize_nf4_double_quant,
            self.quantize_fp4_bfloat16,
            self.quantize_fp4_standard,
            self.quantize_int8_fallback
        ]
        
        results = []
        
        for i, method in enumerate(quantization_methods, 1):
            try:
                print(f"\n[{i}/{len(quantization_methods)}] Running {method.__name__}...")
                result = method()
                results.append(result)
                
                # Brief pause between methods
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Method {method.__name__} completely failed: {e}")
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
        
        # Generate comprehensive report
        self.generate_quantization_report(results)
        
        return results
    
    def generate_quantization_report(self, results: List[QuantizationResult]):
        """Generate detailed quantization report"""
        successful = [r for r in results if r.success]
        working_generation = [r for r in successful if r.generation_test_passed]
        failed = [r for r in results if not r.success]
        quantized_no_gen = [r for r in successful if not r.generation_test_passed]
        
        print(f"\n{'='*70}")
        print("QUANTIZATION SUITE RESULTS")
        print('='*70)
        
        print(f"\n📊 SUMMARY:")
        print(f"  ✅ Successfully quantized: {len(successful)}/{len(results)}")
        print(f"  🎯 Working text generation: {len(working_generation)}/{len(results)}")
        print(f"  ⚠️  Quantized but no generation: {len(quantized_no_gen)}")
        print(f"  ❌ Complete failures: {len(failed)}")
        
        if working_generation:
            print(f"\n🎯 WORKING METHODS (Ready for OCR/Markdown):")
            print(f"{'Method':<20} {'Size (MB)':<12} {'Time (s)':<10} {'Dtype':<15}")
            print("-" * 70)
            
            for r in working_generation:
                print(f"{r.method:<20} {r.model_size_mb:<12.1f} "
                      f"{r.quantization_time:<10.1f} {r.model_dtype:<15}")
            
            # Find best options
            best_compression = min(working_generation, key=lambda x: x.model_size_mb)
            fastest_setup = min(working_generation, key=lambda x: x.quantization_time)
            
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"  🥇 Best Compression: {best_compression.method}")
            print(f"     Size: {best_compression.model_size_mb:.1f}MB")
            print(f"     Usage: python working_ocr_inference.py --model_path ./quantized_models/{best_compression.method}")
            
            print(f"\n  ⚡ Fastest Setup: {fastest_setup.method}")
            print(f"     Time: {fastest_setup.quantization_time:.1f}s")
            print(f"     Usage: python working_ocr_inference.py --model_path ./quantized_models/{fastest_setup.method}")
        
        if quantized_no_gen:
            print(f"\n⚠️  QUANTIZED BUT GENERATION ISSUES:")
            for r in quantized_no_gen:
                print(f"  {r.method}: Model quantized but generation test failed")
        
        if failed:
            print(f"\n❌ FAILED METHODS:")
            for r in failed:
                print(f"  {r.method}: {r.error_message}")
        
        # Save comprehensive JSON report
        report_data = {
            "suite_name": "HuggingFace FP4 Quantization Suite",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": self.model_id,
            "device": self.device,
            "pytorch_version": torch.__version__,
            "summary": {
                "total_methods": len(results),
                "successful_quantizations": len(successful),
                "working_generation_methods": len(working_generation),
                "failed_methods": len(failed)
            },
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
        
        report_path = self.output_dir / "quantization_suite_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved: {report_path}")
        
        # Final usage instructions
        if working_generation:
            print(f"\n🔧 NEXT STEPS:")
            print(f"Test your quantized models with:")
            for r in working_generation:
                model_path = f"./quantized_models/{r.method}"
                print(f"  python working_ocr_inference.py --model_path {model_path}")
        else:
            print(f"\n⚠️  NO WORKING QUANTIZED MODELS")
            print(f"All quantization attempts had generation issues.")
            print(f"Consider:")
            print(f"  - Using the original (non-quantized) model")
            print(f"  - Checking your GPU compatibility with 4-bit quantization")
            print(f"  - Trying different quantization libraries")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="HuggingFace FP4 Quantization Suite for Kosmos-2.5")
    parser.add_argument("--model_id", default="microsoft/kosmos-2.5",
                       help="Model ID to quantize (default: microsoft/kosmos-2.5)")
    parser.add_argument("--output_dir", default="./quantized_models",
                       help="Output directory for quantized models")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting HuggingFace FP4 Quantization Suite")
    print(f"📦 Model: {args.model_id}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔢 PyTorch: {torch.__version__}")
    
    # Check GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU Memory: {gpu_memory:.1f}GB")
    
    quantizer = HuggingFaceQuantizer(
        model_id=args.model_id,
        output_dir=args.output_dir
    )
    
    try:
        results = quantizer.run_complete_quantization_suite()
        
        working_methods = [r for r in results if r.success and r.generation_test_passed]
        
        if working_methods:
            print(f"\n🎉 QUANTIZATION COMPLETED SUCCESSFULLY!")
            print(f"✅ {len(working_methods)} working quantized models ready")
            print(f"📁 Models saved in: {args.output_dir}")
        else:
            print(f"\n❌ QUANTIZATION SUITE COMPLETED WITH ISSUES")
            print(f"No quantized models passed generation testing")
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
