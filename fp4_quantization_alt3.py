"""
Comprehensive FP4 Quantization Suite for Kosmos-2.5
Handles models without standard generate() methods
Tests multiple text extraction approaches for each quantized model
"""

import torch
import os
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import numpy as np

@dataclass
class QuantizationResult:
    method: str
    success: bool
    model_size_mb: float
    quantization_time: float
    compression_ratio: float
    has_generate: bool
    has_language_model: bool
    forward_pass_works: bool
    custom_generation_works: bool
    best_extraction_method: str
    model_dtype: str
    error_message: str = ""

class AdvancedKosmosQuantizer:
    """Advanced quantizer with multiple text extraction validation methods"""
    
    def __init__(self, model_id="microsoft/kosmos-2.5", output_dir="./quantized_models"):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def cleanup_memory(self):
        """Thorough memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def test_standard_generation(self, model, processor) -> Tuple[bool, str]:
        """Test if model has working standard generation methods"""
        try:
            test_image = Image.new('RGB', (100, 50), color=(255, 255, 255))
            
            # Check method availability
            has_lm_gen = hasattr(model, 'language_model') and hasattr(model.language_model, 'generate')
            has_direct_gen = hasattr(model, 'generate')
            
            if not (has_lm_gen or has_direct_gen):
                return False, "no_generate_methods"
            
            # Prepare inputs
            inputs = processor(text="<ocr>", images=test_image, return_tensors="pt")
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            input_ids = inputs["input_ids"].to(device=model_device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    dtype=model_dtype, device=model_device
                )
            
            # Test generation methods
            with torch.no_grad():
                if has_lm_gen:
                    generated = model.language_model.generate(
                        input_ids=input_ids,
                        max_new_tokens=3,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                    if generated.shape[1] > input_ids.shape[1]:
                        return True, "language_model_generate"
                
                if has_direct_gen:
                    generated = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=3,
                        do_sample=False,
                        **{k: v for k, v in inputs.items() if k != "input_ids"}
                    )
                    if generated.shape[1] > input_ids.shape[1]:
                        return True, "direct_generate"
            
            return False, "generate_no_output"
        
        except Exception as e:
            return False, f"generate_error: {str(e)}"
    
    def test_forward_pass_extraction(self, model, processor) -> Tuple[bool, str]:
        """Test custom forward pass text extraction"""
        try:
            test_image = Image.new('RGB', (100, 50), color=(255, 255, 255))
            
            # Prepare inputs
            inputs = processor(text="<ocr>", images=test_image, return_tensors="pt")
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            # Convert inputs to model's device/dtype
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype.is_floating_point:
                        inputs[key] = value.to(dtype=model_dtype, device=model_device)
                    else:
                        inputs[key] = value.to(device=model_device)
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Check if we get logits
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits
                    if logits.numel() > 0:
                        return True, "forward_pass_logits"
                
                # Check for prediction_logits (some Kosmos models use this)
                if hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
                    logits = outputs.prediction_logits
                    if logits.numel() > 0:
                        return True, "forward_pass_prediction_logits"
                
                return False, "no_usable_logits"
        
        except Exception as e:
            return False, f"forward_pass_error: {str(e)}"
    
    def test_custom_token_generation(self, model, processor) -> Tuple[bool, str]:
        """Test custom token-by-token generation"""
        try:
            test_image = Image.new('RGB', (100, 50), color=(255, 255, 255))
            
            # Prepare inputs
            inputs = processor(text="<ocr>", images=test_image, return_tensors="pt")
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            
            # Convert inputs to model's device/dtype
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype.is_floating_point:
                        inputs[key] = value.to(dtype=model_dtype, device=model_device)
                    else:
                        inputs[key] = value.to(device=model_device)
            
            # Try custom generation loop
            with torch.no_grad():
                input_ids = inputs["input_ids"]
                generated_tokens = []
                
                for step in range(3):  # Generate a few tokens
                    # Forward pass
                    outputs = model(**inputs)
                    
                    # Get logits from different possible locations
                    logits = None
                    if hasattr(outputs, 'logits') and outputs.logits is not None:
                        logits = outputs.logits
                    elif hasattr(outputs, 'prediction_logits') and outputs.prediction_logits is not None:
                        logits = outputs.prediction_logits
                    
                    if logits is None:
                        return False, "no_logits_in_output"
                    
                    # Get next token
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for EOS
                    if next_token.item() == processor.tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token.item())
                    
                    # Update input_ids for next iteration
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    inputs["input_ids"] = input_ids
                
                if len(generated_tokens) > 0:
                    return True, "custom_token_generation"
                else:
                    return False, "no_tokens_generated"
        
        except Exception as e:
            return False, f"custom_generation_error: {str(e)}"
    
    def comprehensive_generation_test(self, model, processor) -> Dict:
        """Run comprehensive generation capability testing"""
        print(f"        🔍 Testing generation capabilities...")
        
        # Test 1: Standard generation methods
        has_generate, gen_method = self.test_standard_generation(model, processor)
        print(f"        {'✅' if has_generate else '❌'} Standard generation: {gen_method}")
        
        # Test 2: Forward pass extraction
        has_forward, forward_method = self.test_forward_pass_extraction(model, processor)
        print(f"        {'✅' if has_forward else '❌'} Forward pass: {forward_method}")
        
        # Test 3: Custom token generation
        has_custom, custom_method = self.test_custom_token_generation(model, processor)
        print(f"        {'✅' if has_custom else '❌'} Custom generation: {custom_method}")
        
        # Determine best method
        best_method = "none"
        if has_generate:
            best_method = gen_method
        elif has_custom:
            best_method = custom_method
        elif has_forward:
            best_method = forward_method
        
        return {
            "has_standard_generate": has_generate,
            "standard_method": gen_method,
            "has_forward_pass": has_forward,
            "forward_method": forward_method,
            "has_custom_generation": has_custom,
            "custom_method": custom_method,
            "best_method": best_method,
            "any_method_works": has_generate or has_forward or has_custom
        }
    
    def quantize_nf4_double_quant(self) -> QuantizationResult:
        """NF4 with double quantization"""
        print("\n🔧 Quantizing with NF4 + Double Quantization...")
        
        try:
            start_time = time.time()
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
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
            
            print(f"    ✅ Model loaded: {type(model).__name__}")
            
            # Comprehensive generation testing
            gen_results = self.comprehensive_generation_test(model, processor)
            
            # Calculate model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            # Save model
            output_path = self.output_dir / "NF4_DoubleQuant"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            # Save comprehensive metadata
            metadata = {
                "quantization_method": "NF4_DoubleQuant",
                "model_size_mb": model_size_mb,
                "model_dtype": str(next(model.parameters()).dtype),
                "generation_capabilities": gen_results,
                "huggingface_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "torch.float16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            }
            
            with open(output_path / "quantization_info.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                method="NF4_DoubleQuant",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=4.0,
                has_generate=gen_results["has_standard_generate"],
                has_language_model=hasattr(model, 'language_model'),
                forward_pass_works=gen_results["has_forward_pass"],
                custom_generation_works=gen_results["has_custom_generation"],
                best_extraction_method=gen_results["best_method"],
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Best method: {gen_results['best_method']}")
            
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
                has_generate=False,
                has_language_model=False,
                forward_pass_works=False,
                custom_generation_works=False,
                best_extraction_method="none",
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_fp4_bfloat16(self) -> QuantizationResult:
        """FP4 with BFloat16"""
        print("\n🔧 Quantizing with FP4 + BFloat16...")
        
        try:
            start_time = time.time()
            
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
            
            print(f"    ✅ Model loaded with BF16")
            gen_results = self.comprehensive_generation_test(model, processor)
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            output_path = self.output_dir / "FP4_BFloat16"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "FP4_BFloat16",
                "model_size_mb": model_size_mb,
                "model_dtype": str(next(model.parameters()).dtype),
                "generation_capabilities": gen_results,
                "huggingface_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "torch.bfloat16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "fp4"
                }
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
                has_generate=gen_results["has_standard_generate"],
                has_language_model=hasattr(model, 'language_model'),
                forward_pass_works=gen_results["has_forward_pass"],
                custom_generation_works=gen_results["has_custom_generation"],
                best_extraction_method=gen_results["best_method"],
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Best method: {gen_results['best_method']}")
            
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
                has_generate=False,
                has_language_model=False,
                forward_pass_works=False,
                custom_generation_works=False,
                best_extraction_method="none",
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_fp4_standard(self) -> QuantizationResult:
        """Standard FP4 quantization"""
        print("\n🔧 Quantizing with Standard FP4...")
        
        try:
            start_time = time.time()
            
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
            
            print(f"    ✅ Standard FP4 loaded")
            gen_results = self.comprehensive_generation_test(model, processor)
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            output_path = self.output_dir / "FP4_Standard"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "FP4_Standard",
                "model_size_mb": model_size_mb,
                "model_dtype": str(next(model.parameters()).dtype),
                "generation_capabilities": gen_results,
                "huggingface_config": {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "torch.float16",
                    "bnb_4bit_use_double_quant": False,
                    "bnb_4bit_quant_type": "fp4"
                }
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
                has_generate=gen_results["has_standard_generate"],
                has_language_model=hasattr(model, 'language_model'),
                forward_pass_works=gen_results["has_forward_pass"],
                custom_generation_works=gen_results["has_custom_generation"],
                best_extraction_method=gen_results["best_method"],
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Best method: {gen_results['best_method']}")
            
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
                has_generate=False,
                has_language_model=False,
                forward_pass_works=False,
                custom_generation_works=False,
                best_extraction_method="none",
                model_dtype="",
                error_message=str(e)
            )
    
    def quantize_int8_fallback(self) -> QuantizationResult:
        """INT8 quantization fallback"""
        print("\n🔧 Quantizing with INT8 (Fallback)...")
        
        try:
            start_time = time.time()
            
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
            gen_results = self.comprehensive_generation_test(model, processor)
            
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / (1024 * 1024)
            
            output_path = self.output_dir / "INT8_Fallback"
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            metadata = {
                "quantization_method": "INT8_Fallback",
                "model_size_mb": model_size_mb,
                "model_dtype": str(next(model.parameters()).dtype),
                "generation_capabilities": gen_results,
                "huggingface_config": {
                    "load_in_8bit": True
                }
            }
            
            with open(output_path / "quantization_info.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            quantization_time = time.time() - start_time
            
            result = QuantizationResult(
                method="INT8_Fallback",
                success=True,
                model_size_mb=model_size_mb,
                quantization_time=quantization_time,
                compression_ratio=2.0,
                has_generate=gen_results["has_standard_generate"],
                has_language_model=hasattr(model, 'language_model'),
                forward_pass_works=gen_results["has_forward_pass"],
                custom_generation_works=gen_results["has_custom_generation"],
                best_extraction_method=gen_results["best_method"],
                model_dtype=str(next(model.parameters()).dtype)
            )
            
            print(f"    ✅ Success: {model_size_mb:.1f}MB, {quantization_time:.1f}s")
            print(f"    🎯 Best method: {gen_results['best_method']}")
            
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
                has_generate=False,
                has_language_model=False,
                forward_pass_works=False,
                custom_generation_works=False,
                best_extraction_method="none",
                model_dtype="",
                error_message=str(e)
            )
    
    def run_complete_quantization_suite(self) -> List[QuantizationResult]:
        """Run comprehensive quantization suite with generation testing"""
        print("🚀 ADVANCED KOSMOS-2.5 FP4 QUANTIZATION SUITE")
        print("=" * 80)
        print("📋 Multiple quantization methods with comprehensive generation testing")
        print("🔍 Tests: Standard generate(), Forward pass, Custom token generation")  
        print("=" * 80)
        
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
                time.sleep(2)  # Brief pause between methods
                
            except Exception as e:
                print(f"❌ Method {method.__name__} completely failed: {e}")
                method_name = method.__name__.replace('quantize_', '').upper()
                results.append(QuantizationResult(
                    method=method_name,
                    success=False,
                    model_size_mb=0,
                    quantization_time=0,
                    compression_ratio=0,
                    has_generate=False,
                    has_language_model=False,
                    forward_pass_works=False,
                    custom_generation_works=False,
                    best_extraction_method="none",
                    model_dtype="",
                    error_message=str(e)
                ))
        
        self.generate_comprehensive_report(results)
        return results
    
    def generate_comprehensive_report(self, results: List[QuantizationResult]):
        """Generate detailed quantization report with text extraction capabilities"""
        successful = [r for r in results if r.success]
        
        # Categorize by text extraction capability
        has_generate = [r for r in successful if r.has_generate]
        has_forward = [r for r in successful if r.forward_pass_works]
        has_custom = [r for r in successful if r.custom_generation_works]
        any_extraction = [r for r in successful if r.best_extraction_method != "none"]
        
        failed = [r for r in results if not r.success]
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE QUANTIZATION SUITE RESULTS")
        print('='*80)
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  ✅ Successfully quantized: {len(successful)}/{len(results)}")
        print(f"  🎯 With text extraction capability: {len(any_extraction)}/{len(results)}")
        print(f"  ❌ Complete failures: {len(failed)}")
        
        print(f"\n🔍 TEXT EXTRACTION CAPABILITIES:")
        print(f"  🚀 Standard generate() method: {len(has_generate)}")
        print(f"  🔧 Forward pass extraction: {len(has_forward)}")
        print(f"  ⚙️  Custom token generation: {len(has_custom)}")
        
        if any_extraction:
            print(f"\n🎯 WORKING QUANTIZED MODELS:")
            print(f"{'Method':<20} {'Size(MB)':<10} {'Time(s)':<8} {'Best Extraction':<25} {'Dtype':<15}")
            print("-" * 85)
            
            for r in any_extraction:
                print(f"{r.method:<20} {r.model_size_mb:<10.1f} {r.quantization_time:<8.1f} "
                      f"{r.best_extraction_method:<25} {r.model_dtype:<15}")
            
            # Find best options
            best_compression = min(any_extraction, key=lambda x: x.model_size_mb)
            fastest_setup = min(any_extraction, key=lambda x: x.quantization_time)
            
            print(f"\n💡 RECOMMENDATIONS:")
            print(f"  🥇 Best Compression: {best_compression.method}")
            print(f"     Size: {best_compression.model_size_mb:.1f}MB")
            print(f"     Extraction: {best_compression.best_extraction_method}")
            print(f"     Usage: python advanced_ocr_inference.py --model_path ./quantized_models/{best_compression.method}")
            
            print(f"\n  ⚡ Fastest Setup: {fastest_setup.method}")
            print(f"     Time: {fastest_setup.quantization_time:.1f}s")
            print(f"     Extraction: {fastest_setup.best_extraction_method}")
            print(f"     Usage: python advanced_ocr_inference.py --model_path ./quantized_models/{fastest_setup.method}")
            
            # Show which methods work for different extraction types
            print(f"\n🔧 EXTRACTION METHOD COMPATIBILITY:")
            if has_generate:
                print(f"  ✅ Standard Generate: {', '.join([r.method for r in has_generate])}")
            if has_forward:
                print(f"  ✅ Forward Pass: {', '.join([r.method for r in has_forward])}")
            if has_custom:
                print(f"  ✅ Custom Generation: {', '.join([r.method for r in has_custom])}")
        
        else:
            print(f"\n❌ NO WORKING TEXT EXTRACTION METHODS FOUND")
            print(f"All quantized models failed text extraction testing.")
        
        if failed:
            print(f"\n❌ FAILED QUANTIZATION METHODS:")
            for r in failed:
                print(f"  {r.method}: {r.error_message}")
        
        # Save comprehensive report
        report_data = {
            "suite_name": "Advanced Kosmos-2.5 FP4 Quantization Suite",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": self.model_id,
            "device": self.device,
            "pytorch_version": torch.__version__,
            "summary": {
                "total_methods": len(results),
                "successful_quantizations": len(successful),
                "working_extraction_methods": len(any_extraction),
                "standard_generate_available": len(has_generate),
                "forward_pass_available": len(has_forward),
                "custom_generation_available": len(has_custom),
                "failed_methods": len(failed)
            },
            "results": [
                {
                    "method": r.method,
                    "success": r.success,
                    "model_size_mb": r.model_size_mb,
                    "quantization_time": r.quantization_time,
                    "compression_ratio": r.compression_ratio,
                    "has_generate": r.has_generate,
                    "has_language_model": r.has_language_model,
                    "forward_pass_works": r.forward_pass_works,
                    "custom_generation_works": r.custom_generation_works,
                    "best_extraction_method": r.best_extraction_method,
                    "model_dtype": r.model_dtype,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }
        
        report_path = self.output_dir / "advanced_quantization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📋 Comprehensive report saved: {report_path}")
        
        if any_extraction:
            print(f"\n🔧 NEXT STEPS - Test your quantized models:")
            for r in any_extraction:
                model_path = f"./quantized_models/{r.method}"
                print(f"  python advanced_ocr_inference.py --model_path {model_path}")
        else:
            print(f"\n⚠️  TROUBLESHOOTING REQUIRED")
            print(f"Consider using the original (non-quantized) model or")
            print(f"check GPU compatibility with quantization methods.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced FP4 Quantization Suite for Kosmos-2.5")
    parser.add_argument("--model_id", default="microsoft/kosmos-2.5",
                       help="Model ID to quantize")
    parser.add_argument("--output_dir", default="./quantized_models",
                       help="Output directory for quantized models")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Advanced Kosmos-2.5 FP4 Quantization Suite")
    print(f"📦 Model: {args.model_id}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🎮 GPU Memory: {gpu_memory:.1f}GB")
    
    quantizer = AdvancedKosmosQuantizer(
        model_id=args.model_id,
        output_dir=args.output_dir
    )
    
    try:
        results = quantizer.run_complete_quantization_suite()
        
        working_methods = [r for r in results if r.success and r.best_extraction_method != "none"]
        
        if working_methods:
            print(f"\n🎉 QUANTIZATION SUITE COMPLETED SUCCESSFULLY!")
            print(f"✅ {len(working_methods)} quantized models with working text extraction")
        else:
            print(f"\n❌ QUANTIZATION SUITE COMPLETED WITH ISSUES")
            print(f"No quantized models have working text extraction methods")
            
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
