import torch
import time
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO

class Kosmos25QuantizationTester:
    def __init__(self):
        self.model_id = "microsoft/kosmos-2.5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_image_url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        
    def get_test_image(self):
        """Download a test image"""
        try:
            response = requests.get(self.test_image_url)
            return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"Failed to download test image: {e}")
            # Create a simple test image as fallback
            return Image.new('RGB', (300, 400), color=(255, 255, 255))
    
    def measure_model_size(self, model):
        """Measure model size in MB"""
        param_size = 0
        param_count = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_count += param.nelement()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb, param_count
    
    def measure_memory(self):
        """Measure GPU memory usage"""
        if self.device == "cuda" and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].memoryUsed
            except:
                pass
        return psutil.virtual_memory().used / 1024 / 1024  # MB
    
    def prepare_inputs_with_correct_dtype(self, processor, image, prompt, model_dtype):
        """Prepare inputs with correct dtype to match model"""
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Convert tensors to correct dtype and device
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    # Convert floating point tensors to model dtype
                    processed_inputs[key] = value.to(dtype=model_dtype, device=self.device)
                else:
                    # Keep integer tensors as-is but move to device
                    processed_inputs[key] = value.to(device=self.device)
            else:
                processed_inputs[key] = value
        
        return processed_inputs
    
    def run_kosmos_inference(self, model, processor, image, prompt="<ocr>"):
        """Run inference using the correct Kosmos-2.5 API with proper dtype handling"""
        try:
            # Determine model dtype
            model_dtype = next(model.parameters()).dtype
            print(f"Model dtype: {model_dtype}")
            
            # Prepare inputs with correct dtype
            inputs = self.prepare_inputs_with_correct_dtype(processor, image, prompt, model_dtype)
            
            print(f"Input dtypes: {[(k, v.dtype if hasattr(v, 'dtype') else type(v)) for k, v in inputs.items()]}")
            
            # Use the model's forward method directly
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"Output logits shape: {logits.shape}")
                
                # Get the most likely tokens for a sample of positions
                if logits.dim() >= 2:
                    predicted_ids = torch.argmax(logits, dim=-1)
                    
                    # Decode a portion of the prediction
                    if predicted_ids.dim() > 1:
                        # Take first sequence and first 50 tokens
                        sample_ids = predicted_ids[0][:50]
                    else:
                        sample_ids = predicted_ids[:50]
                    
                    decoded = processor.tokenizer.decode(sample_ids, skip_special_tokens=True)
                    return f"Decoded sample: {decoded}"
                else:
                    return f"Logits shape: {logits.shape}, unable to decode"
            else:
                return f"Model output type: {type(outputs)}, no logits found"
                
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return f"Inference failed: {str(e)}"
    
    def test_model_loading(self, quantization_config=None, test_name="", use_fp16=False):
        """Generic method to test model loading with different quantization settings"""
        print(f"\n=== Testing {test_name} ===")
        
        try:
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            elif use_fp16:
                model_kwargs["torch_dtype"] = torch.float16
            elif test_name == "INT8":
                model_kwargs["load_in_8bit"] = True
            
            print(f"Loading model with kwargs: {model_kwargs}")
            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            
            print(f"Model loaded successfully! Type: {type(model)}")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Model dtype: {next(model.parameters()).dtype}")
            
            size_mb, param_count = self.measure_model_size(model)
            memory_before = self.measure_memory()
            
            # Test inference
            image = self.get_test_image()
            
            start_time = time.time()
            result = self.run_kosmos_inference(model, processor, image)
            inference_time = time.time() - start_time
            
            memory_after = self.measure_memory()
            
            print(f"Model size: {size_mb:.2f} MB")
            print(f"Parameters: {param_count:,}")
            print(f"Memory usage: {abs(memory_after - memory_before):.2f} MB")
            print(f"Inference time: {inference_time:.2f}s")
            print(f"Output: {str(result)[:300]}...")
            
            # Cleanup
            del model
            del processor
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "size_mb": size_mb,
                "memory_mb": abs(memory_after - memory_before),
                "inference_time": inference_time,
                "output": result,
                "success": True
            }
            
        except Exception as e:
            print(f"Error in {test_name} test: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def test_fp16_baseline(self):
        """Test baseline FP16 model"""
        return self.test_model_loading(test_name="FP16 Baseline", use_fp16=True)
    
    def test_fp32_baseline(self):
        """Test baseline FP32 model"""
        return self.test_model_loading(test_name="FP32 Baseline")
    
    def test_fp4_bitsandbytes(self):
        """Test FP4 quantization using BitsAndBytes"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4"
        )
        return self.test_model_loading(quantization_config, "FP4 BitsAndBytes")
    
    def test_nf4_bitsandbytes(self):
        """Test NF4 quantization using BitsAndBytes"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        return self.test_model_loading(quantization_config, "NF4 BitsAndBytes")
    
    def test_int8_quantization(self):
        """Test INT8 quantization"""
        return self.test_model_loading(test_name="INT8")
    
    def run_all_tests(self):
        """Run all quantization tests and compare results"""
        test_methods = [
            ("fp32", self.test_fp32_baseline),
            ("fp16", self.test_fp16_baseline),
            ("int8", self.test_int8_quantization),
            ("fp4", self.test_fp4_bitsandbytes),
            ("nf4", self.test_nf4_bitsandbytes),
        ]
        
        results = {}
        
        for name, method in test_methods:
            try:
                print(f"\n{'='*50}")
                print(f"Starting test: {name}")
                print(f"{'='*50}")
                results[name] = method()
                
                if results[name].get("success", False):
                    print(f"✓ {name} test completed successfully")
                else:
                    print(f"✗ {name} test failed")
                    
            except Exception as e:
                print(f"Test {name} failed with error: {e}")
                results[name] = {"success": False, "error": str(e)}
        
        # Print comparison
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Method':<15} {'Success':<8} {'Size (MB)':<12} {'Memory (MB)':<12} {'Time (s)':<10}")
        print("-" * 80)
        
        successful_results = {}
        for method, result in results.items():
            if result.get("success", False):
                successful_results[method] = result
                print(f"{method:<15} {'✓':<8} {result['size_mb']:<12.2f} {result['memory_mb']:<12.2f} {result['inference_time']:<10.2f}")
            else:
                error_msg = result.get("error", "Unknown error")[:30]
                print(f"{method:<15} {'✗':<8} {'Failed':<12} {'Failed':<12} {error_msg:<10}")
        
        # Calculate compression ratios for successful tests
        if len(successful_results) >= 2:
            print(f"\n{'='*50}")
            print("COMPRESSION ANALYSIS")
            print("="*50)
            
            baseline_methods = ["fp32", "fp16"]
            quantized_methods = ["fp4", "nf4", "int8"]
            
            for baseline in baseline_methods:
                if baseline in successful_results:
                    for quantized in quantized_methods:
                        if quantized in successful_results:
                            compression_ratio = successful_results[baseline]["size_mb"] / successful_results[quantized]["size_mb"]
                            speed_ratio = successful_results[baseline]["inference_time"] / successful_results[quantized]["inference_time"]
                            memory_ratio = successful_results[baseline]["memory_mb"] / successful_results[quantized]["memory_mb"]
                            
                            print(f"{quantized.upper()} vs {baseline.upper()}:")
                            print(f"  Size compression: {compression_ratio:.2f}x smaller")
                            print(f"  Speed ratio: {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'}")
                            print(f"  Memory ratio: {memory_ratio:.2f}x {'less' if memory_ratio > 1 else 'more'} memory")
                            print()
        
        return results

if __name__ == "__main__":
    print("Kosmos-2.5 FP4 Quantization Tester")
    print("=" * 50)
    print("This script tests various quantization methods for Kosmos-2.5")
    print("Required packages:")
    print("pip install transformers accelerate bitsandbytes torch torchvision pillow requests psutil")
    if GPUtil is None:
        print("Optional: pip install gputil (for GPU memory monitoring)")
    print()
    
    tester = Kosmos25QuantizationTester()
    results = tester.run_all_tests()
    
    # Print final summary
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Successful tests: {successful_tests}/{len(results)}")
    
    if successful_tests == 0:
        print("\nNo tests succeeded. Common issues:")
        print("- Missing dependencies (bitsandbytes, accelerate)")
        print("- Insufficient GPU memory")
        print("- Model compatibility issues")
        print("- CUDA/PyTorch version mismatches")
        print("\nTry running the debug script first: python debug_kosmos25.py")
    elif successful_tests < len(results):
        print(f"\nSome tests failed. This is normal - not all quantization methods")
        print(f"work on every system. Focus on the successful methods.")
