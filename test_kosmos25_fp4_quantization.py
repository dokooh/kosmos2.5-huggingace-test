import torch
import time
import psutil
import GPUtil
from transformers import AutoModel, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests
from io import BytesIO
import numpy as np

class Kosmos25QuantizationTester:
    def __init__(self):
        self.model_id = "microsoft/kosmos-2.5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_image_url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        
    def get_test_image(self):
        """Download a test image"""
        response = requests.get(self.test_image_url)
        return Image.open(BytesIO(response.content))
    
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
        if self.device == "cuda":
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        return psutil.virtual_memory().used / 1024  # MB
    
    def test_fp16_baseline(self):
        """Test baseline FP16 model"""
        print("\n=== Testing FP16 Baseline ===")
        
        # Load model in FP16
        model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        size_mb, param_count = self.measure_model_size(model)
        memory_before = self.measure_memory()
        
        # Test inference
        image = self.get_test_image()
        prompt = "<ocr>"
        
        start_time = time.time()
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
            )
        
        inference_time = time.time() - start_time
        memory_after = self.measure_memory()
        
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Model size: {size_mb:.2f} MB")
        print(f"Parameters: {param_count:,}")
        print(f"Memory usage: {memory_after - memory_before:.2f} MB")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Output preview: {result[:100]}...")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return {
            "size_mb": size_mb,
            "memory_mb": memory_after - memory_before,
            "inference_time": inference_time
        }
    
    def test_fp4_bitsandbytes(self):
        """Test FP4 quantization using BitsAndBytes"""
        print("\n=== Testing FP4 with BitsAndBytes ===")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4"  # Use FP4 instead of NF4
        )
        
        # Load model with 4-bit quantization
        model = AutoModel.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        size_mb, param_count = self.measure_model_size(model)
        memory_before = self.measure_memory()
        
        # Test inference
        image = self.get_test_image()
        prompt = "<ocr>"
        
        start_time = time.time()
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
            )
        
        inference_time = time.time() - start_time
        memory_after = self.measure_memory()
        
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Model size: {size_mb:.2f} MB")
        print(f"Parameters: {param_count:,}")
        print(f"Memory usage: {memory_after - memory_before:.2f} MB")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Output preview: {result[:100]}...")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return {
            "size_mb": size_mb,
            "memory_mb": memory_after - memory_before,
            "inference_time": inference_time
        }
    
    def test_nf4_bitsandbytes(self):
        """Test NF4 quantization using BitsAndBytes"""
        print("\n=== Testing NF4 with BitsAndBytes ===")
        
        # Configure 4-bit quantization with NF4
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # Normal Float 4
        )
        
        # Load model with 4-bit quantization
        model = AutoModel.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        size_mb, param_count = self.measure_model_size(model)
        memory_before = self.measure_memory()
        
        # Test inference
        image = self.get_test_image()
        prompt = "<ocr>"
        
        start_time = time.time()
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
            )
        
        inference_time = time.time() - start_time
        memory_after = self.measure_memory()
        
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Model size: {size_mb:.2f} MB")
        print(f"Parameters: {param_count:,}")
        print(f"Memory usage: {memory_after - memory_before:.2f} MB")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Output preview: {result[:100]}...")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return {
            "size_mb": size_mb,
            "memory_mb": memory_after - memory_before,
            "inference_time": inference_time
        }
    
    def test_dynamic_quantization(self):
        """Test PyTorch dynamic quantization"""
        print("\n=== Testing Dynamic Quantization ===")
        
        # Load model normally
        model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Apply dynamic quantization
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        if self.device == "cuda":
            model = model.to(self.device)
        
        size_mb, param_count = self.measure_model_size(model)
        memory_before = self.measure_memory()
        
        # Test inference
        image = self.get_test_image()
        prompt = "<ocr>"
        
        start_time = time.time()
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
            )
        
        inference_time = time.time() - start_time
        memory_after = self.measure_memory()
        
        result = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Model size: {size_mb:.2f} MB")
        print(f"Parameters: {param_count:,}")
        print(f"Memory usage: {memory_after - memory_before:.2f} MB")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Output preview: {result[:100]}...")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return {
            "size_mb": size_mb,
            "memory_mb": memory_after - memory_before,
            "inference_time": inference_time
        }
    
    def run_all_tests(self):
        """Run all quantization tests and compare results"""
        results = {}
        
        try:
            # Test baseline FP16
            results["fp16"] = self.test_fp16_baseline()
        except Exception as e:
            print(f"FP16 test failed: {e}")
            results["fp16"] = None
        
        try:
            # Test FP4 quantization
            results["fp4"] = self.test_fp4_bitsandbytes()
        except Exception as e:
            print(f"FP4 test failed: {e}")
            results["fp4"] = None
        
        try:
            # Test NF4 quantization
            results["nf4"] = self.test_nf4_bitsandbytes()
        except Exception as e:
            print(f"NF4 test failed: {e}")
            results["nf4"] = None
        
        try:
            # Test dynamic quantization
            results["dynamic"] = self.test_dynamic_quantization()
        except Exception as e:
            print(f"Dynamic quantization test failed: {e}")
            results["dynamic"] = None
        
        # Print comparison
        print("\n=== COMPARISON SUMMARY ===")
        print(f"{'Method':<15} {'Size (MB)':<12} {'Memory (MB)':<12} {'Time (s)':<10}")
        print("-" * 50)
        
        for method, result in results.items():
            if result:
                print(f"{method:<15} {result['size_mb']:<12.2f} {result['memory_mb']:<12.2f} {result['inference_time']:<10.2f}")
        
        # Calculate compression ratios
        if results.get("fp16") and results.get("fp4"):
            compression_ratio = results["fp16"]["size_mb"] / results["fp4"]["size_mb"]
            speedup = results["fp16"]["inference_time"] / results["fp4"]["inference_time"]
            print(f"\nFP4 Compression ratio: {compression_ratio:.2f}x")
            print(f"FP4 Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    # Install required packages
    print("Make sure you have installed the required packages:")
    print("pip install transformers accelerate bitsandbytes torch torchvision pillow requests psutil gputil")
    
    tester = Kosmos25QuantizationTester()
    tester.run_all_tests()
