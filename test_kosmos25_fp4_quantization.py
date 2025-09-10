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
        return psutil.virtual_memory().used / 1024 / 1024
    
    def prepare_inputs_with_dtype(self, processor, image, prompt, target_dtype, device):
        """Prepare inputs with correct dtype to match model"""
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    processed_inputs[key] = value.to(dtype=target_dtype, device=device)
                else:
                    processed_inputs[key] = value.to(device=device)
            else:
                processed_inputs[key] = value
        
        return processed_inputs
    
    def analyze_kosmos_output(self, outputs):
        """Analyze the Kosmos2_5ModelOutput structure"""
        output_info = {
            "type": str(type(outputs)),
            "attributes": [],
            "data": {}
        }
        
        # Get all attributes
        for attr in dir(outputs):
            if not attr.startswith('_'):
                output_info["attributes"].append(attr)
                try:
                    value = getattr(outputs, attr)
                    if isinstance(value, torch.Tensor):
                        output_info["data"][attr] = {
                            "type": "tensor",
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "device": str(value.device)
                        }
                    elif callable(value):
                        output_info["data"][attr] = {"type": "method"}
                    else:
                        output_info["data"][attr] = {"type": type(value).__name__, "value": str(value)[:100]}
                except:
                    output_info["data"][attr] = {"type": "error_accessing"}
        
        return output_info
    
    def run_kosmos_inference(self, model, processor, image, prompt="<ocr>"):
        """Run inference and analyze the Kosmos-2.5 output structure"""
        try:
            model_dtype = next(model.parameters()).dtype
            
            # Prepare inputs with correct dtype
            inputs = self.prepare_inputs_with_dtype(processor, image, prompt, model_dtype, self.device)
            
            # Run forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Analyze the output structure
            output_info = self.analyze_kosmos_output(outputs)
            
            # Try to extract meaningful information
            result_summary = {
                "output_type": output_info["type"],
                "available_attributes": output_info["attributes"],
                "tensor_attributes": []
            }
            
            # Look for common attributes that might contain the results
            for attr_name, attr_data in output_info["data"].items():
                if attr_data.get("type") == "tensor":
                    tensor_info = f"{attr_name}: {attr_data['shape']} ({attr_data['dtype']})"
                    result_summary["tensor_attributes"].append(tensor_info)
            
            # Try to access specific known attributes for Kosmos models
            if hasattr(outputs, 'last_hidden_state'):
                result_summary["last_hidden_state_shape"] = list(outputs.last_hidden_state.shape)
            
            if hasattr(outputs, 'logits'):
                result_summary["logits_shape"] = list(outputs.logits.shape)
            
            if hasattr(outputs, 'prediction_logits'):
                result_summary["prediction_logits_shape"] = list(outputs.prediction_logits.shape)
            
            # For text generation models, check for specific generation attributes
            for attr in ['logits', 'prediction_logits', 'language_model_outputs', 'text_outputs']:
                if hasattr(outputs, attr):
                    attr_value = getattr(outputs, attr)
                    if isinstance(attr_value, torch.Tensor):
                        result_summary[f"{attr}_found"] = True
                        result_summary[f"{attr}_shape"] = list(attr_value.shape)
            
            return result_summary
                
        except Exception as e:
            return {"error": str(e), "traceback": str(e)}
    
    def test_model_loading(self, quantization_config=None, test_name="", use_fp16=False):
        """Test model loading with different quantization settings"""
        print(f"\n{'='*60}")
        print(f"Testing {test_name}")
        print('='*60)
        
        try:
            # Prepare model loading arguments
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
            
            print(f"Loading model with configuration: {test_name}")
            print(f"Model arguments: {model_kwargs}")
            
            # Load model and processor
            model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            
            # Get model information
            model_dtype = next(model.parameters()).dtype
            model_device = next(model.parameters()).device
            size_mb, param_count = self.measure_model_size(model)
            
            print(f"✓ Model loaded successfully!")
            print(f"  Model dtype: {model_dtype}")
            print(f"  Model device: {model_device}")
            print(f"  Model size: {size_mb:.2f} MB")
            print(f"  Parameters: {param_count:,}")
            
            # Measure memory before inference
            memory_before = self.measure_memory()
            
            # Test inference
            image = self.get_test_image()
            print(f"\nRunning inference...")
            
            start_time = time.time()
            result = self.run_kosmos_inference(model, processor, image)
            inference_time = time.time() - start_time
            
            memory_after = self.measure_memory()
            memory_used = abs(memory_after - memory_before)
            
            print(f"✓ Inference completed in {inference_time:.2f}s")
            print(f"  Memory used: {memory_used:.2f} MB")
            print(f"  Output analysis:")
            
            if "error" not in result:
                print(f"    Output type: {result['output_type']}")
                print(f"    Available attributes: {len(result['available_attributes'])}")
                print(f"    Tensor attributes: {len(result['tensor_attributes'])}")
                
                for tensor_attr in result['tensor_attributes']:
                    print(f"      - {tensor_attr}")
                
                # Print any found logits or similar
                for key, value in result.items():
                    if 'shape' in key or 'found' in key:
                        print(f"    {key}: {value}")
            else:
                print(f"    Error: {result['error']}")
            
            # Cleanup
            del model
            del processor
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "size_mb": size_mb,
                "memory_mb": memory_used,
                "inference_time": inference_time,
                "param_count": param_count,
                "model_dtype": str(model_dtype),
                "output_analysis": result,
                "success": True
            }
            
        except Exception as e:
            print(f"✗ Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self):
        """Run comprehensive quantization testing"""
        print("Kosmos-2.5 Comprehensive Quantization Testing")
        print("=" * 60)
        
        # Define test configurations
        test_configs = [
            ("FP32_Baseline", None, False),
            ("FP16_Baseline", None, True),
            ("INT8_Quantization", None, False),
            ("FP4_Quantization", BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4"
            ), False),
            ("NF4_Quantization", BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ), False),
        ]
        
        results = {}
        
        # Run each test
        for test_name, quant_config, use_fp16 in test_configs:
            try:
                if test_name == "INT8_Quantization":
                    results[test_name] = self.test_model_loading(test_name="INT8")
                else:
                    results[test_name] = self.test_model_loading(
                        quantization_config=quant_config,
                        test_name=test_name,
                        use_fp16=use_fp16
                    )
            except Exception as e:
                print(f"Test {test_name} failed: {e}")
                results[test_name] = {"success": False, "error": str(e)}
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results):
        """Generate a comprehensive comparison report"""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE QUANTIZATION COMPARISON REPORT")
        print('='*80)
        
        # Table header
        print(f"{'Method':<20} {'Success':<8} {'Size (MB)':<12} {'Memory (MB)':<12} {'Time (s)':<10} {'Params':<12}")
        print('-' * 80)
        
        successful_results = {}
        
        # Print results table
        for method, result in results.items():
            if result.get("success", False):
                successful_results[method] = result
                size_mb = result.get('size_mb', 0)
                memory_mb = result.get('memory_mb', 0)
                time_s = result.get('inference_time', 0)
                params = result.get('param_count', 0)
                
                print(f"{method:<20} {'✓':<8} {size_mb:<12.1f} {memory_mb:<12.1f} {time_s:<10.2f} {params:<12,}")
            else:
                error = result.get('error', 'Unknown')[:20]
                print(f"{method:<20} {'✗':<8} {'Failed':<12} {'Failed':<12} {'Failed':<10} {error:<12}")
        
        # Compression analysis
        if len(successful_results) >= 2:
            print(f"\n{'='*60}")
            print("COMPRESSION ANALYSIS")
            print('='*60)
            
            # Use FP32 as baseline if available, otherwise FP16
            baseline_methods = ["FP32_Baseline", "FP16_Baseline"]
            baseline = None
            
            for method in baseline_methods:
                if method in successful_results:
                    baseline = method
                    break
            
            if baseline:
                baseline_result = successful_results[baseline]
                print(f"Using {baseline} as baseline")
                print(f"Baseline size: {baseline_result['size_mb']:.1f} MB")
                print()
                
                for method, result in successful_results.items():
                    if method != baseline:
                        size_ratio = baseline_result['size_mb'] / result['size_mb']
                        memory_ratio = baseline_result['memory_mb'] / result['memory_mb'] if result['memory_mb'] > 0 else 1
                        speed_ratio = baseline_result['inference_time'] / result['inference_time']
                        
                        print(f"{method}:")
                        print(f"  Size reduction: {size_ratio:.2f}x smaller ({100*(1-1/size_ratio):.1f}% reduction)")
                        print(f"  Memory efficiency: {memory_ratio:.2f}x")
                        print(f"  Speed ratio: {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'}")
                        print()
        
        # Success summary
        successful_count = len(successful_results)
        total_count = len(results)
        
        print(f"{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"Successful tests: {successful_count}/{total_count}")
        
        if successful_count > 0:
            print("✓ FP4 quantization testing completed successfully!")
            
            # Find best compression
            if len(successful_results) >= 2:
                quantized_methods = [m for m in successful_results.keys() if 'Quantization' in m]
                if quantized_methods:
                    best_compression = min(quantized_methods, 
                                         key=lambda x: successful_results[x]['size_mb'])
                    print(f"✓ Best compression method: {best_compression}")
                    print(f"  Size: {successful_results[best_compression]['size_mb']:.1f} MB")
        else:
            print("✗ No tests succeeded. Check your environment setup.")

if __name__ == "__main__":
    tester = Kosmos25QuantizationTester()
    results = tester.run_all_tests()
