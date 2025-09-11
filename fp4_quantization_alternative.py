"""
Simplified Alternative FP4 Quantization for Kosmos-2.5
Focuses on the most effective methods for your use case
"""

import torch
import os
import json
import time
from pathlib import Path
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig

class KosmosQuantizer:
    def __init__(self, model_id="microsoft/kosmos-2.5", output_dir="./quantized_models"):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def quantize_method_1_nf4_double(self):
        """NF4 with double quantization - Often the best balance"""
        print("Quantizing with NF4 + Double Quantization...")
        
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        return self._save_quantized_model("NF4_DoubleQuant", config)
    
    def quantize_method_2_fp4_bf16(self):
        """FP4 with BF16 compute - Good for newer GPUs"""
        print("Quantizing with FP4 + BF16...")
        
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4"
        )
        
        return self._save_quantized_model("FP4_BF16", config)
    
    def quantize_method_3_int8_fallback(self):
        """INT8 as fallback - More compatible"""
        print("Quantizing with INT8...")
        
        return self._save_quantized_model("INT8", {"load_in_8bit": True})
    
    def _save_quantized_model(self, method_name, config):
        """Save quantized model with given config"""
        try:
            start_time = time.time()
            
            if isinstance(config, BitsAndBytesConfig):
                model = AutoModel.from_pretrained(
                    self.model_id,
                    quantization_config=config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModel.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    trust_remote_code=True,
                    **config
                )
            
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Save model
            output_path = self.output_dir / method_name
            output_path.mkdir(exist_ok=True)
            
            model.save_pretrained(output_path)
            processor.save_pretrained(output_path)
            
            # Calculate size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            size_mb = param_size / 1024 / 1024
            
            # Save metadata
            metadata = {
                "method": method_name,
                "size_mb": size_mb,
                "quantization_time": time.time() - start_time,
                "config": str(config)
            }
            
            with open(output_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"✓ {method_name}: {size_mb:.1f}MB, {time.time() - start_time:.1f}s")
            
            del model, processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return True, size_mb
            
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")
            return False, 0

def main():
    quantizer = KosmosQuantizer()
    
    print("🚀 Starting Alternative Quantization for Kosmos-2.5")
    print("=" * 60)
    
    methods = [
        quantizer.quantize_method_1_nf4_double,
        quantizer.quantize_method_2_fp4_bf16, 
        quantizer.quantize_method_3_int8_fallback
    ]
    
    results = []
    for method in methods:
        success, size = method()
        results.append((method.__name__, success, size))
    
    print(f"\n📊 Results Summary:")
    print("-" * 40)
    for name, success, size in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}: {size:.1f}MB")
    
    successful = [r for r in results if r[1]]
    if successful:
        best = min(successful, key=lambda x: x[2])
        print(f"\n🏆 Best compression: {best[0]} ({best[2]:.1f}MB)")

if __name__ == "__main__":
    main()
