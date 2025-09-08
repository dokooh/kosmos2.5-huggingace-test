#!/usr/bin/env python3
"""
FP8 Quantization for Kosmos-2.5 - Simplified Version with NumPy Fix

This script implements FP8 quantization for Kosmos-2.5 without dependency management.
Assumes all required packages are already installed and compatible.

Requirements:
- PyTorch >= 2.1.0 with CUDA support
- transformers >= 4.36.0
- bitsandbytes (optional, for 8-bit quantization)
"""

import sys
import os
import time
import argparse
import traceback
import warnings

# Suppress NumPy warnings that might interfere
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def debug_print(message, level="INFO"):
    """Simple debug printing"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}", flush=True)

def apply_numpy_workaround():
    """Apply NumPy compatibility workaround before any imports"""
    debug_print("Applying NumPy compatibility workaround...", "INFO")
    
    try:
        # Clear any existing problematic modules from cache
        modules_to_clear = [mod for mod in sys.modules.keys() 
                           if any(keyword in mod.lower() for keyword in ['sklearn', 'scipy', 'numpy', 'transformers'])]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
                debug_print(f"Cleared module: {mod}", "DEBUG")
        
        # Set environment variables to ignore NumPy API warnings
        os.environ['NUMPY_DISABLE_API_COMPATIBILITY_WARNING'] = '1'
        os.environ['SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL'] = 'True'
        
        # Try to import and fix NumPy first
        import numpy as np
        debug_print(f"✓ NumPy imported with workaround: {np.__version__}", "INFO")
        
        return True
        
    except Exception as e:
        debug_print(f"⚠ NumPy workaround failed: {e}", "WARNING")
        return False

def safe_import_transformers():
    """Safely import transformers with NumPy compatibility workaround"""
    debug_print("Attempting to import transformers with NumPy workaround...", "INFO")
    
    # Apply workaround first
    apply_numpy_workaround()
    
    # Strategy 1: Try direct import
    try:
        import transformers
        debug_print(f"✓ Transformers imported directly: {transformers.__version__}", "INFO")
        return transformers
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            debug_print("⚠ NumPy compatibility issue detected, trying advanced workaround...", "WARNING")
        else:
            debug_print(f"✗ Transformers import failed: {e}", "ERROR")
            raise
    
    # Strategy 2: Advanced workaround - monkey patch sklearn import
    try:
        debug_print("Attempting advanced NumPy compatibility fix...", "INFO")
        
        # Temporarily disable sklearn import in transformers
        original_import = __builtins__.__import__
        
        def patched_import(name, *args, **kwargs):
            if 'sklearn' in name or 'scikit-learn' in name:
                debug_print(f"Blocking sklearn import: {name}", "DEBUG")
                raise ImportError(f"Blocked sklearn import: {name}")
            return original_import(name, *args, **kwargs)
        
        # Apply patch temporarily
        __builtins__.__import__ = patched_import
        
        try:
            import transformers
            debug_print(f"✓ Transformers imported with sklearn blocking: {transformers.__version__}", "INFO")
            return transformers
        finally:
            # Restore original import
            __builtins__.__import__ = original_import
            
    except Exception as e:
        debug_print(f"✗ Advanced workaround failed: {e}", "ERROR")
        
        # Strategy 3: Last resort - try with minimal transformers
        try:
            debug_print("Trying minimal transformers import...", "INFO")
            
            # Import only what we need
            import transformers.models.auto.tokenization_auto
            import transformers.models.auto.modeling_auto
            
            import transformers
            debug_print(f"✓ Minimal transformers imported: {transformers.__version__}", "INFO")
            return transformers
            
        except Exception as e2:
            debug_print(f"✗ All import strategies failed: {e2}", "ERROR")
            raise

class SimpleKosmosQuantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        debug_print(f"Initializing SimpleKosmosQuantizer...", "INFO")
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Apply NumPy workaround first
        apply_numpy_workaround()
        
        # Import required libraries
        try:
            import torch
            self.torch = torch
            debug_print(f"✓ PyTorch {torch.__version__} loaded", "INFO")
            debug_print(f"✓ CUDA available: {torch.cuda.is_available()}", "INFO")
        except ImportError as e:
            debug_print(f"✗ PyTorch import failed: {e}", "ERROR")
            raise ImportError("PyTorch not available")
        
        # Import transformers with workaround
        try:
            self.transformers = safe_import_transformers()
            
            # Pre-import the components we'll need to avoid later import issues
            debug_print("Pre-importing transformers components...", "INFO")
            from transformers import AutoTokenizer, AutoProcessor
            debug_print("✓ Transformers components pre-imported", "INFO")
            
        except Exception as e:
            debug_print(f"✗ Transformers import failed: {e}", "ERROR")
            raise ImportError("Transformers not available")
    
    def load_tokenizer_and_processor(self):
        """Load tokenizer and processor"""
        debug_print("Loading tokenizer and processor...", "INFO")
        
        try:
            # Since we pre-imported, this should work now
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            debug_print("✓ Tokenizer loaded", "INFO")
        except Exception as e:
            debug_print(f"✗ Tokenizer loading failed: {e}", "ERROR")
            raise
        
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            debug_print("✓ Processor loaded", "INFO")
        except Exception as e:
            debug_print(f"⚠ Processor loading failed, using tokenizer only: {e}", "WARNING")
            self.processor = None
    
    def get_model_class(self):
        """Get the appropriate model class"""
        model_classes = [
            ('Kosmos2_5ForConditionalGeneration', 'transformers'),
            ('AutoModelForImageTextToText', 'transformers'),
            ('AutoModelForVision2Seq', 'transformers'),
            ('AutoModelForCausalLM', 'transformers')
        ]
        
        for class_name, module_name in model_classes:
            try:
                if class_name == 'Kosmos2_5ForConditionalGeneration':
                    from transformers import Kosmos2_5ForConditionalGeneration
                    model_class = Kosmos2_5ForConditionalGeneration
                elif class_name == 'AutoModelForImageTextToText':
                    from transformers import AutoModelForImageTextToText
                    model_class = AutoModelForImageTextToText
                elif class_name == 'AutoModelForVision2Seq':
                    from transformers import AutoModelForVision2Seq
                    model_class = AutoModelForVision2Seq
                elif class_name == 'AutoModelForCausalLM':
                    from transformers import AutoModelForCausalLM
                    model_class = AutoModelForCausalLM
                
                debug_print(f"✓ Using model class: {class_name}", "INFO")
                return model_class
                
            except ImportError:
                debug_print(f"⚠ {class_name} not available, trying next...", "WARNING")
                continue
        
        raise ImportError("No suitable model class found")
    
    def quantize_8bit(self, save_path="./kosmos2.5-8bit"):
        """8-bit quantization using BitsAndBytes"""
        debug_print("="*50, "INFO")
        debug_print("STARTING 8-BIT QUANTIZATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            # Load tokenizer and processor
            self.load_tokenizer_and_processor()
            
            # Get model class
            model_class = self.get_model_class()
            
            # Configure 8-bit quantization
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0,
                )
                debug_print("✓ BitsAndBytesConfig created", "INFO")
                
                # Load model with 8-bit quantization
                start_time = time.time()
                model = model_class.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=self.torch.float16,
                    cache_dir=self.cache_dir,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                load_time = time.time() - start_time
                debug_print(f"✓ Model loaded with 8-bit quantization in {load_time:.2f} seconds", "INFO")
                
            except ImportError:
                debug_print("BitsAndBytesConfig not available, loading in FP16...", "WARNING")
                start_time = time.time()
                model = model_class.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch.float16,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                load_time = time.time() - start_time
                debug_print(f"✓ Model loaded in FP16 in {load_time:.2f} seconds", "INFO")
            
            # Save model
            self._save_model(model, save_path)
            debug_print(f"✓ 8-bit quantized model saved to {save_path}", "INFO")
            
            return model
            
        except Exception as e:
            debug_print(f"✗ 8-bit quantization failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    def quantize_fp16(self, save_path="./kosmos2.5-fp16"):
        """FP16 quantization"""
        debug_print("="*50, "INFO")
        debug_print("STARTING FP16 QUANTIZATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            # Load tokenizer and processor
            self.load_tokenizer_and_processor()
            
            # Get model class
            model_class = self.get_model_class()
            
            # Load model in FP16
            start_time = time.time()
            model = model_class.from_pretrained(
                self.model_name,
                torch_dtype=self.torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            load_time = time.time() - start_time
            debug_print(f"✓ Model loaded in FP16 in {load_time:.2f} seconds", "INFO")
            
            # Set to eval mode for inference
            if hasattr(model, 'eval'):
                model.eval()
                debug_print("✓ Model set to eval mode", "INFO")
            
            # Save model
            self._save_model(model, save_path)
            debug_print(f"✓ FP16 model saved to {save_path}", "INFO")
            
            return model
            
        except Exception as e:
            debug_print(f"✗ FP16 quantization failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    def _save_model(self, model, save_path):
        """Save model, tokenizer, and processor"""
        debug_print(f"Saving model to: {save_path}", "INFO")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save model
            model.save_pretrained(save_path, safe_serialization=True)
            debug_print("✓ Model saved", "INFO")
            
            # Save tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
                debug_print("✓ Tokenizer saved", "INFO")
            
            # Save processor if available
            if hasattr(self, 'processor') and self.processor:
                self.processor.save_pretrained(save_path)
                debug_print("✓ Processor saved", "INFO")
                
        except Exception as e:
            debug_print(f"✗ Failed to save model: {e}", "ERROR")
            raise

def main():
    debug_print("="*80, "INFO")
    debug_print("KOSMOS-2.5 QUANTIZATION TOOL", "INFO")
    debug_print("="*80, "INFO")
    
    # Apply NumPy workaround at the very beginning
    apply_numpy_workaround()
    
    parser = argparse.ArgumentParser(description='Kosmos-2.5 Quantization Tool')
    parser.add_argument('--method', 
                       choices=['8bit', 'fp16'], 
                       required=True,
                       help='Quantization method')
    parser.add_argument('--model_name', 
                       default='microsoft/kosmos-2.5',
                       help='Model name or path')
    parser.add_argument('--save_path', 
                       required=True,
                       help='Path to save quantized model')
    parser.add_argument('--cache_dir', 
                       default=None,
                       help='Cache directory for model downloads')
    
    args = parser.parse_args()
    
    try:
        # Initialize quantizer
        debug_print("Initializing quantizer...", "INFO")
        quantizer = SimpleKosmosQuantizer(args.model_name, args.cache_dir)
        
        # Execute quantization based on method
        if args.method == '8bit':
            model = quantizer.quantize_8bit(args.save_path)
        elif args.method == 'fp16':
            model = quantizer.quantize_fp16(args.save_path)
        else:
            debug_print(f"✗ Unknown method: {args.method}", "ERROR")
            return 1
        
        if model is None:
            debug_print(f"✗ Quantization failed", "ERROR")
            return 1
        
        debug_print("="*80, "INFO")
        debug_print("✓ QUANTIZATION COMPLETED SUCCESSFULLY!", "INFO")
        debug_print(f"Method: {args.method.upper()}", "INFO")
        debug_print(f"Save path: {args.save_path}", "INFO")
        debug_print("="*80, "INFO")
        
        return 0
        
    except Exception as e:
        debug_print(f"✗ Quantization failed: {e}", "ERROR")
        debug_print(f"Full error: {traceback.format_exc()}", "DEBUG")
        return 1

if __name__ == "__main__":
    sys.exit(main())
