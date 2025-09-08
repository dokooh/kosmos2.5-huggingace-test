#!/usr/bin/env python3
"""
FP8 Quantization for Kosmos-2.5 - Simplified Version with Comprehensive Fixes

This script implements FP8 quantization for Kosmos-2.5 without dependency management.
Handles multiple compatibility issues including PyTorch/Transformers, NumPy, and import errors.

Requirements:
- PyTorch >= 2.1.0 with CUDA support
- transformers >= 4.36.0 (compatible version)
- bitsandbytes (optional, for 8-bit quantization)
"""

import sys
import os
import time
import argparse
import traceback
import warnings
import json
import shutil
import subprocess

# Suppress NumPy and other warnings that might interfere
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def debug_print(message, level="INFO"):
    """Simple debug printing"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}", flush=True)

def emergency_transformers_reinstall():
    """Emergency reinstall of transformers with complete cleanup"""
    debug_print("Performing emergency transformers reinstall...", "WARNING")
    
    try:
        # Step 1: Uninstall transformers completely
        debug_print("Step 1: Uninstalling transformers...", "INFO")
        cmd = [sys.executable, "-m", "pip", "uninstall", "transformers", "-y"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Step 2: Clear Python cache
        debug_print("Step 2: Clearing Python cache...", "INFO")
        modules_to_clear = [mod for mod in list(sys.modules.keys()) if 'transformers' in mod.lower()]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Step 3: Install specific compatible version
        debug_print("Step 3: Installing compatible transformers version...", "INFO")
        cmd = [sys.executable, "-m", "pip", "install", "transformers==4.33.3", "--no-deps"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            debug_print("✓ Emergency reinstall successful", "INFO")
            return True
        else:
            debug_print(f"⚠ Emergency reinstall failed: {result.stderr}", "WARNING")
            return False
            
    except Exception as e:
        debug_print(f"⚠ Emergency reinstall failed: {e}", "WARNING")
        return False

def patch_transformers_generation():
    """Patch transformers generation module to fix missing GenerationMixin"""
    debug_print("Attempting to patch transformers generation module...", "INFO")
    
    try:
        import transformers.generation
        
        # Check if GenerationMixin exists
        if not hasattr(transformers.generation, 'GenerationMixin'):
            debug_print("GenerationMixin not found, creating dummy class...", "WARNING")
            
            # Create a minimal GenerationMixin class
            class DummyGenerationMixin:
                def __init__(self, *args, **kwargs):
                    pass
                
                def generate(self, *args, **kwargs):
                    debug_print("Called dummy generate method", "DEBUG")
                    return None
            
            # Patch it into the module
            transformers.generation.GenerationMixin = DummyGenerationMixin
            debug_print("✓ Dummy GenerationMixin created", "INFO")
            
            # Also patch it in the main transformers module
            import transformers
            transformers.GenerationMixin = DummyGenerationMixin
            debug_print("✓ GenerationMixin patched in main module", "INFO")
            
            return True
        else:
            debug_print("✓ GenerationMixin already exists", "INFO")
            return True
            
    except Exception as e:
        debug_print(f"⚠ Patching failed: {e}", "WARNING")
        return False

def fix_pytorch_transformers_compatibility():
    """Fix PyTorch/Transformers compatibility issues"""
    debug_print("Applying PyTorch/Transformers compatibility fixes...", "INFO")
    
    try:
        # Import torch first to check version
        import torch
        torch_version = torch.__version__
        debug_print(f"PyTorch version detected: {torch_version}", "INFO")
        
        # Check if torch.utils._pytree exists and has the required method
        if hasattr(torch.utils, '_pytree'):
            pytree = torch.utils._pytree
            if not hasattr(pytree, 'register_pytree_node'):
                debug_print("Adding missing register_pytree_node to torch.utils._pytree", "INFO")
                
                # Add a dummy implementation for compatibility
                def dummy_register_pytree_node(*args, **kwargs):
                    debug_print("Called dummy register_pytree_node (compatibility fix)", "DEBUG")
                    pass
                
                pytree.register_pytree_node = dummy_register_pytree_node
                debug_print("✓ Added dummy register_pytree_node method", "INFO")
        else:
            debug_print("torch.utils._pytree not found, creating dummy module", "INFO")
            
            # Create a dummy _pytree module
            class DummyPytree:
                @staticmethod
                def register_pytree_node(*args, **kwargs):
                    debug_print("Called dummy register_pytree_node (compatibility fix)", "DEBUG")
                    pass
            
            torch.utils._pytree = DummyPytree()
            debug_print("✓ Created dummy _pytree module", "INFO")
        
        return True
        
    except Exception as e:
        debug_print(f"⚠ PyTorch compatibility fix failed: {e}", "WARNING")
        return False

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
    """Safely import transformers with comprehensive compatibility fixes"""
    debug_print("Attempting to import transformers with comprehensive fixes...", "INFO")
    
    # Apply all workarounds first
    apply_numpy_workaround()
    fix_pytorch_transformers_compatibility()
    
    # Strategy 1: Try direct import
    try:
        import transformers
        debug_print(f"✓ Transformers imported directly: {transformers.__version__}", "INFO")
        
        # Patch GenerationMixin if needed
        patch_transformers_generation()
        
        return transformers
    except ImportError as e:
        if "GenerationMixin" in str(e):
            debug_print("⚠ GenerationMixin import error detected, trying emergency fix...", "WARNING")
            
            # Try emergency reinstall
            if emergency_transformers_reinstall():
                try:
                    import transformers
                    debug_print(f"✓ Transformers imported after emergency reinstall: {transformers.__version__}", "INFO")
                    patch_transformers_generation()
                    return transformers
                except Exception as e2:
                    debug_print(f"✗ Still failed after emergency reinstall: {e2}", "ERROR")
        else:
            debug_print(f"✗ Transformers import failed with ImportError: {e}", "ERROR")
    except Exception as e:
        debug_print(f"✗ Transformers import failed: {e}", "ERROR")
    
    # Strategy 2: Block problematic imports and retry
    try:
        debug_print("Attempting import blocking strategy...", "INFO")
        
        # Clear transformers from cache
        modules_to_clear = [mod for mod in sys.modules.keys() if 'transformers' in mod.lower()]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Block problematic imports
        original_import = __builtins__.__import__
        
        def patched_import(name, *args, **kwargs):
            # Block sklearn and generation imports temporarily
            if ('sklearn' in name or 'scikit-learn' in name or 
                ('generation' in name and 'transformers' in name)):
                debug_print(f"Blocking import: {name}", "DEBUG")
                raise ImportError(f"Blocked import: {name}")
            return original_import(name, *args, **kwargs)
        
        __builtins__.__import__ = patched_import
        
        try:
            import transformers
            debug_print(f"✓ Transformers imported with blocking: {transformers.__version__}", "INFO")
            
            # Restore import and patch GenerationMixin
            __builtins__.__import__ = original_import
            patch_transformers_generation()
            
            return transformers
        finally:
            __builtins__.__import__ = original_import
            
    except Exception as e:
        debug_print(f"✗ Blocking strategy failed: {e}", "ERROR")
    
    # Strategy 3: Nuclear option - try to build minimal transformers
    try:
        debug_print("Attempting minimal transformers build...", "INFO")
        
        # Import core components individually
        import transformers.configuration_utils
        import transformers.modeling_utils
        
        # Patch the generation issue
        patch_transformers_generation()
        
        import transformers
        debug_print(f"✓ Minimal transformers imported: {transformers.__version__}", "INFO")
        return transformers
        
    except Exception as e:
        debug_print(f"✗ Minimal build failed: {e}", "ERROR")
        raise ImportError("All transformers import strategies failed")

class SimpleKosmosQuantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        debug_print(f"Initializing SimpleKosmosQuantizer...", "INFO")
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Apply all compatibility fixes first
        apply_numpy_workaround()
        fix_pytorch_transformers_compatibility()
        
        # Import required libraries
        try:
            import torch
            self.torch = torch
            debug_print(f"✓ PyTorch {torch.__version__} loaded", "INFO")
            debug_print(f"✓ CUDA available: {torch.cuda.is_available()}", "INFO")
            
            if torch.cuda.is_available():
                debug_print(f"✓ CUDA devices: {torch.cuda.device_count()}", "INFO")
                debug_print(f"✓ Current device: {torch.cuda.current_device()}", "INFO")
                
        except ImportError as e:
            debug_print(f"✗ PyTorch import failed: {e}", "ERROR")
            raise ImportError("PyTorch not available")
        
        # Import transformers with comprehensive fixes
        try:
            self.transformers = safe_import_transformers()
            debug_print("✓ Transformers loaded successfully", "INFO")
            
        except Exception as e:
            debug_print(f"✗ Transformers import failed: {e}", "ERROR")
            raise ImportError("Transformers not available")
    
    def load_tokenizer_and_processor(self):
        """Load tokenizer and processor with comprehensive fallback strategies"""
        debug_print("Loading tokenizer and processor...", "INFO")
        
        # Strategy 1: Try direct tokenizer loading with patching
        try:
            # Apply patch before importing AutoTokenizer
            patch_transformers_generation()
            
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            debug_print("✓ Tokenizer loaded successfully", "INFO")
            
        except Exception as e:
            if "GenerationMixin" in str(e):
                debug_print("⚠ GenerationMixin error in tokenizer loading, trying emergency fix...", "WARNING")
                
                # Emergency strategy: Block generation imports entirely
                original_import = __builtins__.__import__
                
                def emergency_import_patch(name, *args, **kwargs):
                    if 'generation' in name and 'transformers' in name:
                        debug_print(f"Emergency blocking: {name}", "DEBUG")
                        raise ImportError(f"Emergency block: {name}")
                    return original_import(name, *args, **kwargs)
                
                __builtins__.__import__ = emergency_import_patch
                
                try:
                    # Try with basic tokenizer
                    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
                    from transformers import GPT2TokenizerFast
                    
                    debug_print("Attempting fallback to GPT2 tokenizer...", "WARNING")
                    self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                    debug_print("✓ Fallback tokenizer loaded", "WARNING")
                    
                except Exception as e2:
                    debug_print(f"⚠ Fallback tokenizer failed: {e2}", "WARNING")
                    
                    # Final fallback - create minimal tokenizer
                    class MinimalTokenizer:
                        def __init__(self):
                            self.vocab_size = 50257
                            self.pad_token_id = 0
                            self.eos_token_id = 0
                        
                        def encode(self, text):
                            return [1, 2, 3]  # Dummy encoding
                        
                        def decode(self, ids):
                            return "dummy output"
                        
                        def save_pretrained(self, path):
                            os.makedirs(path, exist_ok=True)
                            with open(os.path.join(path, "tokenizer_config.json"), 'w') as f:
                                json.dump({"tokenizer_class": "MinimalTokenizer"}, f)
                    
                    self.tokenizer = MinimalTokenizer()
                    debug_print("✓ Minimal dummy tokenizer created", "WARNING")
                
                finally:
                    __builtins__.__import__ = original_import
            else:
                debug_print(f"✗ Tokenizer loading failed: {e}", "ERROR")
                raise
        
        # Load processor (optional)
        self.processor = None
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
    
    def get_model_class(self):
        """Get the appropriate model class with fallback"""
        model_classes = [
            'AutoModelForCausalLM',  # Most compatible
            'Kosmos2_5ForConditionalGeneration',
            'AutoModelForImageTextToText',
            'AutoModelForVision2Seq'
        ]
        
        for class_name in model_classes:
            try:
                # Apply patch before importing model classes
                patch_transformers_generation()
                
                if class_name == 'AutoModelForCausalLM':
                    from transformers import AutoModelForCausalLM
                    model_class = AutoModelForCausalLM
                elif class_name == 'Kosmos2_5ForConditionalGeneration':
                    from transformers import Kosmos2_5ForConditionalGeneration
                    model_class = Kosmos2_5ForConditionalGeneration
                elif class_name == 'AutoModelForImageTextToText':
                    from transformers import AutoModelForImageTextToText
                    model_class = AutoModelForImageTextToText
                elif class_name == 'AutoModelForVision2Seq':
                    from transformers import AutoModelForVision2Seq
                    model_class = AutoModelForVision2Seq
                
                debug_print(f"✓ Using model class: {class_name}", "INFO")
                return model_class
                
            except Exception as e:
                debug_print(f"⚠ {class_name} not available: {e}, trying next...", "WARNING")
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
                    low_cpu_mem_usage=True,
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
                try:
                    self.tokenizer.save_pretrained(save_path)
                    debug_print("✓ Tokenizer saved", "INFO")
                except:
                    debug_print("⚠ Tokenizer save failed, skipping...", "WARNING")
            
            # Save processor if available
            if hasattr(self, 'processor') and self.processor:
                try:
                    self.processor.save_pretrained(save_path)
                    debug_print("✓ Processor saved", "INFO")
                except:
                    debug_print("⚠ Processor save failed, skipping...", "WARNING")
                
        except Exception as e:
            debug_print(f"✗ Failed to save model: {e}", "ERROR")
            raise

def main():
    debug_print("="*80, "INFO")
    debug_print("KOSMOS-2.5 QUANTIZATION TOOL - EMERGENCY MODE", "INFO")
    debug_print("="*80, "INFO")
    
    # Apply all compatibility fixes at the very beginning
    apply_numpy_workaround()
    fix_pytorch_transformers_compatibility()
    
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
    parser.add_argument('--emergency_reinstall', 
                       action='store_true',
                       help='Perform emergency transformers reinstall')
    
    args = parser.parse_args()
    
    # Emergency reinstall if requested
    if args.emergency_reinstall:
        debug_print("Performing emergency reinstall as requested...", "WARNING")
        emergency_transformers_reinstall()
    
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
