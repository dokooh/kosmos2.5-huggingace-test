#!/usr/bin/env python3
"""
FP8 Quantization for Kosmos-2.5 with Enhanced Dependency Handling

This script implements multiple FP8 quantization approaches with robust dependency management:
1. Automatic dependency conflict resolution
2. Graceful fallbacks for version mismatches
3. CUDA version compatibility checking
4. Alternative import strategies
5. Comprehensive debugging output

Requirements (Auto-detected and fixed):
- GPU with Compute Capability >= 8.0 (recommended >= 8.9)
- PyTorch >= 2.1.0 with CUDA support
- Compatible transformers and torchvision versions
"""

import sys
import os
import warnings
import logging
import time
import argparse
import numpy as np
import subprocess
from typing import Optional, Dict, Any, Tuple
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('quantization_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def debug_print(message, level="INFO"):
    """Enhanced debug printing with multiple output methods"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] [{level}] {message}"
    
    # Print to console
    print(formatted_msg, flush=True)
    
    # Log with appropriate level
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)

def run_command(command, description=""):
    """Run a system command with detailed logging"""
    debug_print(f"Running command: {command}", "DEBUG")
    if description:
        debug_print(f"Purpose: {description}", "DEBUG")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            debug_print(f"✓ Command succeeded", "DEBUG")
            if result.stdout.strip():
                debug_print(f"Output: {result.stdout.strip()}", "DEBUG")
            return True, result.stdout
        else:
            debug_print(f"✗ Command failed with code {result.returncode}", "ERROR")
            if result.stderr.strip():
                debug_print(f"Error: {result.stderr.strip()}", "ERROR")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        debug_print(f"✗ Command timed out", "ERROR")
        return False, "Command timed out"
    except Exception as e:
        debug_print(f"✗ Command execution failed: {e}", "ERROR")
        return False, str(e)

def detect_cuda_version():
    """Detect available CUDA version on the system"""
    debug_print("Detecting CUDA version...", "INFO")
    
    # Try nvidia-smi first
    success, output = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits", 
                                  "Check NVIDIA driver version")
    
    if success:
        debug_print(f"NVIDIA driver detected: {output.strip()}", "INFO")
    
    # Try nvcc
    success, output = run_command("nvcc --version", "Check CUDA compiler version")
    if success:
        lines = output.strip().split('\n')
        for line in lines:
            if 'release' in line.lower():
                debug_print(f"CUDA compiler: {line.strip()}", "INFO")
                # Extract version like "release 11.8" or "release 12.4"
                if 'release' in line:
                    try:
                        version_part = line.split('release')[1].strip().split(',')[0].strip()
                        major_version = version_part.split('.')[0]
                        debug_print(f"Detected CUDA major version: {major_version}", "INFO")
                        return major_version
                    except:
                        pass
    
    # Try checking PyTorch CUDA version
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            debug_print(f"PyTorch CUDA version: {cuda_version}", "INFO")
            major_version = cuda_version.split('.')[0]
            debug_print(f"PyTorch CUDA major version: {major_version}", "INFO")
            return major_version
    except:
        pass
    
    debug_print("Could not detect CUDA version, assuming 11.8", "WARNING")
    return "11"

def fix_dependency_mismatch():
    """Automatically fix CUDA version mismatches"""
    debug_print("="*60, "INFO")
    debug_print("ATTEMPTING AUTOMATIC DEPENDENCY FIX", "INFO")
    debug_print("="*60, "INFO")
    
    # Detect CUDA version
    cuda_major = detect_cuda_version()
    
    # Determine the correct PyTorch index URL
    if cuda_major == "12":
        index_url = "https://download.pytorch.org/whl/cu121"
        cuda_version = "cu121"
    elif cuda_major == "11":
        index_url = "https://download.pytorch.org/whl/cu118"
        cuda_version = "cu118"
    else:
        debug_print(f"Unsupported CUDA version {cuda_major}, using CUDA 11.8", "WARNING")
        index_url = "https://download.pytorch.org/whl/cu118"
        cuda_version = "cu118"
    
    debug_print(f"Selected CUDA version: {cuda_version}", "INFO")
    debug_print(f"PyTorch index URL: {index_url}", "INFO")
    
    # Uninstall existing torch/torchvision to avoid conflicts
    debug_print("Uninstalling existing PyTorch packages...", "INFO")
    uninstall_cmd = "pip uninstall torch torchvision torchaudio -y"
    success, _ = run_command(uninstall_cmd, "Remove conflicting packages")
    
    if not success:
        debug_print("⚠ Failed to uninstall existing packages, continuing...", "WARNING")
    
    # Install compatible versions
    debug_print(f"Installing PyTorch with {cuda_version}...", "INFO")
    install_cmd = f"pip install torch torchvision torchaudio --index-url {index_url}"
    success, output = run_command(install_cmd, f"Install PyTorch for {cuda_version}")
    
    if success:
        debug_print("✓ PyTorch installation completed", "INFO")
        return True
    else:
        debug_print("✗ PyTorch installation failed", "ERROR")
        debug_print("Trying CPU-only version as fallback...", "INFO")
        
        # Fallback to CPU version
        cpu_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        success, _ = run_command(cpu_cmd, "Install CPU-only PyTorch")
        
        if success:
            debug_print("✓ CPU-only PyTorch installed successfully", "INFO")
            return True
        else:
            debug_print("✗ All installation attempts failed", "ERROR")
            return False

def check_and_install_dependencies(auto_fix=True):
    """Check and handle dependency issues with automatic fixing"""
    debug_print("="*60, "INFO")
    debug_print("STARTING ENHANCED DEPENDENCY CHECK", "INFO")
    debug_print("="*60, "INFO")
    
    # Check PyTorch first
    debug_print("Checking PyTorch installation...", "INFO")
    pytorch_ok = False
    cuda_mismatch = False
    
    try:
        import torch
        debug_print(f"✓ PyTorch version: {torch.__version__}", "INFO")
        debug_print(f"✓ PyTorch location: {torch.__file__}", "DEBUG")
        debug_print(f"✓ CUDA available: {torch.cuda.is_available()}", "INFO")
        
        if torch.cuda.is_available():
            debug_print(f"✓ CUDA version: {torch.version.cuda}", "INFO")
            debug_print(f"✓ CUDA device count: {torch.cuda.device_count()}", "INFO")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                debug_print(f"✓ GPU {i}: {device_name}", "INFO")
        else:
            debug_print("⚠ CUDA not available - will use CPU", "WARNING")
        
        pytorch_ok = True
        
    except ImportError as e:
        debug_print(f"✗ PyTorch not found: {e}", "ERROR")
        pytorch_ok = False
    except Exception as e:
        debug_print(f"✗ PyTorch check failed: {e}", "ERROR")
        pytorch_ok = False
    
    # Check for CUDA mismatch
    debug_print("Checking torchvision compatibility...", "INFO")
    try:
        import torchvision
        debug_print(f"✓ Torchvision version: {torchvision.__version__}", "INFO")
        debug_print(f"✓ Torchvision location: {torchvision.__file__}", "DEBUG")
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            debug_print(f"✗ CUDA version mismatch detected: {e}", "ERROR")
            cuda_mismatch = True
            
            if auto_fix:
                debug_print("Attempting automatic fix...", "INFO")
                if fix_dependency_mismatch():
                    debug_print("✓ Dependency mismatch fixed, retesting...", "INFO")
                    try:
                        # Force reimport
                        if 'torch' in sys.modules:
                            del sys.modules['torch']
                        if 'torchvision' in sys.modules:
                            del sys.modules['torchvision']
                        
                        import torch
                        import torchvision
                        debug_print("✓ PyTorch and torchvision now compatible", "INFO")
                        cuda_mismatch = False
                        pytorch_ok = True
                        
                    except Exception as e2:
                        debug_print(f"✗ Fix verification failed: {e2}", "ERROR")
                else:
                    debug_print("✗ Automatic fix failed", "ERROR")
            else:
                debug_print("Manual fix required:", "ERROR")
                cuda_major = detect_cuda_version()
                if cuda_major == "12":
                    debug_print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121", "ERROR")
                else:
                    debug_print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118", "ERROR")
        else:
            debug_print(f"✗ Torchvision error: {e}", "ERROR")
            
    except ImportError as e:
        debug_print(f"⚠ Torchvision not found: {e}", "WARNING")
        debug_print("This may not be critical for quantization", "INFO")
    
    # If we still have issues, return early
    if cuda_mismatch or not pytorch_ok:
        if not auto_fix:
            debug_print("✗ Please fix dependency issues manually and retry", "ERROR")
            return False
        else:
            debug_print("⚠ Some issues remain, but continuing with available components", "WARNING")
    
    # Check transformers with detailed debugging
    debug_print("Checking transformers library...", "INFO")
    try:
        import transformers
        debug_print(f"✓ Transformers version: {transformers.__version__}", "INFO")
        debug_print(f"✓ Transformers location: {transformers.__file__}", "DEBUG")
        
        # Try importing components individually with fallbacks
        debug_print("Testing transformers components...", "DEBUG")
        
        # AutoTokenizer (required)
        try:
            from transformers import AutoTokenizer
            debug_print("✓ AutoTokenizer import successful", "DEBUG")
        except ImportError as e:
            debug_print(f"✗ AutoTokenizer import failed: {e}", "ERROR")
            debug_print("Installing/upgrading transformers...", "INFO")
            success, _ = run_command("pip install --upgrade transformers>=4.36.0", 
                                   "Upgrade transformers library")
            if not success:
                return False
            
        # AutoProcessor (optional with fallback)
        try:
            from transformers import AutoProcessor
            debug_print("✓ AutoProcessor import successful", "DEBUG")
        except Exception as e:
            debug_print(f"⚠ AutoProcessor import failed: {e}", "WARNING")
            debug_print("Will use alternative import method", "INFO")
            
    except ImportError as e:
        debug_print(f"✗ Transformers not found: {e}", "ERROR")
        debug_print("Installing transformers...", "INFO")
        success, _ = run_command("pip install transformers>=4.36.0", "Install transformers")
        if not success:
            return False
    
    # Check optional dependencies
    debug_print("Checking optional dependencies...", "INFO")
    
    # BitsAndBytes
    try:
        import bitsandbytes
        debug_print(f"✓ BitsAndBytes version: {bitsandbytes.__version__}", "INFO")
    except ImportError:
        debug_print("⚠ BitsAndBytes not found (installing for 8-bit quantization)", "WARNING")
        success, _ = run_command("pip install bitsandbytes", "Install BitsAndBytes")
        if success:
            debug_print("✓ BitsAndBytes installed successfully", "INFO")
    
    # Accelerate
    try:
        import accelerate
        debug_print(f"✓ Accelerate version: {accelerate.__version__}", "INFO")
    except ImportError:
        debug_print("⚠ Accelerate not found (installing for optimization)", "WARNING")
        success, _ = run_command("pip install accelerate", "Install Accelerate")
        if success:
            debug_print("✓ Accelerate installed successfully", "INFO")
    
    debug_print("✓ Enhanced dependency check completed!", "INFO")
    debug_print("="*60, "INFO")
    return True

def safe_import_transformers():
    """Safely import transformers components with detailed debugging and fallbacks"""
    debug_print("Starting safe transformers import with fallbacks...", "INFO")
    components = {}
    
    # AutoTokenizer (required)
    debug_print("Importing AutoTokenizer...", "DEBUG")
    try:
        from transformers import AutoTokenizer
        components['AutoTokenizer'] = AutoTokenizer
        debug_print("✓ AutoTokenizer imported successfully", "DEBUG")
    except ImportError as e:
        debug_print(f"✗ Failed to import AutoTokenizer: {e}", "ERROR")
        return None
    
    # AutoProcessor (optional with multiple fallbacks)
    debug_print("Importing AutoProcessor...", "DEBUG")
    try:
        from transformers import AutoProcessor
        components['AutoProcessor'] = AutoProcessor
        debug_print("✓ AutoProcessor imported successfully", "DEBUG")
    except ImportError as e:
        debug_print(f"⚠ AutoProcessor not available: {e}", "WARNING")
        
        # Try alternative processor imports
        try:
            from transformers import CLIPProcessor
            components['AutoProcessor'] = CLIPProcessor
            debug_print("✓ CLIPProcessor imported as fallback", "WARNING")
        except ImportError:
            debug_print("Will use tokenizer only", "INFO")
            components['AutoProcessor'] = None
    
    # Model classes with comprehensive fallbacks
    debug_print("Importing model classes...", "DEBUG")
    model_classes_to_try = [
        ('Kosmos2_5ForConditionalGeneration', 'transformers'),
        ('AutoModelForImageTextToText', 'transformers'), 
        ('AutoModelForVision2Seq', 'transformers'),
        ('AutoModelForCausalLM', 'transformers'),
    ]
    
    model_imported = False
    for class_name, module_name in model_classes_to_try:
        try:
            if module_name == 'transformers':
                from transformers import AutoModelForImageTextToText
                if class_name == 'Kosmos2_5ForConditionalGeneration':
                    from transformers import Kosmos2_5ForConditionalGeneration
                    components['ModelClass'] = Kosmos2_5ForConditionalGeneration
                elif class_name == 'AutoModelForImageTextToText':
                    components['ModelClass'] = AutoModelForImageTextToText
                elif class_name == 'AutoModelForVision2Seq':
                    from transformers import AutoModelForVision2Seq
                    components['ModelClass'] = AutoModelForVision2Seq
                elif class_name == 'AutoModelForCausalLM':
                    from transformers import AutoModelForCausalLM
                    components['ModelClass'] = AutoModelForCausalLM
                    
            debug_print(f"✓ {class_name} imported successfully", "DEBUG")
            model_imported = True
            break
            
        except ImportError:
            debug_print(f"⚠ {class_name} not found, trying next...", "DEBUG")
            continue
    
    if not model_imported:
        debug_print("✗ Failed to import any model class", "ERROR")
        return None
    
    # Quantization configs
    debug_print("Importing quantization configs...", "DEBUG")
    try:
        from transformers import BitsAndBytesConfig
        components['BitsAndBytesConfig'] = BitsAndBytesConfig
        debug_print("✓ BitsAndBytesConfig imported", "DEBUG")
    except ImportError as e:
        debug_print(f"⚠ BitsAndBytesConfig not available: {e}", "WARNING")
        components['BitsAndBytesConfig'] = None
    
    debug_print(f"✓ Safe import completed. Available components: {list(components.keys())}", "INFO")
    return components

def check_fp8_compatibility():
    """Check if the system supports FP8 quantization with detailed debugging"""
    debug_print("Checking FP8 compatibility...", "INFO")
    
    try:
        import torch
    except ImportError:
        debug_print("✗ PyTorch not available for FP8 check", "ERROR")
        return False
        
    if not torch.cuda.is_available():
        debug_print("⚠ CUDA not available. FP8 requires GPU support.", "WARNING")
        debug_print("Will use CPU-compatible quantization methods", "INFO")
        return False
    
    # Check compute capability
    try:
        device_count = torch.cuda.device_count()
        debug_print(f"Checking {device_count} CUDA devices...", "DEBUG")
        
        fp8_compatible = False
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            compute_capability = major + minor / 10
            device_name = torch.cuda.get_device_name(i)
            
            debug_print(f"GPU {i} ({device_name}): Compute capability {compute_capability}", "INFO")
            
            if compute_capability < 8.0:
                debug_print(f"⚠ GPU {i} compute capability {compute_capability} < 8.0. FP8 not supported.", "WARNING")
            elif compute_capability < 8.9:
                debug_print(f"⚠ GPU {i} compute capability {compute_capability} < 8.9. FP8 may not be optimal.", "WARNING")
                fp8_compatible = True
            else:
                debug_print(f"✓ GPU {i} compute capability: {compute_capability} - FP8 compatible!", "INFO")
                fp8_compatible = True
        
        # Check for FP8 specific features
        debug_print("Checking FP8 specific features...", "DEBUG")
        
        # Check for transformer engine
        try:
            import transformer_engine
            debug_print("✓ Transformer Engine available for FP8", "INFO")
        except ImportError:
            debug_print("⚠ Transformer Engine not available (optional)", "WARNING")
        
        # Check for TorchAO
        try:
            import torchao
            debug_print("✓ TorchAO available for quantization", "INFO")
        except ImportError:
            debug_print("⚠ TorchAO not available (optional)", "WARNING")
        
        return fp8_compatible
        
    except Exception as e:
        debug_print(f"✗ Failed to check GPU capability: {e}", "ERROR")
        debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
        return False

# Keep the existing KosmosFP8Quantizer class with minor improvements for error handling
class KosmosFP8Quantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        debug_print(f"Initializing KosmosFP8Quantizer...", "INFO")
        debug_print(f"Model name: {model_name}", "DEBUG")
        debug_print(f"Cache dir: {cache_dir}", "DEBUG")
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.processor = None
        self.model = None
        
        # Import transformers components with fallbacks
        debug_print("Importing transformers components...", "DEBUG")
        self.components = safe_import_transformers()
        if not self.components:
            debug_print("✗ Failed to import required transformers components", "ERROR")
            raise ImportError("Failed to import required transformers components")
        
        debug_print("✓ Transformers components imported successfully", "INFO")
        
        # Check system compatibility
        debug_print("Checking system compatibility...", "DEBUG")
        self.fp8_supported = check_fp8_compatibility()
        debug_print(f"FP8 supported: {self.fp8_supported}", "INFO")
        
    def load_components(self):
        """Load tokenizer and processor with enhanced error handling"""
        debug_print("="*50, "INFO")
        debug_print("LOADING MODEL COMPONENTS", "INFO")
        debug_print("="*50, "INFO")
        
        # Load tokenizer with retries
        debug_print("Loading tokenizer...", "INFO")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                debug_print(f"Tokenizer attempt {attempt + 1}/{max_retries}", "DEBUG")
                debug_print(f"Tokenizer class: {self.components['AutoTokenizer']}", "DEBUG")
                debug_print(f"Loading from: {self.model_name}", "DEBUG")
                
                self.tokenizer = self.components['AutoTokenizer'].from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    resume_download=True
                )
                debug_print("✓ Tokenizer loaded successfully", "INFO")
                debug_print(f"Tokenizer type: {type(self.tokenizer)}", "DEBUG")
                try:
                    vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else 'Unknown'
                    debug_print(f"Vocab size: {vocab_size}", "DEBUG")
                except:
                    debug_print("Vocab size: Could not determine", "DEBUG")
                break
                
            except Exception as e:
                debug_print(f"⚠ Tokenizer attempt {attempt + 1} failed: {e}", "WARNING")
                if attempt == max_retries - 1:
                    debug_print(f"✗ Failed to load tokenizer after {max_retries} attempts", "ERROR")
                    debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
                    raise
                else:
                    debug_print(f"Retrying in 2 seconds...", "INFO")
                    time.sleep(2)
        
        # Load processor with fallbacks
        if self.components['AutoProcessor']:
            debug_print("Loading processor...", "INFO")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    debug_print(f"Processor attempt {attempt + 1}/{max_retries}", "DEBUG")
                    debug_print(f"Processor class: {self.components['AutoProcessor']}", "DEBUG")
                    
                    self.processor = self.components['AutoProcessor'].from_pretrained(
                        self.model_name, 
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        resume_download=True
                    )
                    debug_print("✓ Processor loaded successfully", "INFO")
                    debug_print(f"Processor type: {type(self.processor)}", "DEBUG")
                    break
                    
                except Exception as e:
                    debug_print(f"⚠ Processor attempt {attempt + 1} failed: {e}", "WARNING")
                    if attempt == max_retries - 1:
                        debug_print(f"⚠ Failed to load processor after {max_retries} attempts, continuing without it", "WARNING")
                        self.processor = None
                        break
                    else:
                        debug_print(f"Retrying in 2 seconds...", "INFO")
                        time.sleep(2)
        else:
            debug_print("AutoProcessor not available, using tokenizer only", "INFO")
            
        debug_print("✓ Component loading completed", "INFO")

    def method_1_bitsandbytes_8bit(self, save_path="./kosmos2.5-8bit"):
        """8-bit quantization with comprehensive debugging"""
        debug_print("="*50, "INFO")
        debug_print("STARTING 8-BIT QUANTIZATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            import torch
            debug_print(f"Using PyTorch version: {torch.__version__}", "DEBUG")
            
            if self.components['BitsAndBytesConfig']:
                debug_print("BitsAndBytesConfig available, configuring 8-bit quantization...", "INFO")
                
                from transformers import BitsAndBytesConfig
                
                # Configure 8-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0,
                )
                debug_print("✓ BitsAndBytesConfig created", "DEBUG")
                debug_print(f"Config: load_in_8bit=True, threshold=6.0", "DEBUG")
                
                # Load model with 8-bit quantization
                debug_print("Loading model with 8-bit quantization...", "INFO")
                debug_print(f"Model class: {self.components['ModelClass']}", "DEBUG")
                
                start_time = time.time()
                self.model = self.components['ModelClass'].from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    cache_dir=self.cache_dir,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                load_time = time.time() - start_time
                debug_print(f"✓ Model loaded in {load_time:.2f} seconds", "INFO")
                
            else:
                debug_print("BitsAndBytesConfig not available, loading in FP16...", "WARNING")
                
                start_time = time.time()
                self.model = self.components['ModelClass'].from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                load_time = time.time() - start_time
                debug_print(f"✓ Model loaded in FP16 in {load_time:.2f} seconds", "INFO")
            
            # Print model info
            self._print_model_info()
            
            # Save model
            debug_print(f"Saving model to: {save_path}", "INFO")
            self._save_model(save_path)
            debug_print(f"✓ 8-bit quantized model saved to {save_path}", "INFO")
            
            return self.model
            
        except Exception as e:
            debug_print(f"✗ 8-bit quantization failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None

    def method_2_torch_native_fp16(self, save_path="./kosmos2.5-fp16"):
        """FP16 quantization with comprehensive debugging"""
        debug_print("="*50, "INFO")
        debug_print("STARTING FP16 QUANTIZATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            import torch
            debug_print(f"Using PyTorch version: {torch.__version__}", "DEBUG")
            
            # Load model in FP16
            debug_print("Loading model in FP16...", "INFO")
            debug_print(f"Model class: {self.components['ModelClass']}", "DEBUG")
            
            start_time = time.time()
            self.model = self.components['ModelClass'].from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            load_time = time.time() - start_time
            debug_print(f"✓ Model loaded in {load_time:.2f} seconds", "INFO")
            
            # Apply additional optimizations
            debug_print("Applying optimizations...", "INFO")
            if hasattr(self.model, 'eval'):
                self.model.eval()
                debug_print("✓ Model set to eval mode", "DEBUG")
            
            # Try torch.compile if available
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                debug_print("Attempting torch.compile optimization...", "INFO")
                try:
                    compile_start = time.time()
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    compile_time = time.time() - compile_start
                    debug_print(f"✓ torch.compile applied in {compile_time:.2f} seconds", "INFO")
                except Exception as e:
                    debug_print(f"⚠ torch.compile failed: {e}", "WARNING")
            else:
                debug_print("torch.compile not available or CUDA not available", "DEBUG")
            
            # Print model info
            self._print_model_info()
            
            # Save model
            debug_print(f"Saving model to: {save_path}", "INFO")
            self._save_model(save_path)
            debug_print(f"✓ FP16 optimized model saved to {save_path}", "INFO")
            
            return self.model
            
        except Exception as e:
            debug_print(f"✗ FP16 optimization failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None

    def method_3_manual_fp8_simulation(self, save_path="./kosmos2.5-fp8-sim"):
        """FP8 simulation with comprehensive debugging"""
        debug_print("="*50, "INFO")
        debug_print("STARTING FP8 SIMULATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            import torch
            import torch.nn as nn
            debug_print(f"Using PyTorch version: {torch.__version__}", "DEBUG")
            
            # Load model in FP32 first
            debug_print("Loading model in FP32 for processing...", "INFO")
            debug_print(f"Model class: {self.components['ModelClass']}", "DEBUG")
            
            start_time = time.time()
            self.model = self.components['ModelClass'].from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",  # Load on CPU for processing
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            load_time = time.time() - start_time
            debug_print(f"✓ Model loaded in FP32 in {load_time:.2f} seconds", "INFO")
            
            # Print model info before quantization
            self._print_model_info()
            
            # Apply FP8 simulation to linear layers
            debug_print("Applying FP8 simulation...", "INFO")
            quant_start = time.time()
            self._apply_fp8_simulation(self.model)
            quant_time = time.time() - quant_start
            debug_print(f"✓ FP8 simulation completed in {quant_time:.2f} seconds", "INFO")
            
            # Convert to FP16 and move to GPU
            debug_print("Converting to FP16 and moving to GPU...", "INFO")
            move_start = time.time()
            if torch.cuda.is_available():
                self.model = self.model.half().cuda()
                debug_print("✓ Model moved to GPU in FP16", "INFO")
            else:
                self.model = self.model.half()
                debug_print("✓ Model converted to FP16 (CPU)", "INFO")
            move_time = time.time() - move_start
            debug_print(f"✓ Conversion completed in {move_time:.2f} seconds", "INFO")
            
            # Print model info after quantization
            self._print_model_info()
            
            # Save model
            debug_print(f"Saving model to: {save_path}", "INFO")
            self._save_model(save_path)
            debug_print(f"✓ FP8 simulated model saved to {save_path}", "INFO")
            
            return self.model
            
        except Exception as e:
            debug_print(f"✗ FP8 simulation failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    def _apply_fp8_simulation(self, model):
        """Apply FP8 simulation with detailed debugging"""
        import torch
        import torch.nn as nn
        
        debug_print("Scanning model for linear layers...", "DEBUG")
        
        total_params = 0
        quantized_params = 0
        layer_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_count += 1
                debug_print(f"Processing layer {layer_count}: {name}", "DEBUG")
                
                # Simulate FP8 quantization by reducing precision
                weight = module.weight.data.clone()
                original_shape = weight.shape
                debug_print(f"  Layer shape: {original_shape}", "DEBUG")
                
                # Simple FP8 simulation: reduce mantissa precision
                scale = weight.abs().max()
                if scale > 0:
                    # Quantize to ~8 effective bits
                    levels = 256  # 2^8
                    quantized = torch.round(weight / scale * levels) * scale / levels
                    module.weight.data = quantized
                    
                    quantized_params += weight.numel()
                    debug_print(f"  ✓ Quantized {weight.numel()} parameters", "DEBUG")
                else:
                    debug_print(f"  ⚠ Skipped (zero scale)", "DEBUG")
                
                total_params += weight.numel()
        
        debug_print(f"✓ Processed {layer_count} linear layers", "INFO")
        debug_print(f"✓ Quantized {quantized_params}/{total_params} parameters ({quantized_params/total_params*100:.1f}%)", "INFO")
    
    def _print_model_info(self):
        """Print detailed model information"""
        if self.model is None:
            debug_print("No model to analyze", "WARNING")
            return
            
        debug_print("Model Information:", "INFO")
        try:
            # Model type
            debug_print(f"  Model type: {type(self.model).__name__}", "INFO")
            
            # Parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            debug_print(f"  Total parameters: {total_params:,}", "INFO")
            debug_print(f"  Trainable parameters: {trainable_params:,}", "INFO")
            
            # Memory usage
            if hasattr(self.model, 'get_memory_footprint'):
                memory_mb = self.model.get_memory_footprint() / 1024**2
                debug_print(f"  Model memory footprint: {memory_mb:.1f} MB", "INFO")
            
            # Device info
            try:
                if next(self.model.parameters()).is_cuda:
                    device = next(self.model.parameters()).device
                    debug_print(f"  Model device: {device}", "INFO")
                else:
                    debug_print(f"  Model device: CPU", "INFO")
            except:
                debug_print(f"  Model device: Could not determine", "DEBUG")
                
            # Data type
            try:
                dtype = next(self.model.parameters()).dtype
                debug_print(f"  Model dtype: {dtype}", "INFO")
            except:
                debug_print(f"  Model dtype: Could not determine", "DEBUG")
            
        except Exception as e:
            debug_print(f"Failed to get model info: {e}", "WARNING")
    
    def _save_model(self, save_path):
        """Save model with detailed debugging"""
        debug_print(f"Creating output directory: {save_path}", "DEBUG")
        try:
            os.makedirs(save_path, exist_ok=True)
            debug_print(f"✓ Directory created/verified: {save_path}", "DEBUG")
        except Exception as e:
            debug_print(f"✗ Failed to create directory: {e}", "ERROR")
            raise
        
        # Save model
        if self.model:
            debug_print("Saving model weights...", "DEBUG")
            try:
                save_start = time.time()
                self.model.save_pretrained(save_path, safe_serialization=True)
                save_time = time.time() - save_start
                debug_print(f"✓ Model saved in {save_time:.2f} seconds", "INFO")
                
                # Check saved files
                model_files = [f for f in os.listdir(save_path) if f.endswith(('.bin', '.safetensors', '.json'))]
                debug_print(f"✓ Saved files: {model_files}", "DEBUG")
                
            except Exception as e:
                debug_print(f"✗ Failed to save model: {e}", "ERROR")
                raise
        
        # Save tokenizer
        if self.tokenizer:
            debug_print("Saving tokenizer...", "DEBUG")
            try:
                self.tokenizer.save_pretrained(save_path)
                debug_print("✓ Tokenizer saved", "DEBUG")
            except Exception as e:
                debug_print(f"⚠ Failed to save tokenizer: {e}", "WARNING")
        
        # Save processor
        if self.processor:
            debug_print("Saving processor...", "DEBUG")
            try:
                self.processor.save_pretrained(save_path)
                debug_print("✓ Processor saved", "DEBUG")
            except Exception as e:
                debug_print(f"⚠ Failed to save processor: {e}", "WARNING")
                
        debug_print(f"✓ All components saved to {save_path}", "INFO")

    def benchmark_model(self, model, iterations=10):
        """Benchmark with detailed debugging"""
        debug_print("="*50, "INFO")
        debug_print("STARTING BENCHMARK", "INFO")
        debug_print("="*50, "INFO")
        
        if not self.tokenizer:
            debug_print("Loading components for benchmark...", "INFO")
            self.load_components()
        
        try:
            import torch
            
            # Prepare test input
            test_prompt = "What do you see in this image?"
            debug_print(f"Test prompt: '{test_prompt}'", "DEBUG")
            
            debug_print("Tokenizing input...", "DEBUG")
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            debug_print(f"Input shape: {inputs['input_ids'].shape}", "DEBUG")
            
            if torch.cuda.is_available():
                debug_print("Moving inputs to GPU...", "DEBUG")
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Warm up
            debug_print(f"Warming up with 3 iterations...", "INFO")
            with torch.no_grad():
                for i in range(3):
                    debug_print(f"  Warmup {i+1}/3", "DEBUG")
                    _ = model.generate(**inputs, max_length=50, do_sample=False)
            debug_print("✓ Warmup completed", "INFO")
            
            # Benchmark
            debug_print(f"Running benchmark with {iterations} iterations...", "INFO")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                for i in range(iterations):
                    debug_print(f"  Iteration {i+1}/{iterations}", "DEBUG")
                    outputs = model.generate(**inputs, max_length=50, do_sample=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            avg_time = total_time / iterations
            tokens_per_sec = (50 * iterations) / total_time
            
            debug_print(f"✓ Benchmark completed in {total_time:.2f} seconds", "INFO")
            
            # Memory usage
            memory_mb = 0
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                debug_print(f"GPU memory allocated: {memory_mb:.0f} MB", "INFO")
            
            results = {
                "avg_time": avg_time,
                "tokens_per_sec": tokens_per_sec,
                "memory_mb": memory_mb,
                "total_time": total_time,
                "iterations": iterations
            }
            
            debug_print("BENCHMARK RESULTS:", "INFO")
            debug_print(f"  Average inference time: {avg_time:.3f}s", "INFO")
            debug_print(f"  Tokens per second: {tokens_per_sec:.1f}", "INFO")
            debug_print(f"  Total time: {total_time:.2f}s", "INFO")
            if memory_mb > 0:
                debug_print(f"  GPU Memory usage: {memory_mb:.0f} MB", "INFO")
            
            return results
            
        except Exception as e:
            debug_print(f"✗ Benchmarking failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None

def main():
    debug_print("="*80, "INFO")
    debug_print("KOSMOS-2.5 FP8/QUANTIZATION TOOL WITH AUTO-FIX", "INFO")
    debug_print("="*80, "INFO")
    
    parser = argparse.ArgumentParser(description='Enhanced FP8/Quantization for Kosmos-2.5 with automatic dependency fixing')
    parser.add_argument('--method', 
                       choices=['8bit', 'fp16', 'fp8_sim'], 
                       required=True,
                       help='Quantization method')
    parser.add_argument('--model_name', default='microsoft/kosmos-2.5')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--check_deps', action='store_true', help='Only check dependencies')
    parser.add_argument('--auto_fix', action='store_true', default=True, help='Automatically fix dependency issues')
    parser.add_argument('--no_auto_fix', action='store_true', help='Disable automatic fixing')
    parser.add_argument('--debug_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set debug level')
    
    args = parser.parse_args()
    
    # Handle auto-fix settings
    if args.no_auto_fix:
        args.auto_fix = False
    
    # Set debug level
    if args.debug_level:
        logging.getLogger().setLevel(getattr(logging, args.debug_level))
        debug_print(f"Debug level set to: {args.debug_level}", "INFO")
    
    debug_print(f"Arguments: {vars(args)}", "DEBUG")
    
    # Check dependencies with optional auto-fix
    debug_print("Starting enhanced dependency check...", "INFO")
    if not check_and_install_dependencies(auto_fix=args.auto_fix):
        debug_print("✗ Dependency check failed. Please fix the issues manually.", "ERROR")
        if not args.auto_fix:
            debug_print("Try running with --auto_fix to automatically resolve issues", "INFO")
        return 1
    
    if args.check_deps:
        debug_print("✓ Dependency check completed successfully!", "INFO")
        debug_print("All components are ready for quantization", "INFO")
        return 0
    
    # Check GPU compatibility
    debug_print("Checking GPU compatibility...", "INFO")
    if not check_fp8_compatibility():
        debug_print("⚠ System may not be optimal for advanced quantization. Using fallback methods...", "WARNING")
    
    try:
        # Initialize quantizer
        debug_print("Initializing quantizer...", "INFO")
        quantizer = KosmosFP8Quantizer(args.model_name, args.cache_dir)
        
        # Load components
        debug_print("Loading model components...", "INFO")
        quantizer.load_components()
        
        # Method dispatch
        methods = {
            '8bit': quantizer.method_1_bitsandbytes_8bit,
            'fp16': quantizer.method_2_torch_native_fp16,
            'fp8_sim': quantizer.method_3_manual_fp8_simulation,
        }
        
        debug_print(f"Selected method: {args.method}", "INFO")
        debug_print(f"Available methods: {list(methods.keys())}", "DEBUG")
        
        # Execute quantization
        debug_print(f"Starting {args.method} quantization...", "INFO")
        overall_start = time.time()
        
        model = methods[args.method](args.save_path)
        if model is None:
            debug_print(f"✗ Failed to quantize with {args.method} method", "ERROR")
            return 1
            
        quantization_time = time.time() - overall_start
        debug_print(f"✓ Quantization completed in {quantization_time:.2f} seconds", "INFO")
        
        # Run benchmark
        benchmark_results = None
        if args.benchmark:
            debug_print("Starting benchmark...", "INFO")
            benchmark_results = quantizer.benchmark_model(model)
            
        # Print final summary
        debug_print("="*80, "INFO")
        debug_print("FINAL SUMMARY", "INFO")
        debug_print("="*80, "INFO")
        debug_print(f"Method: {args.method.upper()}", "INFO")
        debug_print(f"Model: {args.model_name}", "INFO")
        debug_print(f"Save path: {args.save_path}", "INFO")
        debug_print(f"Quantization time: {quantization_time:.2f}s", "INFO")
        debug_print(f"Auto-fix enabled: {args.auto_fix}", "INFO")
        
        if args.benchmark and benchmark_results:
            memory_reduction = "~50% (FP32→FP16)" if args.method == 'fp16' else "~62.5% (FP32→8bit)"
            debug_print(f"Estimated memory reduction: {memory_reduction}", "INFO")
            debug_print(f"Benchmark avg time: {benchmark_results['avg_time']:.3f}s", "INFO")
            debug_print(f"Tokens per second: {benchmark_results['tokens_per_sec']:.1f}", "INFO")
            
        debug_print("="*80, "INFO")
        debug_print("✓ QUANTIZATION COMPLETED SUCCESSFULLY!", "INFO")
        debug_print("="*80, "INFO")
        
        return 0
        
    except KeyboardInterrupt:
        debug_print("⚠ Process interrupted by user", "WARNING")
        return 1
    except Exception as e:
        debug_print(f"✗ Quantization failed: {e}", "ERROR")
        debug_print(f"Full error details: {traceback.format_exc()}", "DEBUG")
        debug_print("Try running with --check_deps to verify dependencies", "INFO")
        return 1

if __name__ == "__main__":
    sys.exit(main())
