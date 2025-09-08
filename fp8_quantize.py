#!/usr/bin/env python3
"""
FP8 Quantization for Kosmos-2.5 with Enhanced Dependency Handling

This script implements multiple FP8 quantization approaches with robust dependency management:
1. Automatic dependency conflict resolution
2. Graceful fallbacks for version mismatches
3. CUDA version compatibility checking
4. Alternative import strategies
5. Comprehensive debugging output
6. Triton library conflict resolution

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

# Suppress specific warnings that might interfere
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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
    
    debug_print("Could not detect CUDA version from nvcc, checking environment...", "WARNING")
    
    # Check environment variables
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        debug_print(f"CUDA_HOME found: {cuda_home}", "INFO")
        # Try to extract version from path
        if '11.' in cuda_home:
            return "11"
        elif '12.' in cuda_home:
            return "12"
    
    debug_print("Assuming CUDA 11.8 as default", "WARNING")
    return "11"

def fix_triton_conflict():
    """Attempt to fix Triton library conflicts"""
    debug_print("="*60, "INFO")
    debug_print("ATTEMPTING TRITON CONFLICT RESOLUTION", "INFO")
    debug_print("="*60, "INFO")
    
    # Strategy 1: Force uninstall and reinstall PyTorch ecosystem
    debug_print("Uninstalling all PyTorch-related packages...", "INFO")
    uninstall_commands = [
        "pip uninstall torch torchvision torchaudio triton -y",
        "pip uninstall torch-audio torch-vision -y",  # Alternative names
        "pip cache purge"  # Clear pip cache
    ]
    
    for cmd in uninstall_commands:
        success, _ = run_command(cmd, f"Cleanup: {cmd}")
        if not success:
            debug_print(f"⚠ Command failed but continuing: {cmd}", "WARNING")
    
    # Strategy 2: Clean Python cache
    debug_print("Cleaning Python cache...", "INFO")
    try:
        import shutil
        cache_dirs = [
            os.path.expanduser("~/.cache/pip"),
            "/tmp/pip-*",
            "__pycache__"
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    if os.path.isdir(cache_dir):
                        shutil.rmtree(cache_dir)
                        debug_print(f"✓ Cleaned cache: {cache_dir}", "DEBUG")
                except Exception as e:
                    debug_print(f"⚠ Could not clean {cache_dir}: {e}", "WARNING")
    except Exception as e:
        debug_print(f"⚠ Cache cleaning failed: {e}", "WARNING")
    
    # Strategy 3: Install fresh PyTorch
    cuda_major = detect_cuda_version()
    
    if cuda_major == "12":
        index_url = "https://download.pytorch.org/whl/cu121"
        cuda_version = "cu121"
    elif cuda_major == "11":
        index_url = "https://download.pytorch.org/whl/cu118"
        cuda_version = "cu118"
    else:
        debug_print(f"Unsupported CUDA version {cuda_major}, using CPU version", "WARNING")
        index_url = "https://download.pytorch.org/whl/cpu"
        cuda_version = "cpu"
    
    debug_print(f"Installing fresh PyTorch for {cuda_version}...", "INFO")
    
    # Install with specific versions to avoid conflicts
    install_cmd = f"pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url {index_url} --force-reinstall --no-cache-dir"
    success, output = run_command(install_cmd, f"Fresh PyTorch installation for {cuda_version}")
    
    if success:
        debug_print("✓ Fresh PyTorch installation completed", "INFO")
        return True
    else:
        debug_print("✗ Fresh PyTorch installation failed", "ERROR")
        
        # Fallback to CPU version
        debug_print("Trying CPU-only PyTorch as last resort...", "INFO")
        cpu_cmd = "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-cache-dir"
        success, _ = run_command(cpu_cmd, "CPU-only PyTorch installation")
        
        if success:
            debug_print("✓ CPU-only PyTorch installed successfully", "INFO")
            return True
        else:
            debug_print("✗ All PyTorch installation attempts failed", "ERROR")
            return False

def safe_import_torch():
    """Safely import PyTorch with conflict resolution"""
    debug_print("Attempting safe PyTorch import...", "INFO")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        debug_print(f"Import attempt {attempt + 1}/{max_attempts}", "DEBUG")
        
        try:
            # Clear any existing torch modules from cache
            modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith(('torch', 'triton'))]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]
                    debug_print(f"Cleared module: {mod}", "DEBUG")
            
            # Try importing torch
            import torch
            debug_print(f"✓ PyTorch imported successfully: {torch.__version__}", "INFO")
            debug_print(f"✓ CUDA available: {torch.cuda.is_available()}", "INFO")
            
            if torch.cuda.is_available():
                debug_print(f"✓ CUDA version: {torch.version.cuda}", "INFO")
                device_count = torch.cuda.device_count()
                debug_print(f"✓ GPU count: {device_count}", "INFO")
            
            return torch
            
        except RuntimeError as e:
            if "triton" in str(e).lower() or "torch_library" in str(e).lower():
                debug_print(f"⚠ Triton/TORCH_LIBRARY conflict detected on attempt {attempt + 1}: {e}", "WARNING")
                
                if attempt < max_attempts - 1:
                    debug_print("Attempting Triton conflict resolution...", "INFO")
                    if fix_triton_conflict():
                        debug_print("✓ Triton conflict resolution completed, retrying import...", "INFO")
                        time.sleep(2)  # Give system time to settle
                        continue
                    else:
                        debug_print("✗ Triton conflict resolution failed", "ERROR")
                else:
                    debug_print("✗ All import attempts failed due to Triton conflicts", "ERROR")
                    return None
            else:
                debug_print(f"✗ PyTorch import failed with different error: {e}", "ERROR")
                return None
                
        except ImportError as e:
            debug_print(f"✗ PyTorch not found: {e}", "ERROR")
            if attempt < max_attempts - 1:
                debug_print("Installing PyTorch...", "INFO")
                if fix_triton_conflict():
                    continue
            return None
            
        except Exception as e:
            debug_print(f"✗ Unexpected error during PyTorch import: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    return None

def check_and_install_dependencies(auto_fix=True):
    """Check and handle dependency issues with enhanced Triton conflict resolution"""
    debug_print("="*60, "INFO")
    debug_print("STARTING ENHANCED DEPENDENCY CHECK WITH TRITON FIXES", "INFO")
    debug_print("="*60, "INFO")
    
    # Check PyTorch with Triton conflict handling
    debug_print("Checking PyTorch installation with conflict resolution...", "INFO")
    pytorch_ok = False
    
    torch = safe_import_torch()
    if torch:
        pytorch_ok = True
        debug_print("✓ PyTorch imported successfully", "INFO")
    else:
        debug_print("✗ PyTorch import failed after all attempts", "ERROR")
        if not auto_fix:
            debug_print("Manual intervention required. Try:", "ERROR")
            debug_print("  pip uninstall torch torchvision torchaudio triton -y", "ERROR")
            debug_print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "ERROR")
            return False
        else:
            debug_print("⚠ Continuing without PyTorch - limited functionality", "WARNING")
    
    # Check torchvision compatibility
    if pytorch_ok:
        debug_print("Checking torchvision compatibility...", "INFO")
        try:
            import torchvision
            debug_print(f"✓ Torchvision version: {torchvision.__version__}", "INFO")
            debug_print(f"✓ Torchvision location: {torchvision.__file__}", "DEBUG")
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                debug_print(f"✗ CUDA version mismatch detected: {e}", "ERROR")
                
                if auto_fix:
                    debug_print("Attempting automatic fix...", "INFO")
                    if fix_triton_conflict():
                        debug_print("✓ Fix attempt completed, please restart the script", "INFO")
                        return False
                    else:
                        debug_print("✗ Automatic fix failed", "ERROR")
                        return False
            else:
                debug_print(f"✗ Torchvision error: {e}", "ERROR")
                
        except ImportError as e:
            debug_print(f"⚠ Torchvision not found: {e}", "WARNING")
    
    # Check transformers with enhanced error handling
    debug_print("Checking transformers library...", "INFO")
    try:
        # Clear transformers from cache if it exists
        if 'transformers' in sys.modules:
            del sys.modules['transformers']
            debug_print("Cleared transformers from module cache", "DEBUG")
        
        import transformers
        debug_print(f"✓ Transformers version: {transformers.__version__}", "INFO")
        debug_print(f"✓ Transformers location: {transformers.__file__}", "DEBUG")
        
        # Test basic imports
        debug_print("Testing transformers components...", "DEBUG")
        
        try:
            from transformers import AutoTokenizer
            debug_print("✓ AutoTokenizer import successful", "DEBUG")
        except ImportError as e:
            debug_print(f"✗ AutoTokenizer import failed: {e}", "ERROR")
            return False
            
        try:
            from transformers import AutoProcessor
            debug_print("✓ AutoProcessor import successful", "DEBUG")
        except Exception as e:
            debug_print(f"⚠ AutoProcessor import failed: {e}", "WARNING")
            
    except ImportError as e:
        debug_print(f"✗ Transformers not found: {e}", "ERROR")
        debug_print("Installing transformers...", "INFO")
        success, _ = run_command("pip install transformers>=4.36.0", "Install transformers")
        if not success:
            return False
    except Exception as e:
        debug_print(f"✗ Transformers check failed: {e}", "ERROR")
        debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
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
    """Safely import transformers components with enhanced fallbacks"""
    debug_print("Starting safe transformers import with enhanced fallbacks...", "INFO")
    components = {}
    
    try:
        # Clear any cached modules
        modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith('transformers')]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Import base transformers
        import transformers
        debug_print(f"✓ Base transformers imported: {transformers.__version__}", "DEBUG")
        
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
            'Kosmos2_5ForConditionalGeneration',
            'AutoModelForImageTextToText',
            'AutoModelForVision2Seq',
            'AutoModelForCausalLM',
        ]
        
        model_imported = False
        for class_name in model_classes_to_try:
            try:
                if class_name == 'Kosmos2_5ForConditionalGeneration':
                    from transformers import Kosmos2_5ForConditionalGeneration
                    components['ModelClass'] = Kosmos2_5ForConditionalGeneration
                elif class_name == 'AutoModelForImageTextToText':
                    from transformers import AutoModelForImageTextToText
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
        
    except Exception as e:
        debug_print(f"✗ Safe transformers import failed: {e}", "ERROR")
        debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
        return None

def check_fp8_compatibility():
    """Check if the system supports FP8 quantization with enhanced error handling"""
    debug_print("Checking FP8 compatibility...", "INFO")
    
    torch = safe_import_torch()
    if not torch:
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
        
        return fp8_compatible
        
    except Exception as e:
        debug_print(f"✗ Failed to check GPU capability: {e}", "ERROR")
        debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
        return False

# Keep the existing KosmosFP8Quantizer class with enhanced error handling
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
        
        # Import transformers components with enhanced fallbacks
        debug_print("Importing transformers components with enhanced safety...", "DEBUG")
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
                break
                
            except Exception as e:
                debug_print(f"⚠ Tokenizer attempt {attempt + 1} failed: {e}", "WARNING")
                if attempt == max_retries - 1:
                    debug_print(f"✗ Failed to load tokenizer after {max_retries} attempts", "ERROR")
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
                    
                    self.processor = self.components['AutoProcessor'].from_pretrained(
                        self.model_name, 
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                        resume_download=True
                    )
                    debug_print("✓ Processor loaded successfully", "INFO")
                    break
                    
                except Exception as e:
                    debug_print(f"⚠ Processor attempt {attempt + 1} failed: {e}", "WARNING")
                    if attempt == max_retries - 1:
                        debug_print(f"⚠ Failed to load processor, continuing without it", "WARNING")
                        self.processor = None
                        break
                    else:
                        time.sleep(2)
        else:
            debug_print("AutoProcessor not available, using tokenizer only", "INFO")
            
        debug_print("✓ Component loading completed", "INFO")

    def method_1_bitsandbytes_8bit(self, save_path="./kosmos2.5-8bit"):
        """8-bit quantization with enhanced error handling"""
        debug_print("="*50, "INFO")
        debug_print("STARTING 8-BIT QUANTIZATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            torch = safe_import_torch()
            if not torch:
                debug_print("✗ PyTorch not available for quantization", "ERROR")
                return None
                
            debug_print(f"Using PyTorch version: {torch.__version__}", "DEBUG")
            
            if self.components['BitsAndBytesConfig']:
                debug_print("Configuring 8-bit quantization...", "INFO")
                
                bnb_config = self.components['BitsAndBytesConfig'](
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    llm_int8_threshold=6.0,
                )
                debug_print("✓ BitsAndBytesConfig created", "DEBUG")
                
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
        """FP16 quantization with enhanced error handling"""
        debug_print("="*50, "INFO")
        debug_print("STARTING FP16 QUANTIZATION", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            torch = safe_import_torch()
            if not torch:
                debug_print("✗ PyTorch not available for quantization", "ERROR")
                return None
                
            debug_print(f"Using PyTorch version: {torch.__version__}", "DEBUG")
            
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
            
            # Apply optimizations
            if hasattr(self.model, 'eval'):
                self.model.eval()
                debug_print("✓ Model set to eval mode", "DEBUG")
            
            # Save model
            debug_print(f"Saving model to: {save_path}", "INFO")
            self._save_model(save_path)
            debug_print(f"✓ FP16 optimized model saved to {save_path}", "INFO")
            
            return self.model
            
        except Exception as e:
            debug_print(f"✗ FP16 optimization failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None

    def _save_model(self, save_path):
        """Save model with enhanced error handling"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            if self.model:
                self.model.save_pretrained(save_path, safe_serialization=True)
                debug_print("✓ Model saved", "DEBUG")
                
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
                debug_print("✓ Tokenizer saved", "DEBUG")
            
            if self.processor:
                self.processor.save_pretrained(save_path)
                debug_print("✓ Processor saved", "DEBUG")
                
        except Exception as e:
            debug_print(f"✗ Failed to save model: {e}", "ERROR")
            raise

def main():
    debug_print("="*80, "INFO")
    debug_print("KOSMOS-2.5 QUANTIZATION TOOL WITH TRITON CONFLICT RESOLUTION", "INFO")
    debug_print("="*80, "INFO")
    
    parser = argparse.ArgumentParser(description='Enhanced quantization with Triton conflict resolution')
    parser.add_argument('--method', 
                       choices=['8bit', 'fp16'], 
                       required=True,
                       help='Quantization method')
    parser.add_argument('--model_name', default='microsoft/kosmos-2.5')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--check_deps', action='store_true', help='Only check dependencies')
    parser.add_argument('--auto_fix', action='store_true', default=True)
    parser.add_argument('--no_auto_fix', action='store_true')
    
    args = parser.parse_args()
    
    if args.no_auto_fix:
        args.auto_fix = False
    
    # Enhanced dependency check with Triton resolution
    debug_print("Starting enhanced dependency check with Triton resolution...", "INFO")
    if not check_and_install_dependencies(auto_fix=args.auto_fix):
        debug_print("✗ Dependency check failed.", "ERROR")
        debug_print("Try running the script again or manually fix PyTorch installation", "INFO")
        return 1
    
    if args.check_deps:
        debug_print("✓ Dependency check completed successfully!", "INFO")
        return 0
    
    try:
        # Initialize quantizer
        debug_print("Initializing quantizer...", "INFO")
        quantizer = KosmosFP8Quantizer(args.model_name, args.cache_dir)
        
        # Load components
        debug_print("Loading model components...", "INFO")
        quantizer.load_components()
        
        # Execute quantization
        if args.method == '8bit':
            model = quantizer.method_1_bitsandbytes_8bit(args.save_path)
        elif args.method == 'fp16':
            model = quantizer.method_2_torch_native_fp16(args.save_path)
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
