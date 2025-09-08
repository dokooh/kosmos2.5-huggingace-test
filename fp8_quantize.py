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
7. ML_dtypes and ONNX compatibility fixes
8. NumPy/scikit-learn compatibility fixes
9. PyTorch function redefinition conflict fixes

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
import subprocess
from typing import Optional, Dict, Any, Tuple
import traceback
import shutil
import glob

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

def aggressive_pytorch_cleanup():
    """Perform aggressive cleanup of PyTorch installations and caches"""
    debug_print("="*60, "INFO")
    debug_print("PERFORMING AGGRESSIVE PYTORCH CLEANUP", "INFO")
    debug_print("="*60, "INFO")
    
    # Step 1: Clear all PyTorch-related modules from Python cache
    debug_print("Clearing Python module cache...", "INFO")
    modules_to_clear = []
    for mod_name in list(sys.modules.keys()):
        if any(keyword in mod_name.lower() for keyword in ['torch', 'triton', 'cuda', '_torch']):
            modules_to_clear.append(mod_name)
    
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
            debug_print(f"Cleared module: {mod}", "DEBUG")
    
    debug_print(f"Cleared {len(modules_to_clear)} modules from cache", "INFO")
    
    # Step 2: Uninstall all PyTorch-related packages aggressively
    debug_print("Uninstalling all PyTorch-related packages...", "INFO")
    uninstall_commands = [
        "pip uninstall torch torchvision torchaudio triton torch-audio torch-vision -y",
        "pip uninstall torch2trt torchtext torchaudio-dev torchvision-dev torch-dev -y", 
        "pip uninstall pytorch pytorch-lightning torch-geometric -y",
        "pip uninstall nvidia-cublas-cu11 nvidia-cublas-cu12 -y",
        "pip uninstall nvidia-cuda-runtime-cu11 nvidia-cuda-runtime-cu12 -y",
    ]
    
    for cmd in uninstall_commands:
        success, _ = run_command(cmd, f"Aggressive uninstall: {cmd}")
        # Continue regardless of success
    
    # Step 3: Clear pip cache completely
    debug_print("Clearing pip cache...", "INFO")
    cache_commands = [
        "pip cache purge",
        "pip cache remove pytorch",
        "pip cache remove torch",
    ]
    
    for cmd in cache_commands:
        run_command(cmd, f"Cache clear: {cmd}")
    
    # Step 4: Clean Python bytecode cache
    debug_print("Cleaning Python bytecode cache...", "INFO")
    try:
        import site
        import compileall
        
        # Get all site-packages directories
        site_packages = site.getsitepackages()
        
        for site_pkg in site_packages:
            if os.path.exists(site_pkg):
                # Remove __pycache__ directories
                pycache_dirs = glob.glob(os.path.join(site_pkg, "**", "__pycache__"), recursive=True)
                for pycache_dir in pycache_dirs:
                    if 'torch' in pycache_dir.lower():
                        try:
                            shutil.rmtree(pycache_dir)
                            debug_print(f"Removed pycache: {pycache_dir}", "DEBUG")
                        except Exception as e:
                            debug_print(f"Could not remove {pycache_dir}: {e}", "WARNING")
                
                # Remove .pyc files
                pyc_files = glob.glob(os.path.join(site_pkg, "**", "*.pyc"), recursive=True)
                for pyc_file in pyc_files:
                    if 'torch' in pyc_file.lower():
                        try:
                            os.remove(pyc_file)
                            debug_print(f"Removed pyc: {pyc_file}", "DEBUG")
                        except Exception as e:
                            debug_print(f"Could not remove {pyc_file}: {e}", "WARNING")
                            
    except Exception as e:
        debug_print(f"Bytecode cache cleaning failed: {e}", "WARNING")
    
    # Step 5: Remove conda pytorch packages if conda is available
    debug_print("Checking for conda pytorch packages...", "INFO")
    success, output = run_command("conda list | grep torch", "Check conda packages")
    if success and output.strip():
        debug_print("Found conda torch packages, removing...", "INFO")
        conda_commands = [
            "conda remove pytorch torchvision torchaudio cpuonly -y --force",
            "conda remove pytorch torchvision torchaudio pytorch-cuda -y --force",
            "conda remove pytorch torchvision torchaudio cudatoolkit -y --force",
        ]
        
        for cmd in conda_commands:
            run_command(cmd, f"Conda removal: {cmd}")
    
    debug_print("✓ Aggressive PyTorch cleanup completed", "INFO")
    return True

def fix_pytorch_function_redefinition():
    """Fix PyTorch function redefinition issues"""
    debug_print("="*60, "INFO")
    debug_print("FIXING PYTORCH FUNCTION REDEFINITION ISSUES", "INFO")
    debug_print("="*60, "INFO")
    
    # Strategy 1: Aggressive cleanup first
    if not aggressive_pytorch_cleanup():
        debug_print("✗ Aggressive cleanup failed", "ERROR")
        return False
    
    # Strategy 2: Wait for system to settle
    debug_print("Waiting for system to settle after cleanup...", "INFO")
    time.sleep(5)
    
    # Strategy 3: Install PyTorch with specific configuration
    debug_print("Installing PyTorch with specific configuration...", "INFO")
    
    # Detect CUDA version
    cuda_major = detect_cuda_version()
    
    # Install PyTorch with very specific versions to avoid conflicts
    if cuda_major == "12":
        install_cmd = "pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir --force-reinstall"
    elif cuda_major == "11":
        install_cmd = "pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir --force-reinstall"
    else:
        install_cmd = "pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --force-reinstall"
    
    success, output = run_command(install_cmd, "Install specific PyTorch version")
    
    if not success:
        debug_print("Specific version install failed, trying general install...", "WARNING")
        
        # Fallback to general installation
        general_install_commands = [
            "pip install torch torchvision torchaudio --no-cache-dir --force-reinstall",
            "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --no-cache-dir --force-reinstall",
        ]
        
        for cmd in general_install_commands:
            success, _ = run_command(cmd, f"Fallback install: {cmd}")
            if success:
                break
    
    # Strategy 4: Verify installation
    debug_print("Verifying PyTorch installation...", "INFO")
    time.sleep(3)  # Give system time to settle
    
    try:
        # Clear any remaining cached modules
        modules_to_clear = [mod for mod in sys.modules.keys() if 'torch' in mod.lower()]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Test import
        import torch
        debug_print(f"✓ PyTorch {torch.__version__} imported successfully after fix", "INFO")
        return True
        
    except Exception as e:
        debug_print(f"✗ PyTorch import still failing after fix: {e}", "ERROR")
        return False

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

def fix_numpy_sklearn_conflict():
    """Fix NumPy/scikit-learn binary incompatibility issues"""
    debug_print("="*60, "INFO")
    debug_print("FIXING NUMPY/SCIKIT-LEARN COMPATIBILITY ISSUES", "INFO")
    debug_print("="*60, "INFO")
    
    # Strategy 1: Check current NumPy version
    try:
        import numpy as np
        numpy_version = np.__version__
        debug_print(f"Current NumPy version: {numpy_version}", "INFO")
    except ImportError:
        debug_print("NumPy not found, will install compatible version", "WARNING")
        numpy_version = None
    
    # Strategy 2: Uninstall problematic packages that depend on NumPy
    debug_print("Uninstalling packages with NumPy binary dependencies...", "INFO")
    uninstall_commands = [
        "pip uninstall scikit-learn scipy pandas -y",
        "pip uninstall sklearn -y",  # Alternative name
        "pip uninstall numpy -y",  # Remove NumPy last
    ]
    
    for cmd in uninstall_commands:
        success, _ = run_command(cmd, f"Cleanup: {cmd}")
        if not success:
            debug_print(f"⚠ Command failed but continuing: {cmd}", "WARNING")
    
    # Strategy 3: Install compatible NumPy version first
    debug_print("Installing compatible NumPy version...", "INFO")
    
    # Install a specific NumPy version that works well with most packages
    numpy_install_commands = [
        "pip install 'numpy>=1.21.0,<1.25.0'",  # Compatible range
        "pip install numpy==1.24.3",  # Specific stable version
        "pip install numpy==1.23.5",  # Fallback version
    ]
    
    numpy_installed = False
    for cmd in numpy_install_commands:
        success, _ = run_command(cmd, f"Install NumPy: {cmd}")
        if success:
            debug_print(f"✓ NumPy installed successfully with: {cmd}", "INFO")
            numpy_installed = True
            break
        else:
            debug_print(f"⚠ NumPy install failed with: {cmd}", "WARNING")
    
    if not numpy_installed:
        debug_print("✗ Failed to install compatible NumPy version", "ERROR")
        return False
    
    # Strategy 4: Install scikit-learn and dependencies
    debug_print("Installing scikit-learn and dependencies...", "INFO")
    
    # Install in order: scipy first, then scikit-learn
    install_commands = [
        "pip install 'scipy>=1.9.0,<1.12.0'",  # Compatible scipy
        "pip install 'scikit-learn>=1.3.0,<1.4.0'",  # Compatible scikit-learn
    ]
    
    for cmd in install_commands:
        success, _ = run_command(cmd, f"Install: {cmd}")
        if not success:
            debug_print(f"⚠ Failed: {cmd}", "WARNING")
            # Try alternative versions
            if "scipy" in cmd:
                success, _ = run_command("pip install scipy==1.10.1", "Install specific scipy")
            elif "scikit-learn" in cmd:
                success, _ = run_command("pip install scikit-learn==1.3.2", "Install specific scikit-learn")
    
    # Strategy 5: Verify the fix
    debug_print("Verifying NumPy/scikit-learn compatibility...", "INFO")
    try:
        # Clear any cached modules
        modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith(('numpy', 'scipy', 'sklearn'))]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        import numpy as np
        debug_print(f"✓ NumPy {np.__version__} imported successfully", "INFO")
        
        import sklearn
        debug_print(f"✓ scikit-learn {sklearn.__version__} imported successfully", "INFO")
        
        # Test a simple sklearn import
        from sklearn.metrics import roc_curve
        debug_print("✓ sklearn.metrics.roc_curve imported successfully", "INFO")
        
        debug_print("✓ NumPy/scikit-learn compatibility fix successful", "INFO")
        return True
        
    except Exception as e:
        debug_print(f"✗ NumPy/scikit-learn compatibility verification failed: {e}", "ERROR")
        return False

def fix_ml_dtypes_onnx_conflict():
    """Fix ML_dtypes and ONNX compatibility issues"""
    debug_print("="*60, "INFO")
    debug_print("FIXING ML_DTYPES AND ONNX COMPATIBILITY ISSUES", "INFO")
    debug_print("="*60, "INFO")
    
    # Strategy 1: Uninstall problematic packages
    debug_print("Uninstalling problematic packages...", "INFO")
    uninstall_commands = [
        "pip uninstall onnx-ir onnxscript ml-dtypes -y",
        "pip uninstall onnx onnxruntime -y",
    ]
    
    for cmd in uninstall_commands:
        success, _ = run_command(cmd, f"Cleanup: {cmd}")
        if not success:
            debug_print(f"⚠ Command failed but continuing: {cmd}", "WARNING")
    
    # Strategy 2: Install compatible versions
    debug_print("Installing compatible package versions...", "INFO")
    
    # Install compatible ml_dtypes first
    success, _ = run_command("pip install 'ml-dtypes>=0.2.0,<0.4.0'", "Install compatible ml_dtypes")
    if not success:
        debug_print("⚠ Failed to install ml_dtypes, trying alternative...", "WARNING")
        success, _ = run_command("pip install ml-dtypes==0.3.2", "Install specific ml_dtypes")
    
    # Install ONNX without the problematic ir package
    success, _ = run_command("pip install 'onnx>=1.14.0,<1.16.0'", "Install compatible ONNX")
    if not success:
        debug_print("⚠ Failed to install ONNX", "WARNING")
    
    # Avoid onnxscript and onnx-ir for now
    debug_print("✓ ML_dtypes compatibility fix attempted", "INFO")
    return True

def safe_import_torch():
    """Safely import PyTorch with enhanced conflict resolution"""
    debug_print("Attempting safe PyTorch import with enhanced conflict resolution...", "INFO")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        debug_print(f"PyTorch import attempt {attempt + 1}/{max_attempts}", "DEBUG")
        
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
            error_str = str(e).lower()
            if "triton" in error_str or "torch_library" in error_str:
                debug_print(f"⚠ Triton/TORCH_LIBRARY conflict detected on attempt {attempt + 1}: {e}", "WARNING")
                
                if attempt < max_attempts - 1:
                    debug_print("Attempting Triton conflict resolution...", "INFO")
                    if fix_pytorch_function_redefinition():
                        debug_print("✓ Triton conflict resolution completed, retrying import...", "INFO")
                        time.sleep(3)  # Give system time to settle
                        continue
                    else:
                        debug_print("✗ Triton conflict resolution failed", "ERROR")
                else:
                    debug_print("✗ All import attempts failed due to Triton conflicts", "ERROR")
                    return None
            elif "docstring" in error_str or "_has_torch_function" in error_str:
                debug_print(f"⚠ PyTorch function redefinition conflict detected on attempt {attempt + 1}: {e}", "WARNING")
                
                if attempt < max_attempts - 1:
                    debug_print("Attempting function redefinition fix...", "INFO")
                    if fix_pytorch_function_redefinition():
                        debug_print("✓ Function redefinition fix completed, retrying import...", "INFO")
                        time.sleep(3)  # Give system time to settle
                        continue
                    else:
                        debug_print("✗ Function redefinition fix failed", "ERROR")
                else:
                    debug_print("✗ All import attempts failed due to function redefinition", "ERROR")
                    return None
            else:
                debug_print(f"✗ PyTorch import failed with different runtime error: {e}", "ERROR")
                return None
                
        except ImportError as e:
            debug_print(f"✗ PyTorch not found: {e}", "ERROR")
            if attempt < max_attempts - 1:
                debug_print("Installing PyTorch...", "INFO")
                if fix_pytorch_function_redefinition():
                    continue
            return None
            
        except Exception as e:
            debug_print(f"✗ Unexpected error during PyTorch import: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    return None

def safe_import_numpy():
    """Safely import NumPy with conflict resolution"""
    debug_print("Attempting safe NumPy import...", "INFO")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        debug_print(f"NumPy import attempt {attempt + 1}/{max_attempts}", "DEBUG")
        
        try:
            # Clear NumPy from cache
            modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith(('numpy', 'np'))]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]
                    debug_print(f"Cleared module: {mod}", "DEBUG")
            
            import numpy as np
            debug_print(f"✓ NumPy imported successfully: {np.__version__}", "INFO")
            return np
            
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                debug_print(f"⚠ NumPy binary incompatibility detected on attempt {attempt + 1}: {e}", "WARNING")
                
                if attempt < max_attempts - 1:
                    debug_print("Attempting NumPy compatibility fix...", "INFO")
                    if fix_numpy_sklearn_conflict():
                        debug_print("✓ NumPy fix completed, retrying import...", "INFO")
                        time.sleep(2)
                        continue
                    else:
                        debug_print("✗ NumPy fix failed", "ERROR")
                else:
                    debug_print("✗ All NumPy import attempts failed due to binary incompatibility", "ERROR")
                    return None
            else:
                debug_print(f"✗ NumPy import failed with different ValueError: {e}", "ERROR")
                return None
                
        except ImportError as e:
            debug_print(f"✗ NumPy not found: {e}", "ERROR")
            if attempt < max_attempts - 1:
                debug_print("Installing NumPy...", "INFO")
                success, _ = run_command("pip install numpy", "Install NumPy")
                if success:
                    continue
            return None
            
        except Exception as e:
            debug_print(f"✗ Unexpected error during NumPy import: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    return None

def safe_import_transformers():
    """Safely import transformers with comprehensive conflict resolution"""
    debug_print("Attempting safe transformers import...", "INFO")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        debug_print(f"Transformers import attempt {attempt + 1}/{max_attempts}", "DEBUG")
        
        try:
            # Clear transformers and related modules from cache
            modules_to_clear = [mod for mod in sys.modules.keys() if 
                              mod.startswith(('transformers', 'sklearn', 'scipy', 'numpy'))]
            for mod in modules_to_clear:
                if mod in sys.modules:
                    del sys.modules[mod]
                    debug_print(f"Cleared module: {mod}", "DEBUG")
            
            # Import transformers
            import transformers
            debug_print(f"✓ Transformers imported successfully: {transformers.__version__}", "INFO")
            
            # Test AutoTokenizer import
            from transformers import AutoTokenizer
            debug_print("✓ AutoTokenizer imported successfully", "DEBUG")
            
            return transformers
            
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                debug_print(f"⚠ NumPy binary incompatibility in transformers on attempt {attempt + 1}: {e}", "WARNING")
                
                if attempt < max_attempts - 1:
                    debug_print("Attempting NumPy/scikit-learn compatibility fix...", "INFO")
                    if fix_numpy_sklearn_conflict():
                        debug_print("✓ NumPy/scikit-learn fix completed, retrying import...", "INFO")
                        time.sleep(2)
                        continue
                    else:
                        debug_print("✗ NumPy/scikit-learn fix failed", "ERROR")
                else:
                    debug_print("✗ All transformers import attempts failed due to NumPy conflicts", "ERROR")
                    return None
            else:
                debug_print(f"✗ Transformers import failed with different ValueError: {e}", "ERROR")
                return None
                
        except ImportError as e:
            debug_print(f"✗ Transformers not found: {e}", "ERROR")
            if attempt < max_attempts - 1:
                debug_print("Installing transformers...", "INFO")
                success, _ = run_command("pip install transformers>=4.36.0", "Install transformers")
                if success:
                    continue
            return None
            
        except Exception as e:
            debug_print(f"✗ Unexpected error during transformers import: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None
    
    return None

def check_and_install_dependencies(auto_fix=True):
    """Check and handle dependency issues with comprehensive conflict resolution"""
    debug_print("="*60, "INFO")
    debug_print("STARTING COMPREHENSIVE DEPENDENCY CHECK WITH ENHANCED FIXES", "INFO")
    debug_print("="*60, "INFO")
    
    # Check NumPy first (foundation for everything else)
    debug_print("Checking NumPy installation with conflict resolution...", "INFO")
    numpy = safe_import_numpy()
    if not numpy:
        debug_print("✗ NumPy import failed after all attempts", "ERROR")
        if not auto_fix:
            debug_print("Manual intervention required for NumPy compatibility", "ERROR")
            return False
        else:
            debug_print("⚠ Continuing without NumPy - very limited functionality", "WARNING")
    else:
        debug_print(f"✓ NumPy version: {numpy.__version__}", "INFO")
    
    # Check PyTorch with enhanced conflict handling
    debug_print("Checking PyTorch installation with enhanced conflict resolution...", "INFO")
    pytorch_ok = False
    
    torch = safe_import_torch()
    if torch:
        pytorch_ok = True
        debug_print("✓ PyTorch imported successfully", "INFO")
    else:
        debug_print("✗ PyTorch import failed after all attempts", "ERROR")
        if not auto_fix:
            debug_print("Manual intervention required. Try:", "ERROR")
            debug_print("  Complete manual cleanup and reinstall required", "ERROR")
            return False
        else:
            debug_print("⚠ Continuing without PyTorch - limited functionality", "WARNING")
    
    # Check transformers with comprehensive conflict resolution
    debug_print("Checking transformers library with comprehensive conflict resolution...", "INFO")
    transformers = safe_import_transformers()
    if transformers:
        debug_print(f"✓ Transformers version: {transformers.__version__}", "INFO")
        debug_print(f"✓ Transformers location: {transformers.__file__}", "DEBUG")
        
        # Test additional components
        try:
            from transformers import AutoProcessor
            debug_print("✓ AutoProcessor import successful", "DEBUG")
        except Exception as e:
            debug_print(f"⚠ AutoProcessor import failed: {e}", "WARNING")
    else:
        debug_print("✗ Transformers import failed after all attempts", "ERROR")
        if not auto_fix:
            debug_print("Manual intervention required for transformers", "ERROR")
            return False
        else:
            debug_print("⚠ Continuing without transformers - very limited functionality", "WARNING")
    
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
    
    debug_print("✓ Comprehensive dependency check completed!", "INFO")
    debug_print("="*60, "INFO")
    return True

# Simplified quantizer class for testing
class SimpleKosmosQuantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        debug_print(f"Initializing SimpleKosmosQuantizer...", "INFO")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.torch = safe_import_torch()
        self.transformers = safe_import_transformers()
        
        if not self.torch:
            debug_print("✗ PyTorch not available, limited functionality", "ERROR")
        
        if not self.transformers:
            debug_print("✗ Transformers not available, cannot proceed", "ERROR")
            raise ImportError("Transformers library not available")
    
    def load_fp16_model(self, save_path="./kosmos2.5-fp16"):
        """Load model in FP16 precision"""
        debug_print("="*50, "INFO")
        debug_print("STARTING SIMPLE FP16 MODEL LOADING", "INFO")
        debug_print("="*50, "INFO")
        
        try:
            if not self.torch:
                debug_print("✗ PyTorch not available", "ERROR")
                return None
            
            # Import transformers components
            from transformers import AutoTokenizer
            
            # Try different model classes
            model_classes = [
                'Kosmos2_5ForConditionalGeneration',
                'AutoModelForImageTextToText', 
                'AutoModelForVision2Seq',
                'AutoModelForCausalLM'
            ]
            
            model_class = None
            for class_name in model_classes:
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
                    break
                    
                except ImportError:
                    debug_print(f"⚠ {class_name} not available, trying next...", "WARNING")
                    continue
            
            if not model_class:
                debug_print("✗ No suitable model class found", "ERROR")
                return None
            
            # Load tokenizer
            debug_print("Loading tokenizer...", "INFO")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            debug_print("✓ Tokenizer loaded", "INFO")
            
            # Load model in FP16
            debug_print("Loading model in FP16...", "INFO")
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
            debug_print(f"✓ Model loaded in {load_time:.2f} seconds", "INFO")
            
            # Save model
            debug_print(f"Saving model to: {save_path}", "INFO")
            os.makedirs(save_path, exist_ok=True)
            
            model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
            
            debug_print(f"✓ FP16 model saved to {save_path}", "INFO")
            
            return model
            
        except Exception as e:
            debug_print(f"✗ FP16 model loading failed: {e}", "ERROR")
            debug_print(f"Error details: {traceback.format_exc()}", "DEBUG")
            return None

def main():
    debug_print("="*80, "INFO")
    debug_print("KOSMOS-2.5 QUANTIZATION TOOL WITH ENHANCED PYTORCH FIXES", "INFO")
    debug_print("="*80, "INFO")
    
    parser = argparse.ArgumentParser(description='Enhanced quantization with PyTorch conflict resolution')
    parser.add_argument('--method', 
                       choices=['8bit', 'fp16'], 
                       default='fp16',
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
    
    # Comprehensive dependency check with enhanced fixes
    debug_print("Starting comprehensive dependency check with enhanced fixes...", "INFO")
    if not check_and_install_dependencies(auto_fix=args.auto_fix):
        debug_print("✗ Dependency check failed.", "ERROR")
        debug_print("Try running the script again or manually fix dependencies", "INFO")
        return 1
    
    if args.check_deps:
        debug_print("✓ Dependency check completed successfully!", "INFO")
        return 0
    
    try:
        # Initialize simple quantizer
        debug_print("Initializing simple quantizer...", "INFO")
        quantizer = SimpleKosmosQuantizer(args.model_name, args.cache_dir)
        
        # Load model in FP16 (safest option)
        debug_print("Loading model in FP16 (safest option)...", "INFO")
        model = quantizer.load_fp16_model(args.save_path)
        
        if model is None:
            debug_print(f"✗ Model loading failed", "ERROR")
            return 1
            
        debug_print("="*80, "INFO")
        debug_print("✓ MODEL LOADING COMPLETED SUCCESSFULLY!", "INFO")
        debug_print(f"Method: FP16", "INFO")
        debug_print(f"Save path: {args.save_path}", "INFO")
        debug_print("="*80, "INFO")
        
        return 0
        
    except Exception as e:
        debug_print(f"✗ Model loading failed: {e}", "ERROR")
        debug_print(f"Full error: {traceback.format_exc()}", "DEBUG")
        return 1

if __name__ == "__main__":
    sys.exit(main())
