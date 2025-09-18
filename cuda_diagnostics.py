#!/usr/bin/env python3
"""
CUDA and cuBLAS Diagnostics Script

This script helps diagnose GPU-related issues including cuBLAS errors.
"""

import torch
import sys
import gc
import psutil

def check_cuda_environment():
    """Check CUDA environment and capabilities"""
    print("=== CUDA Environment Check ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CUDA not available - cannot diagnose GPU issues")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi-processors: {props.multi_processor_count}")
    
    return True

def check_memory_status():
    """Check current memory status"""
    print("\n=== Memory Status ===")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available ({memory.percent:.1f}% used)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"GPU {i} memory: {allocated:.1f} GB allocated, {cached:.1f} GB cached, {total:.1f} GB total")

def test_matrix_operations():
    """Test matrix operations to reproduce cuBLAS errors"""
    print("\n=== Matrix Operation Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU tests")
        return
    
    try:
        # Test the specific dimensions from your error
        print("Testing matrix multiplication with error dimensions...")
        
        # Clear cache first
        torch.cuda.empty_cache()
        gc.collect()
        
        device = torch.cuda.current_device()
        print(f"Using device: {device}")
        
        # Create tensors with the problematic dimensions
        A = torch.randn(1536, 768, device='cuda', dtype=torch.float16)
        B = torch.randn(4096, 768, device='cuda', dtype=torch.float16)
        
        print(f"Tensor A shape: {A.shape}, dtype: {A.dtype}")
        print(f"Tensor B shape: {B.shape}, dtype: {B.dtype}")
        
        # Test the operation that's failing: B @ A.T
        print("Attempting B @ A.T operation...")
        C = torch.mm(B, A.t())
        print(f"Result shape: {C.shape}")
        print("✓ Matrix multiplication successful")
        
        # Test with different dtypes
        print("\nTesting with float32...")
        A_f32 = A.float()
        B_f32 = B.float()
        C_f32 = torch.mm(B_f32, A_f32.t())
        print("✓ Float32 multiplication successful")
        
        # Test with different memory layouts
        print("\nTesting with contiguous tensors...")
        A_cont = A.contiguous()
        B_cont = B.contiguous()
        C_cont = torch.mm(B_cont, A_cont.t())
        print("✓ Contiguous multiplication successful")
        
        del A, B, C, A_f32, B_f32, C_f32, A_cont, B_cont, C_cont
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Matrix operation failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Additional diagnostics
        print("\nAdditional diagnostics:")
        check_memory_status()
        
        # Try with smaller tensors
        print("\nTrying with smaller tensors...")
        try:
            A_small = torch.randn(100, 50, device='cuda', dtype=torch.float16)
            B_small = torch.randn(200, 50, device='cuda', dtype=torch.float16)
            C_small = torch.mm(B_small, A_small.t())
            print("✓ Small matrix multiplication successful")
            del A_small, B_small, C_small
        except Exception as e2:
            print(f"✗ Small matrix multiplication also failed: {e2}")

def test_quantization_compatibility():
    """Test 8-bit quantization compatibility"""
    print("\n=== Quantization Compatibility Test ===")
    
    try:
        # Test if bitsandbytes is available and working
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes version: {bnb.__version__}")
        
        # Test basic 8-bit operations
        if torch.cuda.is_available():
            test_tensor = torch.randn(1024, 768, device='cuda')
            
            # Test Int8 quantization
            try:
                from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
                quantized, state = quantize_blockwise(test_tensor)
                dequantized = dequantize_blockwise(quantized, state)
                print("✓ Basic 8-bit quantization test passed")
                del test_tensor, quantized, dequantized
            except Exception as e:
                print(f"✗ 8-bit quantization test failed: {e}")
        
    except ImportError:
        print("✗ bitsandbytes not available")
    except Exception as e:
        print(f"✗ Quantization test failed: {e}")

def suggest_fixes():
    """Suggest potential fixes for cuBLAS errors"""
    print("\n=== Suggested Fixes ===")
    
    fixes = [
        "1. Update CUDA drivers and toolkit",
        "2. Update PyTorch to latest version: pip install torch --upgrade",
        "3. Set environment variable: CUDA_LAUNCH_BLOCKING=1",
        "4. Reduce batch size or model size to use less GPU memory",
        "5. Use mixed precision training: torch.cuda.amp",
        "6. Clear GPU cache: torch.cuda.empty_cache()",
        "7. Restart Python process to clear memory",
        "8. Try using CPU instead of GPU temporarily",
        "9. Check for GPU hardware issues",
        "10. Use fallback mode in your script: --force_fallback"
    ]
    
    for fix in fixes:
        print(fix)

def main():
    """Main diagnostics function"""
    print("CUDA and cuBLAS Diagnostics")
    print("=" *
