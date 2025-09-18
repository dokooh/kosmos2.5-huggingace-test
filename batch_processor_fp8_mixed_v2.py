#!/usr/bin/env python3
"""
Enhanced Batch Processor for Kosmos-2.5 8-bit Mixed Precision with Resource Monitoring

This module provides comprehensive batch processing capabilities with system resource tracking
for GPU memory, CPU usage, and RAM usage across all batches.
"""

import os
import sys
import argparse
import json
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
import queue
import psutil

# Import the 8-bit mixed precision inference modules
try:
    from ocr_fp8_mixed import EightBitOCRInference, debug_checkpoint, debug_memory_status
    from md_fp8_mixed import EightBitMarkdownInference
except ImportError as e:
    print(f"Error importing 8-bit mixed precision modules: {e}")
    print("Make sure ocr_fp8_mixed.py and md_fp8_mixed.py are in the same directory")
    sys.exit(1)

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_fp8_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class SystemResourceMonitor:
    """Comprehensive system resource monitor for GPU memory, CPU, and RAM usage"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.monitoring_active = False
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Resource samples storage
        self.cpu_samples = []
        self.ram_samples = []
        self.gpu_memory_samples = []
        self.gpu_utilization_samples = []
        
        # Initialize GPU monitoring
        self._initialize_gpu_monitoring()
        
    def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring capabilities"""
        try:
            import torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.gpu_count = torch.cuda.device_count()
                self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.gpu_count)]
                self.gpu_total_memory = []
                
                for i in range(self.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.gpu_total_memory.append(props.total_memory / (1024**3))  # GB
                
                logger.info(f"GPU monitoring initialized - {self.gpu_count} GPU(s) detected")
                for i, name in enumerate(self.gpu_names):
                    logger.info(f"  GPU {i}: {name} ({self.gpu_total_memory[i]:.1f} GB)")
            else:
                logger.info("CUDA not available - GPU monitoring disabled")
                self.gpu_count = 0
                self.gpu_names = []
                self.gpu_total_memory = []
                
        except ImportError:
            logger.warning("PyTorch not available - GPU monitoring disabled")
            self.torch_available = False
            self.cuda_available = False
            self.gpu_count = 0
            self.gpu_names = []
            self.gpu_total_memory = []
    
    def start_monitoring(self, sample_interval: float = 1.0):
        """Start background resource monitoring"""
        if self.monitoring_active:
            return
            
        self.reset_samples()
        self.monitoring_active = True
        self._stop_event.clear()
        
        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    self._collect_sample()
                    time.sleep(sample_interval)
                except Exception as e:
                    logger.debug(f"Error in monitoring loop: {e}")
                    time.sleep(sample_interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Resource monitoring started (interval: {sample_interval}s)")
    
    def stop_monitoring(self):
        """Stop background resource monitoring"""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            
        logger.info("Resource monitoring stopped")
    
    def _collect_sample(self):
        """Collect a single sample of system resources"""
        with self._lock:
            if not self.monitoring_active:
                return
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            
            # RAM usage
            memory_info = psutil.virtual_memory()
            ram_percent = memory_info.percent
            ram_used_gb = memory_info.used / (1024**3)
            self.ram_samples.append({
                'percent': ram_percent,
                'used_gb': ram_used_gb,
                'available_gb': memory_info.available / (1024**3)
            })
            
            # GPU monitoring if available
            if self.cuda_available and self.enable_gpu_monitoring:
                try:
                    import torch
                    gpu_data = []
                    
                    for i in range(self.gpu_count):
                        try:
                            # Set device context
                            torch.cuda.set_device(i)
                            
                            # Memory usage
                            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                            cached = torch.cuda.memory_reserved(i) / (1024**3)      # GB
                            total = self.gpu_total_memory[i]
                            
                            # Calculate percentages
                            allocated_percent = (allocated / total) * 100 if total > 0 else 0
                            cached_percent = (cached / total) * 100 if total > 0 else 0
                            
                            gpu_data.append({
                                'device': i,
                                'allocated_gb': allocated,
                                'cached_gb': cached,
                                'total_gb': total,
                                'allocated_percent': allocated_percent,
                                'cached_percent': cached_percent
                            })
                            
                        except Exception as e:
                            logger.debug(f"Failed to get GPU {i} stats: {e}")
                            gpu_data.append({
                                'device': i,
                                'allocated_gb': 0,
                                'cached_gb': 0,
                                'total_gb': self.gpu_total_memory[i] if i < len(self.gpu_total_memory) else 0,
                                'allocated_percent': 0,
                                'cached_percent': 0,
                                'error': str(e)
                            })
                    
                    self.gpu_memory_samples.append(gpu_data)
                    
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
    
    def get_current_snapshot(self) -> Dict[str, Any]:
        """Get current resource usage snapshot"""
        snapshot = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': psutil.virtual_memory()._asdict(),
        }
        
        if self.cuda_available and self.enable_gpu_monitoring:
            try:
                import torch
                gpu_info = []
                for i in range(self.gpu_count):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    total = self.gpu_total_memory[i]
                    
                    gpu_info.append({
                        'device': i,
                        'name': self.gpu_names[i],
                        'allocated_gb': allocated,
                        'cached_gb': cached,
                        'total_gb': total,
                        'allocated_percent': (allocated / total) * 100 if total > 0 else 0
                    })
                
                snapshot['gpu'] = gpu_info
            except Exception as e:
                snapshot['gpu_error'] = str(e)
        
        return snapshot
    
    def calculate_averages(self) -> Dict[str, Any]:
        """Calculate average resource usage statistics"""
        with self._lock:
            if not self.cpu_samples:
                return {'error': 'No samples collected'}
            
            # CPU averages
            cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples)
            cpu_max = max(self.cpu_samples)
            cpu_min = min(self.cpu_samples)
            
            # RAM averages
            ram_percents = [sample['percent'] for sample in self.ram_samples]
            ram_used_gbs = [sample['used_gb'] for sample in self.ram_samples]
            
            ram_percent_avg = sum(ram_percents) / len(ram_percents)
            ram_used_avg = sum(ram_used_gbs) / len(ram_used_gbs)
            ram_percent_max = max(ram_percents)
            ram_used_max = max(ram_used_gbs)
            
            averages = {
                'sample_count': len(self.cpu_samples),
                'cpu': {
                    'average_percent': cpu_avg,
                    'max_percent': cpu_max,
                    'min_percent': cpu_min
                },
                'ram': {
                    'average_percent': ram_percent_avg,
                    'average_used_gb': ram_used_avg,
                    'max_percent': ram_percent_max,
                    'max_used_gb': ram_used_max
                }
            }
            
            # GPU averages if available
            if self.gpu_memory_samples and self.cuda_available:
                gpu_averages = {}
                
                for device_id in range(self.gpu_count):
                    device_samples = []
                    for sample_set in self.gpu_memory_samples:
                        for gpu_sample in sample_set:
                            if gpu_sample.get('device') == device_id and 'error' not in gpu_sample:
                                device_samples.append(gpu_sample)
                    
                    if device_samples:
                        avg_allocated = sum(s['allocated_gb'] for s in device_samples) / len(device_samples)
                        avg_cached = sum(s['cached_gb'] for s in device_samples) / len(device_samples)
                        avg_allocated_percent = sum(s['allocated_percent'] for s in device_samples) / len(device_samples)
                        max_allocated = max(s['allocated_gb'] for s in device_samples)
                        max_allocated_percent = max(s['allocated_percent'] for s in device_samples)
                        
                        gpu_averages[f'gpu_{device_id}'] = {
                            'name': self.gpu_names[device_id] if device_id < len(self.gpu_names) else f'GPU {device_id}',
                            'total_gb': self.gpu_total_memory[device_id] if device_id < len(self.gpu_total_memory) else 0,
                            'average_allocated_gb': avg_allocated,
                            'average_cached_gb': avg_cached,
                            'average_allocated_percent': avg_allocated_percent,
                            'max_allocated_gb': max_allocated,
                            'max_allocated_percent': max_allocated_percent,
                            'sample_count': len(device_samples)
                        }
                
                averages['gpu'] = gpu_averages
                
                # Overall GPU summary
                if gpu_averages:
                    total_avg_allocated = sum(gpu['average_allocated_gb'] for gpu in gpu_averages.values())
                    total_max_allocated = sum(gpu['max_allocated_gb'] for gpu in gpu_averages.values())
                    total_capacity = sum(gpu['total_gb'] for gpu in gpu_averages.values())
                    
                    averages['gpu_summary'] = {
                        'total_average_allocated_gb': total_avg_allocated,
                        'total_max_allocated_gb': total_max_allocated,
                        'total_capacity_gb': total_capacity,
                        'average_utilization_percent': (total_avg_allocated / total_capacity) * 100 if total_capacity > 0 else 0,
                        'max_utilization_percent': (total_max_allocated / total_capacity) * 100 if total_capacity > 0 else 0
                    }
            
            return averages
    
    def reset_samples(self):
        """Reset all collected samples"""
        with self._lock:
            self.cpu_samples.clear()
            self.ram_samples.clear()
            self.gpu_memory_samples.clear()
            self.gpu_utilization_samples.clear()
    
    def print_current_status(self, prefix: str = ""):
        """Print current system resource status"""
        snapshot = self.get_current_snapshot()
        
        output = f"{prefix}CPU: {snapshot['cpu_percent']:.1f}% | "
        output += f"RAM: {snapshot['memory']['percent']:.1f}% ({snapshot['memory']['used']/1024**3:.1f}GB)"
        
        if 'gpu' in snapshot:
            gpu_info = snapshot['gpu']
            total_gpu_allocated = sum(gpu['allocated_gb'] for gpu in gpu_info)
            total_gpu_capacity = sum(gpu['total_gb'] for gpu in gpu_info)
            gpu_percent = (total_gpu_allocated / total_gpu_capacity) * 100 if total_gpu_capacity > 0 else 0
            output += f" | GPU: {gpu_percent:.1f}% ({total_gpu_allocated:.1f}GB)"
        
        print(output)

class EightBitBatchProcessor:
    """Enhanced batch processor with comprehensive resource monitoring"""
    
    def __init__(self, 
                 model_checkpoint: str,
                 device: str = None,
                 cache_dir: str = None,
                 use_8bit: bool = True,
                 mixed_precision: bool = True,
                 force_fallback: bool = False,
                 max_workers: int = 1,
                 use_process_pool: bool = False,
                 enable_resource_monitoring: bool = True):
        """Initialize the batch processor with resource monitoring"""
        
        debug_checkpoint("Initializing EightBitBatchProcessor with Resource Monitoring", "BATCH_INIT_START")
        
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit
        self.mixed_precision = mixed_precision
        self.force_fallback = force_fallback
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Initialize resource monitor
        self.resource_monitor = SystemResourceMonitor(enable_gpu_monitoring=True)
        
        # Validate model checkpoint
        self.is_local_checkpoint = os.path.exists(model_checkpoint)
        
        # Thread-local storage for inference engines
        self._local = threading.local()
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.total_tasks = 0
        self.completed_tasks = 0
        
        # Performance metrics
        self.start_time = None
        self.system_info = self._get_system_info()
        
        logger.info(f"Initialized 8-bit Mixed Precision Batch Processor with Resource Monitoring")
        logger.info(f"Model checkpoint: {self.model_checkpoint}")
        logger.info(f"Device: {self.device}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Resource monitoring: {self.enable_resource_monitoring}")
        
        debug_checkpoint("EightBitBatchProcessor initialization completed", "BATCH_INIT_END")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            import torch
            system_info = {
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                system_info['gpu_devices'] = gpu_info
            
            return system_info
        except Exception as e:
            logger.debug(f"Failed to get system info: {e}")
            return {}
    
    def _get_ocr_engine(self) -> EightBitOCRInference:
        """Get thread-local OCR engine"""
        if not hasattr(self._local, 'ocr_engine') or self._local.ocr_engine is None:
            debug_checkpoint("Creating thread-local OCR engine")
            try:
                self._local.ocr_engine = EightBitOCRInference(
                    model_checkpoint=self.model_checkpoint,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    use_8bit=self.use_8bit,
                    mixed_precision=self.mixed_precision,
                    force_fallback=self.force_fallback
                )
                debug_checkpoint("Thread-local OCR engine created successfully")
            except Exception as e:
                debug_checkpoint(f"Failed to create OCR engine: {e}")
                raise
        return self._local.ocr_engine
    
    def _get_md_engine(self) -> EightBitMarkdownInference:
        """Get thread-local Markdown engine"""
        if not hasattr(self._local, 'md_engine') or self._local.md_engine is None:
            debug_checkpoint("Creating thread-local Markdown engine")
            try:
                self._local.md_engine = EightBitMarkdownInference(
                    model_checkpoint=self.model_checkpoint,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    use_8bit=self.use_8bit,
                    mixed_precision=self.mixed_precision,
                    force_fallback=self.force_fallback
                )
                debug_checkpoint("Thread-local Markdown engine created successfully")
            except Exception as e:
                debug_checkpoint(f"Failed to create Markdown engine: {e}")
                raise
        return self._local.md_engine
    
    def find_images(self, input_folder: str, extensions: List[str]) -> List[str]:
        """Find and validate image files"""
        debug_checkpoint(f"Scanning for images in: {input_folder}", "FIND_IMAGES_START")
        
        image_files = []
        input_path = Path(input_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_folder}")
        
        # Search for images with case-insensitive extensions
        for ext in extensions:
            for case_ext in [ext.lower(), ext.upper()]:
                pattern = f"*{case_ext}"
                found_files = list(input_path.glob(pattern))
                image_files.extend(found_files)
        
        # Remove duplicates and convert to strings
        image_files = list(set([str(f) for f in image_files]))
        image_files.sort()
        
        # Validate image files
        valid_images = []
        for img_file in image_files:
            try:
                from PIL import Image
                with Image.open(img_file) as img:
                    img.verify()
                valid_images.append(img_file)
            except Exception as e:
                logger.warning(f"Skipping invalid image {Path(img_file).name}: {e}")
        
        logger.info(f"Found {len(valid_images)} valid images in {input_folder}")
        debug_checkpoint(f"Image scanning completed. Found {len(valid_images)} valid images", "FIND_IMAGES_END")
        
        return valid_images
    
    def create_output_structure(self, output_folder: str) -> Dict[str, str]:
        """Create comprehensive output folder structure"""
        debug_checkpoint(f"Creating output structure: {output_folder}", "OUTPUT_STRUCT_START")
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        folders = {
            'markdown': output_path / "markdown_output",
            'ocr_images': output_path / "ocr_annotated_images", 
            'ocr_text': output_path / "ocr_text_results",
            'logs': output_path / "processing_logs",
            'reports': output_path / "analysis_reports",
            'debug': output_path / "debug_info",
            'errors': output_path / "error_logs",
            'performance': output_path / "performance_metrics"
        }
        
        for folder_name, folder_path in folders.items():
            folder_path.mkdir(exist_ok=True)
            debug_checkpoint(f"Created folder: {folder_name} -> {folder_path}")
        
        folder_paths = {k: str(v) for k, v in folders.items()}
        
        logger.info("Created comprehensive output folder structure:")
        for name, path in folder_paths.items():
            logger.info(f"  {name}: {path}")
        
        debug_checkpoint("Output structure creation completed", "OUTPUT_STRUCT_END")
        return folder_paths
    
    def process_single_image_ocr(self, 
                                image_path: str, 
                                output_folders: Dict[str, str],
                                max_tokens: int,
                                task_id: int = 0) -> Dict[str, Any]:
        """Process a single image for OCR with error handling"""
        image_name = Path(image_path).stem
        start_time = time.time()
        
        debug_checkpoint(f"Starting OCR processing for: {image_name}", f"OCR_TASK_{task_id}_START")
        
        try:
            # Get thread-local OCR engine
            ocr_engine = self._get_ocr_engine()
            
            # Generate output paths
            output_image = Path(output_folders['ocr_images']) / f"{image_name}_ocr_annotated.png"
            output_text = Path(output_folders['ocr_text']) / f"{image_name}_ocr_results.txt"
            
            logger.info(f"[Task {task_id}] Processing OCR for: {Path(image_path).name}")
            
            if self.enable_resource_monitoring:
                self.resource_monitor.print_current_status(f"[Task {task_id}] Pre-OCR Resources - ")
            
            debug_memory_status()
            
            # Perform OCR
            result = ocr_engine.perform_ocr(
                image_path=image_path,
                max_tokens=max_tokens,
                save_image=str(output_image),
                save_text=str(output_text)
            )
            
            processing_time = time.time() - start_time
            
            if self.enable_resource_monitoring:
                self.resource_monitor.print_current_status(f"[Task {task_id}] Post-OCR Resources - ")
            
            # Extract statistics
            stats = result.get('statistics', {})
            
            success_result = {
                'success': True,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'output_image': str(output_image),
                'output_text': str(output_text),
                'text_regions': stats.get('total_regions', 0),
                'total_text_length': stats.get('total_text_length', 0),
                'avg_confidence': stats.get('avg_confidence', 0.0),
                'image_size': stats.get('image_size', [0, 0]),
                'processing_time': processing_time,
                'inference_time': result.get('inference_time', 0),
                'quantization_used': '8-bit' if ocr_engine.use_8bit else 'FP16/BF16',
                'results': result.get('results', []),
                'raw_output_length': len(result.get('raw_output', ''))
            }
            
            debug_checkpoint(f"OCR processing completed successfully for: {image_name}", f"OCR_TASK_{task_id}_SUCCESS")
            return success_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR failed for {Path(image_path).name}: {str(e)}"
            logger.error(error_msg)
            
            # Save error information
            error_file = Path(output_folders['errors']) / f"{image_name}_ocr_error.txt"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Processing Error Report\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Processing time: {processing_time:.2f}s\n\n")
                    f.write(f"Full traceback:\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            
            error_result = {
                'success': False,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': processing_time,
                'error_file': str(error_file)
            }
            
            debug_checkpoint(f"OCR processing failed for: {image_name}", f"OCR_TASK_{task_id}_FAILED")
            return error_result
    
    def process_single_image_markdown(self, 
                                    image_path: str, 
                                    output_folders: Dict[str, str],
                                    max_tokens: int,
                                    temperature: float,
                                    task_id: int = 0) -> Dict[str, Any]:
        """Process a single image for Markdown generation"""
        image_name = Path(image_path).stem
        start_time = time.time()
        
        debug_checkpoint(f"Starting Markdown processing for: {image_name}", f"MD_TASK_{task_id}_START")
        
        try:
            # Get thread-local Markdown engine
            md_engine = self._get_md_engine()
            
            # Generate output path
            output_file = Path(output_folders['markdown']) / f"{image_name}_document.md"
            
            logger.info(f"[Task {task_id}] Processing Markdown for: {Path(image_path).name}")
            
            if self.enable_resource_monitoring:
                self.resource_monitor.print_current_status(f"[Task {task_id}] Pre-MD Resources - ")
            
            debug_memory_status()
            
            # Generate markdown
            result = md_engine.generate_markdown(
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                save_output=str(output_file)
            )
            
            processing_time = time.time() - start_time
            
            if self.enable_resource_monitoring:
                self.resource_monitor.print_current_status(f"[Task {task_id}] Post-MD Resources - ")
            
            # Extract statistics
            stats = result.get('statistics', {})
            
            success_result = {
                'success': True,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'output_file': str(output_file),
                'word_count': stats.get('word_count', 0),
                'char_count': stats.get('char_count', 0),
                'line_count': stats.get('line_count', 0),
                'headers': stats.get('headers', 0),
                'lists': stats.get('lists', 0),
                'tables': stats.get('tables', 0),
                'code_blocks': stats.get('code_blocks', 0),
                'processing_time': processing_time,
                'inference_time': result.get('inference_time', 0),
                'quantization_used': '8-bit' if md_engine.use_8bit else 'FP16/BF16',
                'markdown_length': len(result.get('markdown', '')),
                'raw_output_length': len(result.get('raw_output', ''))
            }
            
            debug_checkpoint(f"Markdown processing completed successfully for: {image_name}", f"MD_TASK_{task_id}_SUCCESS")
            return success_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Markdown generation failed for {Path(image_path).name}: {str(e)}"
            logger.error(error_msg)
            
            # Save error information
            error_file = Path(output_folders['errors']) / f"{image_name}_md_error.txt"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Markdown Generation Error Report\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Processing time: {processing_time:.2f}s\n\n")
                    f.write(f"Full traceback:\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            
            error_result = {
                'success': False,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': processing_time,
                'error_file': str(error_file)
            }
            
            debug_checkpoint(f"Markdown processing failed for: {image_name}", f"MD_TASK_{task_id}_FAILED")
            return error_result
    
    def process_batch_sequential(self, 
                               image_paths: List[str],
                               output_folders: Dict[str, str],
                               process_ocr: bool,
                               process_md: bool,
                               ocr_max_tokens: int,
                               md_max_tokens: int,
                               temperature: float) -> Dict[str, Any]:
        """Process images sequentially with resource monitoring"""
        debug_checkpoint("Starting sequential batch processing", "BATCH_SEQ_START")
        
        # Start resource monitoring
        if self.enable_resource_monitoring:
            self.resource_monitor.start_monitoring(sample_interval=0.5)
        
        results = {
            'ocr_results': [],
            'md_results': [],
            'total_images': len(image_paths),
            'start_time': time.time(),
            'processing_mode': 'sequential'
        }
        
        self.total_tasks = len(image_paths) * (int(process_ocr) + int(process_md))
        self.completed_tasks = 0
        
        try:
            for i, image_path in enumerate(image_paths, 1):
                image_name = Path(image_path).name
                logger.info(f"Processing image {i}/{len(image_paths)}: {image_name}")
                
                # Process OCR if requested
                if process_ocr:
                    debug_checkpoint(f"Processing OCR for image {i}: {image_name}")
                    ocr_result = self.process_single_image_ocr(
                        image_path, output_folders, ocr_max_tokens, task_id=i
                    )
                    results['ocr_results'].append(ocr_result)
                    self.completed_tasks += 1
                    
                    progress = (self.completed_tasks / self.total_tasks) * 100
                    logger.info(f"Progress: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks})")
                
                # Process Markdown if requested
                if process_md:
                    debug_checkpoint(f"Processing Markdown for image {i}: {image_name}")
                    md_result = self.process_single_image_markdown(
                        image_path, output_folders, md_max_tokens, temperature, task_id=i
                    )
                    results['md_results'].append(md_result)
                    self.completed_tasks += 1
                    
                    progress = (self.completed_tasks / self.total_tasks) * 100
                    logger.info(f"Progress: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks})")
        
        finally:
            # Stop resource monitoring and collect averages
            if self.enable_resource_monitoring:
                self.resource_monitor.stop_monitoring()
                results['resource_averages'] = self.resource_monitor.calculate_averages()
        
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        debug_checkpoint("Sequential batch processing completed", "BATCH_SEQ_END")
        return results
    
    def process_batch_parallel(self, 
                             image_paths: List[str],
                             output_folders: Dict[str, str],
                             process_ocr: bool,
                             process_md: bool,
                             ocr_max_tokens: int,
                             md_max_tokens: int,
                             temperature: float) -> Dict[str, Any]:
        """Process images in parallel with resource monitoring"""
        debug_checkpoint("Starting parallel batch processing", "BATCH_PAR_START")
        
        # Start resource monitoring
        if self.enable_resource_monitoring:
            self.resource_monitor.start_monitoring(sample_interval=0.5)
        
        results = {
            'ocr_results': [],
            'md_results': [],
            'total_images': len(image_paths),
            'start_time': time.time(),
            'processing_mode': 'parallel'
        }
        
        self.total_tasks = len(image_paths) * (int(process_ocr) + int(process_md))
        self.completed_tasks = 0
        
        executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        logger.info(f"Using {executor_class.__name__} with {self.max_workers} workers")
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                futures = []
                task_id = 0
                
                # Submit OCR tasks
                if process_ocr:
                    for image_path in image_paths:
                        task_id += 1
                        future = executor.submit(
                            self.process_single_image_ocr,
                            image_path, output_folders, ocr_max_tokens, task_id
                        )
                        futures.append(('ocr', future, task_id))
                
                # Submit Markdown tasks
                if process_md:
                    for image_path in image_paths:
                        task_id += 1
                        future = executor.submit(
                            self.process_single_image_markdown,
                            image_path, output_folders, md_max_tokens, temperature, task_id
                        )
                        futures.append(('md', future, task_id))
                
                # Collect results
                for task_type, future in as_completed([f[1] for f in futures]):
                    self.completed_tasks += 1
                    progress = (self.completed_tasks / self.total_tasks) * 100
                    logger.info(f"Completed task {self.completed_tasks}/{self.total_tasks} ({progress:.1f}%)")
                    
                    try:
                        result = future.result()
                        if task_type == 'ocr':
                            results['ocr_results'].append(result)
                        else:
                            results['md_results'].append(result)
                            
                        if result.get('success', False):
                            logger.info(f"✓ {task_type.upper()} task completed: {result.get('image_name', 'unknown')}")
                        else:
                            logger.error(f"✗ {task_type.upper()} task failed: {result.get('image_name', 'unknown')} - {result.get('error', 'unknown error')}")
                            
                    except Exception as e:
                        logger.error(f"Task execution failed: {e}")
        
        finally:
            # Stop resource monitoring and collect averages
            if self.enable_resource_monitoring:
                self.resource_monitor.stop_monitoring()
                results['resource_averages'] = self.resource_monitor.calculate_averages()
        
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        debug_checkpoint("Parallel batch processing completed", "BATCH_PAR_END")
        return results
    
    def print_detailed_summary(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Print detailed summary with resource usage averages"""
        debug_checkpoint("Generating detailed summary", "SUMMARY_START")
        
        ocr_results = results.get('ocr_results', [])
        md_results = results.get('md_results', [])
        resource_averages = results.get('resource_averages', {})
        
        ocr_successful = [r for r in ocr_results if r.get('success', False)]
        md_successful = [r for r in md_results if r.get('success', False)]
        
        total_text_regions = sum(r.get('text_regions', 0) for r in ocr_successful)
        total_words = sum(r.get('word_count', 0) for r in md_successful)
        
        print(f"\n{'='*100}")
        print("8-BIT MIXED PRECISION BATCH PROCESSING SUMMARY WITH RESOURCE MONITORING")
        print(f"{'='*100}")
        
        print(f"\n📋 CONFIGURATION:")
        print(f"  Model Checkpoint: {config['model_checkpoint']}")
        print(f"  Device: {config['device'] or 'Auto-detected'}")
        print(f"  8-bit Quantization: {'Enabled' if config['use_8bit'] else 'Disabled'}")
        print(f"  Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
        print(f"  Max Workers: {config['max_workers']}")
        print(f"  Processing Mode: {results.get('processing_mode', 'Unknown').title()}")
        print(f"  Resource Monitoring: {'Enabled' if self.enable_resource_monitoring else 'Disabled'}")
        
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"  Total Images: {results['total_images']}")
        print(f"  Total Processing Time: {results['total_time']:.2f}s")
        print(f"  Average Time per Image: {results['total_time']/max(results['total_images'], 1):.2f}s")
        print(f"  Images per Second: {results['total_images']/max(results['total_time'], 1):.2f}")
        
        # Print resource usage averages
        if resource_averages and 'sample_count' in resource_averages:
            print(f"\n📊 RESOURCE USAGE AVERAGES ACROSS ALL BATCHES:")
            print(f"  Monitoring Samples: {resource_averages['sample_count']:,}")
            
            cpu_data = resource_averages.get('cpu', {})
            print(f"  CPU Usage:")
            print(f"    Average: {cpu_data.get('average_percent', 0):.1f}%")
            print(f"    Max: {cpu_data.get('max_percent', 0):.1f}%")
            print(f"    Min: {cpu_data.get('min_percent', 0):.1f}%")
            
            ram_data = resource_averages.get('ram', {})
            print(f"  RAM Usage:")
            print(f"    Average: {ram_data.get('average_percent', 0):.1f}% ({ram_data.get('average_used_gb', 0):.1f} GB)")
            print(f"    Max: {ram_data.get('max_percent', 0):.1f}% ({ram_data.get('max_used_gb', 0):.1f} GB)")
            
            gpu_data = resource_averages.get('gpu', {})
            gpu_summary = resource_averages.get('gpu_summary', {})
            
            if gpu_data:
                print(f"  GPU Memory Usage:")
                for gpu_id, gpu_info in gpu_data.items():
                    print(f"    {gpu_info['name']}:")
                    print(f"      Average Allocated: {gpu_info['average_allocated_gb']:.1f} GB ({gpu_info['average_allocated_percent']:.1f}%)")
                    print(f"      Max Allocated: {gpu_info['max_allocated_gb']:.1f} GB ({gpu_info['max_allocated_percent']:.1f}%)")
                    print(f"      Total Capacity: {gpu_info['total_gb']:.1f} GB")
                
                if gpu_summary:
                    print(f"  Overall GPU Summary:")
                    print(f"    Total Average Allocated: {gpu_summary['total_average_allocated_gb']:.1f} GB ({gpu_summary['average_utilization_percent']:.1f}%)")
                    print(f"    Total Max Allocated: {gpu_summary['total_max_allocated_gb']:.1f} GB ({gpu_summary['max_utilization_percent']:.1f}%)")
                    print(f"    Total GPU Capacity: {gpu_summary['total_capacity_gb']:.1f} GB")
            else:
                print(f"  GPU Memory Usage: N/A (No GPU detected or monitoring disabled)")
        else:
            print(f"\n📊 RESOURCE USAGE MONITORING:")
            print(f"  Status: Disabled or no samples collected")
        
        if ocr_results:
            print(f"\n🔍 OCR PROCESSING RESULTS:")
            print(f"  Total Processed: {len(ocr_results)}")
            print(f"  Successful: {len(ocr_successful)}")
            print(f"  Failed: {len(ocr_results) - len(ocr_successful)}")
            print(f"  Success Rate: {(len(ocr_successful)/max(len(ocr_results), 1))*100:.1f}%")
            print(f"  Total Text Regions Found: {total_text_regions:,}")
            print(f"  Average Regions per Image: {total_text_regions/max(len(ocr_successful), 1):.1f}")
        
        if md_results:
            print(f"\n📝 MARKDOWN GENERATION RESULTS:")
            print(f"  Total Processed: {len(md_results)}")
            print(f"  Successful: {len(md_successful)}")
            print(f"  Failed: {len(md_results) - len(md_successful)}")
            print(f"  Success Rate: {(len(md_successful)/max(len(md_results), 1))*100:.1f}%")
            print(f"  Total Words Generated: {total_words:,}")
            print(f"  Average Words per Image: {total_words/max(len(md_successful), 1):.0f}")
        
        if self.system_info:
            print(f"\n💻 SYSTEM INFORMATION:")
            print(f"  CPU Cores: {self.system_info.get('cpu_count_physical', 'Unknown')} physical, {self.system_info.get('cpu_count_logical', 'Unknown')} logical")
            print(f"  System Memory: {self.system_info.get('memory_total_gb', 0):.1f} GB")
            print(f"  CUDA Available: {'Yes' if self.system_info.get('cuda_available', False) else 'No'}")
            
            if self.system_info.get('gpu_devices'):
                print(f"  GPU Devices:")
                for i, gpu in enumerate(self.system_info['gpu_devices']):
                    print(f"    GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f} GB, Compute {gpu['compute_capability']})")
        
        print(f"\n{'='*100}")
        debug_checkpoint("Detailed summary generation completed", "SUMMARY_END")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='8-bit Mixed Precision Batch Processor with Resource Monitoring',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input folder containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output folder for results')
    
    # Model configuration
    parser.add_argument('--model_checkpoint', '-m', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    
    # Quantization settings
    parser.add_argument('--no_8bit', action='store_true',
                       help='Disable 8-bit quantization')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision')
    parser.add_argument('--force_fallback', action='store_true',
                       help='Force fallback mode')
    
    # Processing configuration
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='Maximum tokens for OCR')
    parser.add_argument('--md_tokens', type=int, default=2048,
                       help='Maximum tokens for markdown')
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Temperature for markdown generation')
    
    # Task selection
    parser.add_argument('--skip_ocr', action='store_true',
                       help='Skip OCR processing')
    parser.add_argument('--skip_md', action='store_true',
                       help='Skip markdown generation')
    
    # Performance configuration
    parser.add_argument('--max_workers', type=int, default=1,
                       help='Maximum number of parallel workers')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--use_process_pool', action='store_true',
                       help='Use process pool')
    
    # Resource monitoring
    parser.add_argument('--no_resource_monitoring', action='store_true',
                       help='Disable resource monitoring')
    
    # File handling
    parser.add_argument('--image_extensions', type=str, nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                       help='Image file extensions to process')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    debug_checkpoint("8-bit Mixed Precision Batch Processor with Resource Monitoring starting", "MAIN_START")
    
    args = get_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Validate arguments
    if args.skip_ocr and args.skip_md:
        logger.error("Cannot skip both OCR and markdown processing")
        sys.exit(1)
    
    if args.parallel and args.max_workers <= 1:
        logger.warning("Parallel processing requested but max_workers <= 1. Using sequential processing.")
        args.parallel = False
    
    # Configuration
    config = {
        'model_checkpoint': args.model_checkpoint,
        'cache_dir': args.cache_dir,
        'device': args.device,
        'use_8bit': not args.no_8bit,
        'mixed_precision': not args.no_mixed_precision,
        'force_fallback': args.force_fallback,
        'max_tokens': args.max_tokens,
        'md_max_tokens': args.md_tokens,
        'temperature': args.temperature,
        'max_workers': args.max_workers,
        'parallel': args.parallel,
        'use_process_pool': args.use_process_pool,
        'process_ocr': not args.skip_ocr,
        'process_md': not args.skip_md,
        'image_extensions': args.image_extensions,
        'enable_resource_monitoring': not args.no_resource_monitoring
    }
    
    try:
        # Initialize processor
        processor = EightBitBatchProcessor(
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
            use_8bit=not args.no_8bit,
            mixed_precision=not args.no_mixed_precision,
            force_fallback=args.force_fallback,
            max_workers=args.max_workers,
            use_process_pool=args.use_process_pool,
            enable_resource_monitoring=not args.no_resource_monitoring
        )
        
        # Find images
        logger.info(f"Scanning for images in: {args.input}")
        image_files = processor.find_images(args.input, args.image_extensions)
        
        if not image_files:
            logger.error(f"No valid images found in {args.input}")
            sys.exit(1)
        
        # Create output structure
        output_folders = processor.create_output_structure(args.output)
        
        # Process images
        logger.info(f"Starting batch processing of {len(image_files)} images")
        logger.info(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
        
        if args.parallel:
            results = processor.process_batch_parallel(
                image_files, output_folders,
                not args.skip_ocr, not args.skip_md,
                args.max_tokens, args.md_tokens, args.temperature
            )
        else:
            results = processor.process_batch_sequential(
                image_files, output_folders,
                not args.skip_ocr, not args.skip_md,
                args.max_tokens, args.md_tokens, args.temperature
            )
        
        # Print summary
        processor.print_detailed_summary(results, config)
        
        logger.info(f"Batch processing completed successfully!")
        debug_checkpoint("Application completed successfully", "MAIN_END")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
