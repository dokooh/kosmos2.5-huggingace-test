"""
Test script for quantized Kosmos-2.5 models
Tests both NF4_DoubleQuant and FP4_BF16 models with OCR and Markdown tasks
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import subprocess

def get_args():
    parser = argparse.ArgumentParser(description='Test quantized Kosmos-2.5 models')
    parser.add_argument('--models_dir', '-md', type=str, default='./quantized_models',
                       help='Directory containing quantized models')
    parser.add_argument('--test_images_dir', '-td', type=str, default='./web/test_images',
                       help='Directory containing test images')
    parser.add_argument('--output_dir', '-od', type=str, default='./test_results',
                       help='Output directory for test results')
    parser.add_argument('--models', '-m', nargs='+', 
                       default=['NF4_DoubleQuant', 'FP4_BF16'],
                       help='Models to test (default: NF4_DoubleQuant FP4_BF16)')
    parser.add_argument('--tasks', '-t', nargs='+',
                       default=['ocr', 'markdown'],
                       help='Tasks to test (default: ocr markdown)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--device', '-d', type=str, default='auto',
                       help='Device to use (default: auto)')
    return parser.parse_args()

class QuantizedModelTester:
    def __init__(self, models_dir, output_dir, verbose=False):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test results storage
        self.results = {}
    
    def find_test_images(self, test_images_dir):
        """Find available test images"""
        test_dir = Path(test_images_dir)
        if not test_dir.exists():
            print(f"Test images directory not found: {test_dir}")
            return []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        images = []
        
        for ext in image_extensions:
            images.extend(test_dir.glob(f'*{ext}'))
            images.extend(test_dir.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    def check_model_availability(self, model_name):
        """Check if a quantized model is available"""
        model_path = self.models_dir / model_name
        return model_path.exists() and (model_path / 'config.json').exists()
    
    def run_ocr_test(self, model_path, image_path, device='auto', benchmark=False):
        """Run OCR test on a model"""
        output_image = self.output_dir / f"{model_path.name}_ocr_{image_path.stem}.png"
        output_text = self.output_dir / f"{model_path.name}_ocr_{image_path.stem}.txt"
        
        cmd = [
            sys.executable, 'ocr_quantized.py',
            '--image', str(image_path),
            '--model_path', str(model_path),
            '--output', str(output_image),
            '--text_output', str(output_text),
            '--device', device
        ]
        
        if self.verbose:
            cmd.append('--verbose')
        
        if self.verbose:
            print(f"Running OCR test: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            success = result.returncode == 0
            execution_time = end_time - start_time
            
            # Parse output for performance metrics if verbose
            performance_metrics = {}
            if self.verbose and success:
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Inference time:' in line:
                        performance_metrics['inference_time'] = float(line.split(':')[1].strip().replace('s', ''))
                    elif 'CPU memory increase:' in line:
                        performance_metrics['cpu_memory'] = float(line.split(':')[1].strip().replace('MB', ''))
                    elif 'GPU memory increase:' in line:
                        performance_metrics['gpu_memory'] = float(line.split(':')[1].strip().replace('MB', ''))
            
            return {
                'success': success,
                'execution_time': execution_time,
                'output_image': str(output_image) if success else None,
                'output_text': str(output_text) if success else None,
                'error': result.stderr if not success else None,
                'stdout': result.stdout,
                'performance_metrics': performance_metrics
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'execution_time': 300,
                'error': 'Test timed out after 300 seconds',
                'performance_metrics': {}
            }
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'error': str(e),
                'performance_metrics': {}
            }
    
    def run_markdown_test(self, model_path, image_path, device='auto', benchmark=False):
        """Run Markdown test on a model"""
        output_file = self.output_dir / f"{model_path.name}_md_{image_path.stem}.md"
        
        cmd = [
            sys.executable, 'md_quantized.py',
            '--image', str(image_path),
            '--model_path', str(model_path),
            '--output', str(output_file),
            '--device', device
        ]
        
        if self.verbose:
            cmd.append('--verbose')
        
        if benchmark:
            cmd.append('--benchmark')
        
        if self.verbose:
            print(f"Running Markdown test: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            success = result.returncode == 0
            execution_time = end_time - start_time
            
            # Parse output for performance metrics
            performance_metrics = {}
            if self.verbose and success:
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'Inference time:' in line:
                        performance_metrics['inference_time'] = float(line.split(':')[1].strip().replace('s', ''))
                    elif 'Tokens/second:' in line:
                        performance_metrics['tokens_per_second'] = float(line.split(':')[1].strip())
                    elif 'CPU memory increase:' in line:
                        performance_metrics['cpu_memory'] = float(line.split(':')[1].strip().replace('MB', ''))
                    elif 'GPU memory increase:' in line:
                        performance_metrics['gpu_memory'] = float(line.split(':')[1].strip().replace('MB', ''))
            
            return {
                'success': success,
                'execution_time': execution_time,
                'output_file': str(output_file) if success else None,
                'error': result.stderr if not success else None,
                'stdout': result.stdout,
                'performance_metrics': performance_metrics
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'execution_time': 300,
                'error': 'Test timed out after 300 seconds',
                'performance_metrics': {}
            }
        except Exception as e:
            return {
                'success': False,
                'execution_time': 0,
                'error': str(e),
                'performance_metrics': {}
            }
    
    def run_comprehensive_test(self, models, tasks, test_images, device='auto', benchmark=False):
        """Run comprehensive tests on all models and tasks"""
        
        print("🚀 Starting Comprehensive Quantized Model Tests")
        print("=" * 60)
        
        for model_name in models:
            model_path = self.models_dir / model_name
            
            if not self.check_model_availability(model_name):
                print(f"❌ Model not available: {model_name}")
                self.results[model_name] = {'available': False}
                continue
            
            print(f"\n📊 Testing model: {model_name}")
            print("-" * 40)
            
            self.results[model_name] = {
                'available': True,
                'tasks': {}
            }
            
            # Load model metadata if available
            metadata_path = model_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.results[model_name]['metadata'] = metadata
                if self.verbose:
                    print(f"Model size: {metadata.get('size_mb', 'unknown')} MB")
            
            for task in tasks:
                print(f"\n  🔧 Running {task.upper()} tests...")
                
                self.results[model_name]['tasks'][task] = {}
                
                for image_path in test_images:
                    image_name = image_path.stem
                    
                    if self.verbose:
                        print(f"    Testing with {image_name}...")
                    
                    if task == 'ocr':
                        result = self.run_ocr_test(model_path, image_path, device, benchmark)
                    elif task == 'markdown':
                        result = self.run_markdown_test(model_path, image_path, device, benchmark)
                    else:
                        print(f"    ⚠️  Unknown task: {task}")
                        continue
                    
                    self.results[model_name]['tasks'][task][image_name] = result
                    
                    if result['success']:
                        print(f"    ✅ {image_name}: {result['execution_time']:.1f}s")
                    else:
                        print(f"    ❌ {image_name}: {result['error']}")
        
        return self.results
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        report = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {},
            'detailed_results': self.results
        }
        
        # Generate summary statistics
        total_tests = 0
        successful_tests = 0
        avg_execution_times = {}
        
        for model_name, model_results in self.results.items():
            if not model_results.get('available', False):
                continue
            
            model_summary = {
                'total_tests': 0,
                'successful_tests': 0,
                'avg_execution_time': 0,
                'tasks_summary': {}
            }
            
            total_execution_time = 0
            
            for task_name, task_results in model_results.get('tasks', {}).items():
                task_total = len(task_results)
                task_successful = sum(1 for r in task_results.values() if r['success'])
                task_avg_time = sum(r['execution_time'] for r in task_results.values()) / task_total if task_total > 0 else 0
                
                model_summary['tasks_summary'][task_name] = {
                    'total': task_total,
                    'successful': task_successful,
                    'success_rate': task_successful / task_total if task_total > 0 else 0,
                    'avg_execution_time': task_avg_time
                }
                
                model_summary['total_tests'] += task_total
                model_summary['successful_tests'] += task_successful
                total_execution_time += task_avg_time * task_total
                
                total_tests += task_total
                successful_tests += task_successful
            
            if model_summary['total_tests'] > 0:
                model_summary['avg_execution_time'] = total_execution_time / model_summary['total_tests']
                model_summary['success_rate'] = model_summary['successful_tests'] / model_summary['total_tests']
            
            report['summary'][model_name] = model_summary
        
        # Overall summary
        report['summary']['overall'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0
        }
        
        # Save report
        report_path = self.output_dir / 'test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report, report_path
    
    def print_summary(self, report):
        """Print a human-readable test summary"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        overall = report['summary']['overall']
        print(f"Overall Success Rate: {overall['success_rate']:.1%} ({overall['successful_tests']}/{overall['total_tests']})")
        
        print(f"\n📊 Model Performance:")
        print("-" * 40)
        
        for model_name, model_summary in report['summary'].items():
            if model_name == 'overall':
                continue
            
            if not isinstance(model_summary, dict) or 'success_rate' not in model_summary:
                print(f"❌ {model_name}: Not available or failed to load")
                continue
            
            print(f"🔹 {model_name}:")
            print(f"   Success Rate: {model_summary['success_rate']:.1%}")
            print(f"   Avg Time: {model_summary['avg_execution_time']:.1f}s")
            
            for task_name, task_summary in model_summary['tasks_summary'].items():
                print(f"     {task_name.upper()}: {task_summary['success_rate']:.1%} ({task_summary['avg_execution_time']:.1f}s avg)")

def main():
    args = get_args()
    
    # Initialize tester
    tester = QuantizedModelTester(
        args.models_dir,
        args.output_dir,
        args.verbose
    )
    
    # Find test images
    test_images = tester.find_test_images(args.test_images_dir)
    
    if not test_images:
        print(f"❌ No test images found in {args.test_images_dir}")
        print("Please run create_test_files.py to generate test images")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(test_images)} test images:")
        for img in test_images:
            print(f"  - {img.name}")
    
    # Check if quantized models scripts exist
    required_scripts = ['ocr_quantized.py', 'md_quantized.py']
    for script in required_scripts:
        if not Path(script).exists():
            print(f"❌ Required script not found: {script}")
            sys.exit(1)
    
    # Run tests
    results = tester.run_comprehensive_test(
        args.models,
        args.tasks,
        test_images,
        args.device,
        args.benchmark
    )
    
    # Generate and print report
    report, report_path = tester.generate_report()
    tester.print_summary(report)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    print(f"🗂️  Test outputs saved to: {args.output_dir}")
    
    # Print next steps
    print(f"\n🎯 Next Steps:")
    print("1. Review the test results in the output directory")
    print("2. Check individual output files for quality assessment")
    print("3. Compare performance between quantized models")
    print("4. Use the best performing model for your use case")

if __name__ == "__main__":
    main()
