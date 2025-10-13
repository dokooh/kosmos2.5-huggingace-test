#!/usr/bin/env python3
"""
Custom Batch Processor for Kosmos-2.5 with Dedicated Prompts

This script is based on batch_processor_fp8_mixed.py but includes dedicated prompts
for both OCR and Markdown extraction with enhanced prompt validation and display.

Features:
- Custom prompts for OCR and Markdown tasks
- Prompt validation and display
- FP8 mixed precision support
- Enhanced error handling
- Detailed output formatting
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

# Import the 8-bit mixed precision inference modules
try:
    sys.path.append(str(Path(__file__).parent / "fp8"))
    from fp8.ocr_fp8_mixed import EightBitOCRInference, debug_checkpoint
    from fp8.md_fp8_mixed import EightBitMarkdownInference
except ImportError as e:
    print(f"❌ Error importing 8-bit mixed precision modules: {e}")
    print("Make sure ocr_fp8_mixed.py and md_fp8_mixed.py are in the fp8/ directory")
    sys.exit(1)

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomBatchProcessor:
    """Custom batch processor with dedicated prompts for OCR and Markdown extraction"""
    
    def __init__(self, 
                 model_checkpoint: str,
                 device: str = None,
                 use_8bit: bool = True,
                 mixed_precision: bool = True):
        """
        Initialize the custom batch processor
        
        Args:
            model_checkpoint: Path to model checkpoint
            device: Device to use for inference
            use_8bit: Enable 8-bit quantization
            mixed_precision: Enable mixed precision
        """
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.use_8bit = use_8bit
        self.mixed_precision = mixed_precision
        
        # Thread-local storage for inference engines
        self._local = threading.local()
        
        # Custom prompts for different tasks
        self.prompts = {
            'ocr_detailed': """<grounding>Perform comprehensive OCR analysis on this image. Extract all visible text with precise bounding box coordinates. Include:
1. Main headings and titles
2. Body text and paragraphs  
3. Tables, lists, and structured data
4. Form fields and labels
5. Small text and footnotes
6. Any numerical data or codes
Provide exact text extraction with spatial relationships.""",

            'ocr_structured': """<grounding>Extract all text from this document image in a structured format. Identify and categorize:
- Headers and section titles
- Paragraph content
- Bullet points and numbered lists
- Table data with row/column structure
- Captions and labels
- Page numbers and metadata
Maintain the document's hierarchical structure and reading order.""",

            'markdown_comprehensive': """<md>Convert this document image to clean, well-structured Markdown format. Generate comprehensive markdown that includes:

# Document Structure
- Proper heading hierarchy (# ## ### etc.)
- Paragraph formatting with appropriate spacing
- Lists (ordered and unordered) with correct indentation
- Tables with proper alignment
- Code blocks for technical content
- Links and references where applicable

# Content Preservation
- Maintain original document flow and organization
- Preserve technical terminology and formatting
- Include all visible text content
- Represent visual elements appropriately

# Quality Standards
- Use semantic markdown elements
- Ensure proper nesting and hierarchy
- Create readable, professional output
- Follow markdown best practices

Generate complete, publication-ready markdown.""",

            'markdown_technical': """<md>Analyze this technical document and convert it to specialized Markdown optimized for technical content:

## Technical Elements
- Code snippets in appropriate language blocks
- Mathematical expressions and formulas
- Technical diagrams descriptions
- API documentation formatting
- Configuration examples
- Command-line instructions

## Structure
- Clear section hierarchy with descriptive headers
- Technical glossaries and definitions
- Step-by-step procedures as ordered lists
- Important notes and warnings as blockquotes
- Cross-references and internal links

## Formatting
- Consistent code fence languages
- Proper table formatting for specifications  
- Emphasis for key terms and concepts
- Inline code for technical terms
- Professional documentation standards

Create comprehensive technical documentation in markdown format.""",

            'markdown_business': """<md>Transform this business document into professional Markdown suitable for corporate documentation:

## Business Content Structure
- Executive summary sections
- Key findings and recommendations  
- Financial data in well-formatted tables
- Process flows and procedures
- Organizational charts descriptions
- Project timelines and milestones

## Professional Formatting
- Clear heading hierarchy for navigation
- Bullet points for key insights
- Numbered lists for procedures
- Tables for data presentation
- Blockquotes for important statements
- Professional language and tone

## Document Standards
- Consistent formatting throughout
- Appropriate emphasis and highlighting
- Clean, readable layout
- Corporate documentation style
- Comprehensive coverage of all content

Generate polished, business-ready markdown documentation."""
        }
        
        logger.info(f"🚀 Custom Batch Processor initialized")
        logger.info(f"📁 Model: {self.model_checkpoint}")
        logger.info(f"⚙️  8-bit: {self.use_8bit}, Mixed Precision: {self.mixed_precision}")
        logger.info(f"📝 Loaded {len(self.prompts)} custom prompts")
    
    def display_prompts(self):
        """Display all available prompts for validation"""
        print("\\n" + "="*80)
        print("📝 CUSTOM PROMPTS VALIDATION")
        print("="*80)
        
        for prompt_name, prompt_text in self.prompts.items():
            print(f"\\n🎯 {prompt_name.upper().replace('_', ' ')}:")
            print("-" * 60)
            print(prompt_text)
            print("-" * 60)
            print(f"📏 Length: {len(prompt_text)} characters")
            print(f"📐 Token estimate: ~{len(prompt_text.split())} words")
    
    def validate_prompt(self, prompt_text: str, prompt_name: str) -> Dict[str, Any]:
        """Validate prompt quality and characteristics"""
        validation = {
            'name': prompt_name,
            'length': len(prompt_text),
            'word_count': len(prompt_text.split()),
            'has_grounding': '<grounding>' in prompt_text,
            'has_markdown': '<md>' in prompt_text,
            'has_structure': any(keyword in prompt_text.lower() for keyword in ['header', 'section', 'paragraph', 'list']),
            'has_instructions': any(keyword in prompt_text.lower() for keyword in ['extract', 'convert', 'generate', 'analyze']),
            'complexity_score': 0
        }
        
        # Calculate complexity score
        complexity_indicators = [
            'comprehensive', 'detailed', 'structured', 'hierarchical',
            'technical', 'professional', 'quality', 'standards'
        ]
        validation['complexity_score'] = sum(1 for indicator in complexity_indicators 
                                           if indicator in prompt_text.lower())
        
        return validation
    
    def _get_ocr_engine(self) -> EightBitOCRInference:
        """Get thread-local OCR engine"""
        if not hasattr(self._local, 'ocr_engine') or self._local.ocr_engine is None:
            self._local.ocr_engine = EightBitOCRInference(
                model_checkpoint=self.model_checkpoint,
                device=self.device,
                use_8bit=self.use_8bit,
                mixed_precision=self.mixed_precision
            )
        return self._local.ocr_engine
    
    def _get_md_engine(self) -> EightBitMarkdownInference:
        """Get thread-local Markdown engine"""
        if not hasattr(self._local, 'md_engine') or self._local.md_engine is None:
            self._local.md_engine = EightBitMarkdownInference(
                model_checkpoint=self.model_checkpoint,
                device=self.device,
                use_8bit=self.use_8bit,
                mixed_precision=self.mixed_precision
            )
        return self._local.md_engine
    
    def process_image_with_custom_ocr(self, 
                                    image_path: str, 
                                    prompt_key: str = 'ocr_detailed',
                                    max_tokens: int = 1024) -> Dict[str, Any]:
        """Process image with custom OCR prompt"""
        
        if prompt_key not in self.prompts:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        
        prompt_text = self.prompts[prompt_key]
        image_name = Path(image_path).stem
        
        print(f"\\n🔍 OCR Processing: {image_name}")
        print(f"📝 Using prompt: {prompt_key}")
        print(f"🎯 Prompt preview: {prompt_text[:100]}...")
        
        start_time = time.time()
        
        try:
            # Validate prompt
            validation = self.validate_prompt(prompt_text, prompt_key)
            
            # Get OCR engine
            ocr_engine = self._get_ocr_engine()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Process with custom prompt
            print(f"⚡ Starting OCR inference...")
            result = ocr_engine.perform_ocr_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens
            )
            
            processing_time = time.time() - start_time
            
            # Enhanced result formatting
            enhanced_result = {
                'success': True,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'prompt_validation': validation,
                'processing_time': processing_time,
                'ocr_results': result,
                'text_extracted': result.get('extracted_text', ''),
                'text_regions': result.get('text_regions', 0),
                'bounding_boxes': result.get('bounding_boxes', []),
                'confidence_scores': result.get('confidence_scores', []),
                'quantization_used': result.get('quantization_info', 'unknown')
            }
            
            print(f"✅ OCR completed in {processing_time:.2f}s")
            print(f"📊 Text regions found: {enhanced_result['text_regions']}")
            
            return enhanced_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"❌ OCR failed: {e}")
            return error_result
    
    def process_image_with_custom_markdown(self, 
                                         image_path: str, 
                                         prompt_key: str = 'markdown_comprehensive',
                                         max_tokens: int = 2048,
                                         temperature: float = 0.1) -> Dict[str, Any]:
        """Process image with custom Markdown prompt"""
        
        if prompt_key not in self.prompts:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        
        prompt_text = self.prompts[prompt_key]
        image_name = Path(image_path).stem
        
        print(f"\\n📝 Markdown Processing: {image_name}")
        print(f"🎯 Using prompt: {prompt_key}")
        print(f"📄 Prompt preview: {prompt_text[:100]}...")
        
        start_time = time.time()
        
        try:
            # Validate prompt
            validation = self.validate_prompt(prompt_text, prompt_key)
            
            # Get Markdown engine
            md_engine = self._get_md_engine()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Process with custom prompt
            print(f"⚡ Starting Markdown generation...")
            result = md_engine.generate_markdown_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            # Analyze generated markdown
            markdown_text = result.get('generated_markdown', '')
            
            # Enhanced result formatting
            enhanced_result = {
                'success': True,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'prompt_validation': validation,
                'processing_time': processing_time,
                'markdown_results': result,
                'generated_markdown': markdown_text,
                'word_count': len(markdown_text.split()),
                'char_count': len(markdown_text),
                'headers': markdown_text.count('#'),
                'lists': markdown_text.count('- ') + markdown_text.count('* '),
                'tables': markdown_text.count('|'),
                'code_blocks': markdown_text.count('```'),
                'quantization_used': result.get('quantization_info', 'unknown')
            }
            
            print(f"✅ Markdown completed in {processing_time:.2f}s")
            print(f"📊 Generated {enhanced_result['word_count']} words, {enhanced_result['headers']} headers")
            
            return enhanced_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"❌ Markdown generation failed: {e}")
            return error_result
    
    def process_image_dual_mode(self, 
                              image_path: str,
                              ocr_prompt_key: str = 'ocr_detailed',
                              md_prompt_key: str = 'markdown_comprehensive',
                              ocr_max_tokens: int = 1024,
                              md_max_tokens: int = 2048,
                              temperature: float = 0.1) -> Dict[str, Any]:
        """Process image with both OCR and Markdown using custom prompts"""
        
        image_name = Path(image_path).stem
        
        print(f"\\n🎯 DUAL MODE PROCESSING: {image_name}")
        print("="*60)
        
        start_time = time.time()
        
        # Process OCR
        print(f"\\n1️⃣ OCR PHASE:")
        ocr_result = self.process_image_with_custom_ocr(
            image_path=image_path,
            prompt_key=ocr_prompt_key,
            max_tokens=ocr_max_tokens
        )
        
        # Process Markdown
        print(f"\\n2️⃣ MARKDOWN PHASE:")
        md_result = self.process_image_with_custom_markdown(
            image_path=image_path,
            prompt_key=md_prompt_key,
            max_tokens=md_max_tokens,
            temperature=temperature
        )
        
        total_time = time.time() - start_time
        
        # Combined results
        combined_result = {
            'image_path': image_path,
            'image_name': image_name,
            'total_processing_time': total_time,
            'ocr_result': ocr_result,
            'markdown_result': md_result,
            'both_successful': ocr_result.get('success', False) and md_result.get('success', False),
            'prompts_used': {
                'ocr_prompt': ocr_prompt_key,
                'markdown_prompt': md_prompt_key
            }
        }
        
        print(f"\\n🎊 DUAL MODE SUMMARY:")
        print(f"   OCR Success: {'✅' if ocr_result.get('success') else '❌'}")
        print(f"   Markdown Success: {'✅' if md_result.get('success') else '❌'}")
        print(f"   Total Time: {total_time:.2f}s")
        
        return combined_result
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save processing results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = results['image_name']
        
        # Save combined results
        results_file = output_path / f"{image_name}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save OCR text if successful
        if results['ocr_result'].get('success'):
            ocr_text = results['ocr_result'].get('text_extracted', '')
            if ocr_text:
                ocr_file = output_path / f"{image_name}_ocr.txt"
                with open(ocr_file, 'w', encoding='utf-8') as f:
                    f.write(ocr_text)
        
        # Save Markdown if successful
        if results['markdown_result'].get('success'):
            markdown_text = results['markdown_result'].get('generated_markdown', '')
            if markdown_text:
                md_file = output_path / f"{image_name}_markdown.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
        
        print(f"\\n💾 Results saved to: {output_path}")
        return str(output_path)


def main():
    """Main execution function"""
    print("🚀 Custom Batch Processor for Kosmos-2.5")
    print("="*50)
    
    # Configuration
    model_checkpoint = "fp8/models/kosmos-fp8-mixed"  # Adjust path as needed
    image_path = "eds_pages/Final-Tech901-5-16383 (1)_pages/page_042.png"  # Using available page
    output_dir = "./custom_processing_results"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Available images:")
        parent_dir = Path(image_path).parent
        if parent_dir.exists():
            for img in sorted(parent_dir.glob("page_*.png"))[:5]:
                print(f"   📄 {img.name}")
        return
    
    # Initialize processor
    try:
        processor = CustomBatchProcessor(
            model_checkpoint=model_checkpoint,
            device=None,  # Auto-detect
            use_8bit=True,
            mixed_precision=True
        )
        
        # Display prompts for validation
        processor.display_prompts()
        
        # Process image with dual mode
        print(f"\\n🎯 Processing image: {Path(image_path).name}")
        results = processor.process_image_dual_mode(
            image_path=image_path,
            ocr_prompt_key='ocr_structured',
            md_prompt_key='markdown_comprehensive',
            ocr_max_tokens=1024,
            md_max_tokens=2048,
            temperature=0.1
        )
        
        # Save results
        output_path = processor.save_results(results, output_dir)
        
        # Final summary
        print(f"\\n" + "="*80)
        print("🎊 PROCESSING COMPLETE")
        print("="*80)
        print(f"📁 Output saved to: {output_path}")
        print(f"⏱️  Total processing time: {results['total_processing_time']:.2f}s")
        print(f"✅ Both tasks successful: {results['both_successful']}")
        
        if results['ocr_result'].get('success'):
            ocr_stats = results['ocr_result']
            print(f"\\n📊 OCR Results:")
            print(f"   Text regions: {ocr_stats.get('text_regions', 0)}")
            print(f"   Processing time: {ocr_stats.get('processing_time', 0):.2f}s")
        
        if results['markdown_result'].get('success'):
            md_stats = results['markdown_result']
            print(f"\\n📝 Markdown Results:")
            print(f"   Words generated: {md_stats.get('word_count', 0)}")
            print(f"   Headers found: {md_stats.get('headers', 0)}")
            print(f"   Processing time: {md_stats.get('processing_time', 0):.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()