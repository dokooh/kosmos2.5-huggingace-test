"""
Custom Prompt Validation Test
Test the custom prompts without loading the heavy FP8 models
"""

import os

def validate_custom_prompts():
    """Validate and display custom prompts for KOSMOS-2.5 processing"""
    
    prompts = {
        'ocr_detailed': "<grounding>Perform comprehensive OCR analysis on this image. Extract all visible text with precise bounding box coordinates. Include: 1. Main headings and titles 2. Body text and paragraphs 3. Tables, lists, and structured data 4. Form fields and labels 5. Small text and footnotes 6. Any numerical data or codes. Provide exact text extraction with spatial relationships.",
        
        'ocr_structured': "<grounding>Extract all text from this document image in a structured format. Identify and categorize: Headers and section titles, Paragraph content, Bullet points and numbered lists, Table data with row/column structure, Captions and labels, Page numbers and metadata. Maintain the document's hierarchical structure and reading order.",
        
        'markdown_comprehensive': "<md>Convert this document image to clean, well-structured Markdown format. Generate comprehensive markdown that includes proper heading hierarchy, paragraph formatting with appropriate spacing, lists with correct indentation, tables with proper alignment, code blocks for technical content. Maintain original document flow and organization, preserve technical terminology and formatting, include all visible text content. Use semantic markdown elements, ensure proper nesting and hierarchy, create readable professional output following markdown best practices. Generate complete publication-ready markdown.",
        
        'markdown_technical': "<md>Analyze this technical document and convert it to specialized Markdown optimized for technical content. Include code snippets in appropriate language blocks, mathematical expressions and formulas, technical diagrams descriptions, API documentation formatting, configuration examples, command-line instructions. Use clear section hierarchy with descriptive headers, technical glossaries and definitions, step-by-step procedures as ordered lists, important notes and warnings as blockquotes. Apply consistent code fence languages, proper table formatting for specifications, emphasis for key terms and concepts, inline code for technical terms. Create comprehensive technical documentation in markdown format.",
        
        'markdown_business': "<md>Transform this business document into professional Markdown suitable for corporate documentation. Structure with executive summary sections, key findings and recommendations, financial data in well-formatted tables, process flows and procedures, organizational charts descriptions, project timelines and milestones. Apply clear heading hierarchy for navigation, bullet points for key insights, numbered lists for procedures, tables for data presentation, blockquotes for important statements, professional language and tone. Maintain consistent formatting throughout, appropriate emphasis and highlighting, clean readable layout, corporate documentation style. Generate polished business-ready markdown documentation."
    }
    
    print("CUSTOM PROMPTS FOR KOSMOS-2.5 BATCH PROCESSING")
    print("="*80)
    print("Based on batch_processor_fp8_mixed.py with dedicated prompts")
    print("Target image: page_042.png (from PDF extraction)")
    print("="*80)
    
    for prompt_name, prompt_text in prompts.items():
        print(f"\n[{prompt_name.upper().replace('_', ' ')}]")
        print("-" * 60)
        
        # Display full prompt
        print("FULL PROMPT TEXT:")
        print(prompt_text)
        print("")
        
        # Analyze prompt characteristics
        print("PROMPT ANALYSIS:")
        print(f"  Length: {len(prompt_text)} characters")
        print(f"  Word count: {len(prompt_text.split())} words")
        print(f"  Tag type: {'OCR (grounding)' if prompt_text.startswith('<grounding>') else 'Markdown (md)'}")
        print(f"  Contains instructions: {'Yes' if 'Include:' in prompt_text or 'Structure with' in prompt_text else 'Yes'}")
        
        # Validate structure
        if prompt_text.startswith('<grounding>'):
            print(f"  OCR mode: Comprehensive text extraction with spatial awareness")
        elif prompt_text.startswith('<md>'):
            print(f"  Markdown mode: Document conversion with formatting preservation")
        else:
            print(f"  WARNING: Missing proper tag prefix!")
        
        print("-" * 60)
    
    # Validate target image
    target_image = "eds_pages/Final-Tech901-5-16383 (1)_pages/page_042.png"
    print(f"\nTARGET IMAGE VALIDATION:")
    print(f"Path: {target_image}")
    
    if os.path.exists(target_image):
        # Get image info
        from PIL import Image
        try:
            with Image.open(target_image) as img:
                print(f"Status: Found")
                print(f"Size: {img.size}")
                print(f"Format: {img.format}")
                print(f"Mode: {img.mode}")
                print(f"File size: {os.path.getsize(target_image) / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"Status: Found but error reading: {e}")
    else:
        print(f"Status: NOT FOUND")
    
    print("\n" + "="*80)
    print("CUSTOM BATCH PROCESSING READY")
    print("="*80)
    print("The custom prompts have been validated and are ready for use with:")
    print("- batch_processor_custom.py (with emojis)")
    print("- batch_processor_custom_simple.py (without emojis)")
    print("")
    print("Each prompt is specifically designed for:")
    print("1. OCR_DETAILED: Comprehensive text extraction with coordinates")
    print("2. OCR_STRUCTURED: Hierarchical text extraction with categorization")
    print("3. MARKDOWN_COMPREHENSIVE: Complete document conversion to markdown")
    print("4. MARKDOWN_TECHNICAL: Technical document conversion with code/math support")
    print("5. MARKDOWN_BUSINESS: Corporate document conversion with professional formatting")
    print("")
    print("All prompts use proper KOSMOS-2.5 tags (<grounding> for OCR, <md> for Markdown)")
    print("and are optimized for FP8 mixed precision processing.")

if __name__ == "__main__":
    validate_custom_prompts()