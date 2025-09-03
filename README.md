Kosmos-2.5 Document Processing Tools
A collection of Python scripts for document processing using Microsoft's Kosmos-2.5 multimodal model. These tools enable OCR, markdown generation, and PDF-to-image conversion for document analysis workflows.

Features
OCR Processing: Extract text with bounding boxes from images
Markdown Generation: Convert document images to structured markdown
PDF Conversion: Convert PDF pages to individual images for processing
Batch Processing: Process multiple files efficiently
Flexible Input/Output: Support for local files and URLs
Prerequisites
Python 3.8+
CUDA-compatible GPU (optional, CPU fallback available)
Required Python packages (see Installation)
Installation
Clone this repository:
Install required dependencies:
For GPU support (optional):
Usage
1. OCR Processing (ocr.py)
Extract text and bounding boxes from images with visual annotations.

Output:

Annotated image with red bounding boxes around detected text
Optional text file with coordinates and extracted text
Console output with structured OCR results
2. Markdown Generation (md.py)
Convert document images to structured markdown format.

Output:

Clean markdown representation of the document structure
Console output with formatted markdown
3. PDF to Images (pdf2image.py)
Convert PDF documents to individual page images for further processing.

Output:

Folder named {pdf_name}_pages containing individual page images
Images named as page_001.png, page_002.png, etc.
Command Line Options
OCR Script (ocr.py)
Option	Short	Description	Default
--image	-i	Input image path or URL	Required
--output	-o	Output path for annotated image	./output.png
--text_output	-t	Output path for OCR text	None
--device	-d	Processing device	cuda:0
--max_tokens	-m	Maximum tokens to generate	1024
--no_bbox		Skip bounding box drawing	False
Markdown Script (md.py)
Option	Short	Description	Default
--image	-i	Input image path or URL	Required
--device	-d	Processing device	cuda:0
--max_tokens	-m	Maximum tokens to generate	1024
PDF Converter (pdf2image.py)
Option	Short	Description	Default
pdf_path		Path to PDF file	Required
--output	-o	Output directory	Same as PDF location
--dpi	-d	Image resolution	150
--format	-f	Image format (PNG/JPEG/TIFF)	PNG
Examples
Complete Document Processing Workflow
Convert PDF to images:
Process each page with OCR:
Generate markdown for structured pages:
Batch Processing Multiple Images
Output Formats
OCR Output
The OCR script generates text with bounding box coordinates in the format:

Markdown Output
The markdown script produces clean, structured markdown suitable for documentation or further processing.

Troubleshooting
Common Issues
CUDA not available:

Solution: Use --device cpu flag
The scripts automatically fall back to CPU if CUDA is unavailable
Model loading errors:

Ensure you have sufficient RAM/VRAM
Try using CPU mode for lower memory usage
Update transformers: pip install --upgrade transformers
PDF processing errors:

Install PyMuPDF: pip install PyMuPDF
Check PDF file permissions and corruption
Import errors:

Update dependencies: pip install --upgrade transformers torch pillow
Check Python version compatibility (3.8+)
Performance Tips
Use GPU for faster processing when available
Adjust --max_tokens based on document complexity
Use appropriate DPI settings for PDF conversion (150-300 DPI recommended)
Process images in batches to optimize memory usage
Requirements File
Create a requirements.txt file:

Install with: pip install -r requirements.txt

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing
Fork the repository
Create a feature branch (git checkout -b feature/new-feature)
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/new-feature)
Create a Pull Request
Acknowledgments
Microsoft for the Kosmos-2.5 model
Hugging Face for the transformers library
PyMuPDF team for PDF processing capabilities
