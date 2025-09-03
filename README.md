# Kosmos-2.5 Document Processing Tools

A collection of Python scripts for document processing using Microsoft's Kosmos-2.5 multimodal model. These tools enable OCR, markdown generation, and PDF-to-image conversion for document analysis workflows.

## Features

- **OCR Processing**: Extract text with bounding boxes from images
- **Markdown Generation**: Convert document images to structured markdown
- **PDF Conversion**: Convert PDF pages to individual images for processing
- **Batch Processing**: Process multiple files efficiently
- **Flexible Input/Output**: Support for local files and URLs

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, CPU fallback available)
- Required Python packages (see Installation)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd kosmos-2.5-tools
```

2. Install required dependencies:
```bash
pip install torch torchvision torchaudio
pip install transformers pillow requests
pip install PyMuPDF  # For PDF processing
```

3. For GPU support (optional):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. OCR Processing (`ocr.py`)

Extract text and bounding boxes from images with visual annotations.

```bash
# Basic OCR
python ocr.py --image "document.jpg"

# Custom output paths
python ocr.py --image "document.jpg" --output "annotated.png" --text_output "extracted_text.txt"

# Use CPU
python ocr.py --image "document.jpg" --device "cpu"

# Process from URL
python ocr.py --image "https://example.com/document.png" --output "result.png"

# Skip bounding box visualization
python ocr.py --image "document.jpg" --no_bbox
```

**Output:**
- Annotated image with red bounding boxes around detected text
- Optional text file with coordinates and extracted text
- Console output with structured OCR results

### 2. Markdown Generation (`md.py`)

Convert document images to structured markdown format.

```bash
# Generate markdown
python md.py --image "document.jpg"

# Custom settings
python md.py --image "document.jpg" --device "cpu" --max_tokens 2048

# Process from URL
python md.py --image "https://example.com/document.png"
```

**Output:**
- Clean markdown representation of the document structure
- Console output with formatted markdown

### 3. PDF to Images (`pdf2image.py`)

Convert PDF documents to individual page images for further processing.

```bash
# Convert PDF to images
python pdf2image.py document.pdf

# Custom output directory
python pdf2image.py document.pdf --output "output_folder"

# High resolution conversion
python pdf2image.py document.pdf --dpi 300

# Different image format
python pdf2image.py document.pdf --format JPEG
```

**Output:**
- Folder named `{pdf_name}_pages` containing individual page images
- Images named as `page_001.png`, `page_002.png`, etc.

## Command Line Options

### OCR Script (`ocr.py`)
| Option | Short | Description | Default |
|--------|--------|-------------|---------|
| `--image` | `-i` | Input image path or URL | Required |
| `--output` | `-o` | Output path for annotated image | `./output.png` |
| `--text_output` | `-t` | Output path for OCR text | None |
| `--device` | `-d` | Processing device | `cuda:0` |
| `--max_tokens` | `-m` | Maximum tokens to generate | `1024` |
| `--no_bbox` | | Skip bounding box drawing | False |

### Markdown Script (`md.py`)
| Option | Short | Description | Default |
|--------|--------|-------------|---------|
| `--image` | `-i` | Input image path or URL | Required |
| `--device` | `-d` | Processing device | `cuda:0` |
| `--max_tokens` | `-m` | Maximum tokens to generate | `1024` |

### PDF Converter (`pdf2image.py`)
| Option | Short | Description | Default |
|--------|--------|-------------|---------|
| `pdf_path` | | Path to PDF file | Required |
| `--output` | `-o` | Output directory | Same as PDF location |
| `--dpi` | `-d` | Image resolution | `150` |
| `--format` | `-f` | Image format (PNG/JPEG/TIFF) | `PNG` |

## Examples

### Complete Document Processing Workflow

1. **Convert PDF to images:**
```bash
python pdf2image.py report.pdf --output "extracted_pages" --dpi 300
```

2. **Process each page with OCR:**
```bash
python ocr.py --image "extracted_pages/report_pages/page_001.png" --output "ocr_results/page_001_ocr.png" --text_output "ocr_results/page_001_text.txt"
```

3. **Generate markdown for structured pages:**
```bash
python md.py --image "extracted_pages/report_pages/page_001.png" > "markdown_results/page_001.md"
```

### Batch Processing Multiple Images

```bash
# Process all images in a directory
for img in images/*.png; do
    python ocr.py --image "$img" --output "results/$(basename "$img" .png)_ocr.png"
done
```

## Output Formats

### OCR Output
The OCR script generates text with bounding box coordinates in the format:
```
x0,y0,x1,y0,x1,y1,x0,y1,extracted_text
```

### Markdown Output
The markdown script produces clean, structured markdown suitable for documentation or further processing.

## Troubleshooting

### Common Issues

1. **CUDA not available:**
   - Solution: Use `--device cpu` flag
   - The scripts automatically fall back to CPU if CUDA is unavailable

2. **Model loading errors:**
   - Ensure you have sufficient RAM/VRAM
   - Try using CPU mode for lower memory usage
   - Update transformers: `pip install --upgrade transformers`

3. **PDF processing errors:**
   - Install PyMuPDF: `pip install PyMuPDF`
   - Check PDF file permissions and corruption

4. **Import errors:**
   - Update dependencies: `pip install --upgrade transformers torch pillow`
   - Check Python version compatibility (3.8+)

### Performance Tips

- Use GPU for faster processing when available
- Adjust `--max_tokens` based on document complexity
- Use appropriate DPI settings for PDF conversion (150-300 DPI recommended)
- Process images in batches to optimize memory usage

## Requirements File

Create a `requirements.txt` file:
```txt
torch>=1.11.0
torchvision>=0.12.0
torchaudio>=0.11.0
transformers>=4.36.0
pillow>=9.0.0
requests>=2.25.0
PyMuPDF>=1.23.0
```

Install with: `pip install -r requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Acknowledgments

- Microsoft for the Kosmos-2.5 model
- Hugging Face for the transformers library
- PyMuPDF team for PDF processing capabilities
