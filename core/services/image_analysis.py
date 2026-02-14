"""
Image Analysis Service — OCR-based misinformation detection
Extracts text from images using Tesseract OCR, then feeds into
the existing AI analysis pipeline.
"""

import logging
import os
import tempfile
from typing import Dict, Optional

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """Extract text from images using Tesseract OCR"""

    def __init__(self):
        if pytesseract is None:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
        if Image is None:
            raise ImportError("Pillow is required. Install with: pip install Pillow")

        # Auto-detect Tesseract path on Windows
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\patha\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    def preprocess_image(self, image: 'Image.Image') -> 'Image.Image':
        """Preprocess image for better OCR accuracy"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to grayscale
        image = image.convert('L')

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        # Scale up small images for better OCR
        width, height = image.size
        if width < 1000 or height < 1000:
            scale = max(1000 / width, 1000 / height, 1)
            if scale > 1:
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.LANCZOS)

        return image

    def extract_text(self, image_path: str) -> Dict:
        """
        Extract text from an image file using Tesseract OCR.
        Returns dict with extracted_text, confidence, and metadata.
        """
        try:
            # Open and preprocess image
            original = Image.open(image_path)
            processed = self.preprocess_image(original)

            # Run OCR with detailed data
            ocr_data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

            # Extract text
            extracted_text = pytesseract.image_to_string(processed).strip()

            # Calculate average confidence (filter out -1 which means no text detected)
            confidences = [int(c) for c in ocr_data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Get image metadata
            width, height = original.size
            word_count = len(extracted_text.split()) if extracted_text else 0

            return {
                'success': True,
                'extracted_text': extracted_text,
                'confidence': round(avg_confidence, 2),
                'word_count': word_count,
                'image_size': {'width': width, 'height': height},
                'image_format': original.format or 'Unknown',
            }

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                'success': False,
                'extracted_text': '',
                'confidence': 0,
                'error': str(e),
                'word_count': 0,
            }

    def extract_text_from_upload(self, uploaded_file) -> Dict:
        """
        Extract text from a Django UploadedFile object.
        Saves to temp file, runs OCR, then cleans up.
        """
        temp_path = None
        try:
            # Save uploaded file to temp location
            suffix = os.path.splitext(uploaded_file.name)[1] or '.png'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name

            # Extract text
            result = self.extract_text(temp_path)
            result['filename'] = uploaded_file.name
            result['file_size'] = uploaded_file.size
            return result

        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}")
            return {
                'success': False,
                'extracted_text': '',
                'confidence': 0,
                'error': str(e),
                'word_count': 0,
            }
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
