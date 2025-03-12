import io
import base64
from typing import List
import pymupdf
from PIL import Image


def process_document(pdf_path: str) -> List[str]:
    """
    Convert PDF document pages to base64 encoded images.
    """
    image_urls = []
    pdf_document = pymupdf.open(pdf_path)
    
    # Process each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Render page to an image with higher resolution
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        image_urls.append(image_url)
    
    pdf_document.close()
    return image_urls