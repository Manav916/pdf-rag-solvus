import os
from typing import List, Dict
import time
from mistralai import Mistral


def perform_ocr(image_urls: List[str]) -> List[Dict[str, str]]:
    """
    Perform OCR on a list of base64 encoded images using Mistral API.
    """
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    ocr_model = "mistral-ocr-latest"
    ocr_results = []
    
    for image_url in image_urls:
        time.sleep(2)
        response = client.ocr.process(
            model=ocr_model,
            document={
                "type": "image_url",
                "image_url": image_url,
            }
        )
        
        ocr_results.append({
            'image_url': image_url,
            'ocr_content': response.pages[0].markdown
        })
    
    return ocr_results