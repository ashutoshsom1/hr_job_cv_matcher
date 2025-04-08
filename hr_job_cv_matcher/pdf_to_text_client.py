import requests
from pathlib import Path
import json
from typing import Optional
from hr_job_cv_matcher.config import cfg
from hr_job_cv_matcher.log_init import logger

def extract_text_from_pdf(pdf: Path) -> Optional[str]:
    if not pdf.exists():
        raise Exception(f"File {pdf} does not exist.")
    
    try:
        with open(pdf, 'rb') as file:
            multipart_form_data = {
                'file': (pdf.name, file)
            }
            response = requests.post(cfg.remote_pdf_server, files=multipart_form_data)
            
            if response.status_code == 200:
                try:
                    json_response = json.loads(response.content)
                    if 'extracted_text' in json_response:
                        return json_response['extracted_text']
                    elif 'text' in json_response:  # fallback key
                        return json_response['text']
                    else:
                        logger.error(f"Unexpected response format: {json_response}")
                        return None
                except json.JSONDecodeError:
                    # If response is plain text
                    return response.text
            
            logger.error(f"Failed to extract PDF content. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None
