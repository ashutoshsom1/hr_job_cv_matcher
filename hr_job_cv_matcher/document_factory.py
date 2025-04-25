from pathlib import Path

import chainlit
from hr_job_cv_matcher.pdf_to_text_client import extract_text_from_pdf
from langchain.schema import Document

from hr_job_cv_matcher.config import cfg
from hr_job_cv_matcher.log_init import logger



def convert_to_doc(file: chainlit.types.AskFileResponse) -> Document:
    path = write_to_temp_folder(file)
    # Store the absolute path to ensure we can find the file later
    source = str(path.absolute())
    content = extract_text_from_pdf(path)
    
    if content is None or content.strip() == "":
        logger.error(f"Failed to extract text from {path}")
        raise Exception(f"Could not extract text from {path}. Please ensure the PDF is text-based or properly scanned.")
    
    return Document(page_content=content, metadata={'source': source})



def write_to_temp_folder(file) -> Path:
    temp_doc_location = cfg.temp_doc_location
    new_path = temp_doc_location / file.name
    logger.info(f"new path: {new_path}")
    
    try:
        with open(new_path, "wb") as f:
            f.write(file.content)
        return new_path
    except Exception as e:
        logger.error(f"Failed to write file to {new_path}: {str(e)}")
        raise
