from PyPDF2 import PdfReader

def extract_chunks_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # ✅ avoid NoneType errors
            paragraphs = text.split('\n\n')
            for p in paragraphs:
                cleaned = p.strip()
                if len(cleaned) > 100:
                    chunks.append(cleaned)
    return chunks
