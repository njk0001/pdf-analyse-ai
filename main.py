import fitz  # PyMuPDF om PDF's te lezen
from transformers import pipeline

# Laad het samenvattingsmodel
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Fout bij het lezen van PDF: {e}"

def analyse_text(text):
    try:
        max_chunk = 1000
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        summaries = []
        for chunk in chunks[:3]:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        final_summary = "\n".join(summaries)
        return f"Samenvatting:\n{final_summary}"
    except Exception as e:
        return f"Fout bij samenvatten: {e}"

if __name__ == "__main__":
    pdf_file = "document.pdf"
    text = extract_text_from_pdf(pdf_file)
    if text:
        result = analyse_text(text)
        print(result)
    else:
        print("Geen tekst gevonden in de PDF.")
