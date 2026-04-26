# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Data Loading Logic
# """

# import os
# from typing import List
# from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
# from langchain_core.documents import Document

# class DocumentLoader:
#     def __init__(self, data_path: str):
#         self.data_path = data_path

#     def load_documents(self) -> List[Document]:
#         """Loads all documents from the specified directory."""
#         print(f"[*] Loading documents from {self.data_path}...")
        
#         # Support for .txt
#         text_loader = DirectoryLoader(self.data_path, glob="**/*.txt", loader_cls=TextLoader)
#         # Support for .pdf
#         pdf_loader = DirectoryLoader(self.data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
#         docs = text_loader.load() + pdf_loader.load()
#         print(f"[+] Loaded {len(docs)} documents.")
#         return docs

# if __name__ == "__main__":
#     # Test loader
#     loader = DocumentLoader("data/sample_docs")
#     documents = loader.load_documents()
#     for doc in documents:
#         print(f"Source: {doc.metadata.get('source')}, Content snippet: {doc.page_content[:100]}...")




"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Data Loading Logic

CHANGES FROM ORIGINAL:
- Switched from PyPDFLoader (loads entire PDF into RAM) to PyMuPDFLoader
  which streams page-by-page, drastically reducing memory usage on large PDFs
- Added per-page metadata: page_number, total_pages, file_name
- Added lazy loading via load_lazy() generator — documents are yielded one
  at a time instead of all being held in memory simultaneously
- Added per-file error handling so one corrupt/password-protected PDF doesn't
  crash the entire pipeline
- Added support for scanned PDFs via a fallback flag (use_ocr)
- Added file size logging to help debug memory issues early
"""

import os
from pathlib import Path
from typing import List, Iterator
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, data_path: str, use_ocr: bool = False):
        self.data_path = Path(data_path)
        self.use_ocr = use_ocr

    def _load_pdf_lazy(self, pdf_path: Path) -> Iterator[Document]:
        """
        Streams a PDF page by page using PyMuPDF (fitz).
        Never holds the full document in RAM.
        Falls back to pytesseract OCR if use_ocr=True and page has no text.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Install pymupdf: pip install pymupdf")

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        print(f"  [loader] Opening PDF: {pdf_path.name} ({file_size_mb:.1f} MB)")

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            print(f"  [loader] Total pages: {total_pages}")

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text").strip()

                # Fallback to OCR if page is image-based and use_ocr enabled
                if not text and self.use_ocr:
                    text = self._ocr_page(page)

                if not text:
                    print(f"  [loader] Warning: Page {page_num + 1} has no extractable text, skipping.")
                    continue

                yield Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page_number": page_num + 1,
                        "total_pages": total_pages,
                        "file_size_mb": round(file_size_mb, 2),
                    }
                )

            doc.close()

        except Exception as e:
            print(f"  [loader] ERROR loading {pdf_path.name}: {e}")

    def _ocr_page(self, page) -> str:
        """Render page to image and run OCR using pytesseract."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img)
        except Exception as e:
            print(f"  [loader] OCR failed: {e}")
            return ""

    def _load_txt(self, txt_path: Path) -> Iterator[Document]:
        """Loads a .txt file as a single document."""
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                yield Document(
                    page_content=text,
                    metadata={
                        "source": str(txt_path),
                        "file_name": txt_path.name,
                        "page_number": 1,
                        "total_pages": 1,
                        "file_size_mb": round(txt_path.stat().st_size / (1024 * 1024), 2),
                    }
                )
        except Exception as e:
            print(f"  [loader] ERROR loading {txt_path.name}: {e}")

    def load_documents_lazy(self) -> Iterator[Document]:
        """
        Generator that yields Document objects one page at a time.
        Use this in the pipeline to avoid loading everything into RAM.
        """
        pdf_files = list(self.data_path.rglob("*.pdf"))
        txt_files = list(self.data_path.rglob("*.txt"))

        print(f"[*] Found {len(pdf_files)} PDF(s) and {len(txt_files)} TXT(s) in {self.data_path}")

        for pdf_path in pdf_files:
            yield from self._load_pdf_lazy(pdf_path)

        for txt_path in txt_files:
            yield from self._load_txt(txt_path)

    def load_documents(self) -> List[Document]:
        """
        Convenience method that collects all lazy documents into a list.
        WARNING: For very large PDFs, prefer load_documents_lazy() in your pipeline.
        """
        docs = list(self.load_documents_lazy())
        print(f"[+] Loaded {len(docs)} pages/documents total.")
        return docs


if __name__ == "__main__":
    loader = DocumentLoader("data/sample_docs")
    for doc in loader.load_documents_lazy():
        print(f"Source: {doc.metadata['file_name']} | Page: {doc.metadata['page_number']} | Chars: {len(doc.page_content)}")