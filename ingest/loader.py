"""Step 1: load PDFs page by page, keeping filename and page number."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str
    page: int  # 1-based
    text: str


def load_pdfs(data_dir: Path) -> list[Page]:
    pages: list[Page] = []
    pdfs = sorted(data_dir.glob("**/*.pdf"))
    for pdf in pdfs:
        try:
            reader = PdfReader(pdf)
        except Exception as e:  # corrupt / encrypted files should not kill the run
            print(f"  ! skipping {pdf.name}: {e}")
            continue
        for i, p in enumerate(reader.pages, start=1):
            text = (p.extract_text() or "").strip()
            if text:
                pages.append(Page(source=str(pdf.relative_to(data_dir)), page=i, text=text))
    return pages
