from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import fitz
import requests
from lxml import etree
from ..config.settings import settings

@dataclass
class PaperSection:
    title: str
    content: str
    section_type: str


@dataclass
class ExtractedPaper:
    title: str
    abstract: str
    authors: List[str]
    sections: List[PaperSection]
    full_text: str
    metadata: Dict[str, str]


class PDFExtractor:

    def __init__(self):
        self.grobid_url = f"http://{settings.grobid.host}:{settings.grobid.port}"
        self.logger = logging.getLogger(__name__)

    def extract_with_grobid(self, pdf_path: Path) -> Optional[str]:
        try:
            with open(pdf_path, 'rb') as pdf_file:
                response = requests.post(
                    f"{self.grobid_url}/api/processFulltextDocument",
                    files = {'input': pdf_file},
                    timeout = settings.grobid.timeout
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            self.logger.error(f"Grobid extraction failed: {e}")
            return None

    def extract_with_pymupdf(self, pdf_path: Path) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def parse_grobid_xml(self, xml_contentL: str) -> ExtractedPaper:
        root = etree.fromstring(xml_content.encode())

        title_elem = root.xpath(".//titleStmt/title")[0]
        title = title_elem.text if title_elem is not None else "Unknown Title"

        abstract_elem = root.xpath('.//abstract/p')
        abstract = ' '.join([p.text or '' for p in abstract_elem])

        authors = []
        author_elem = root.xpath('.//author/persName')
        for author in author_elems:
            forename = author.xpath('.//forename[@type = "first"]')
            surname = author.xpath('.//surname')
            if forename and surname:
                name = f"{forename[0].text} {surname[0].text}"
                authors.append(name)

        sections = []
        div_elems = root.xpath('.//div[@type = "section"]')
        for div in div_elems:
            section_title = div.xpath('.//head')[0].text if div.xpath('//head') else "Untitled"
            section_content = ' '.join([p.text or '' for p in div.xpath('.//p')])
            section_type = self._classify_section(section_title.lower())

            sections.append(PaperSection(
                title = section_title,
                content = section_content,
                section_type = section_type
            ))

        full_text = abstract + ' ' + ' '.join([s.content for s in sections])

        return ExtractedPaper(
            title = title,
            abstract = abstract,
            authors = authors,
            sections = sections,
            full_text = full_text,
            metadata = {"extraction_method": "grobid"}
        )

    def _classify_section(self, title: str) -> str:
        if any(word in title for word in ['abstract']):
            return 'abstract'
        elif any(word in title for word in ['introduction', 'intro']):
            return 'introduction'
        elif any(word in title for word in ['method', 'methodology', 'approach']):
            return 'methods'
        elif any(word in title for word in ['result', 'finding', 'experiment']):
            return 'results'
        elif any(word in title for word in ['conclusion', 'summary', 'discussion']):
            return 'conclusion'
        elif any(word in title for word in ['related work', 'background', 'literature']):
            return 'background'
        else:
            return 'other'

    def extract_paper(self, pdf_path: Path) -> ExtractedPaper:
        grobid_xml = self.extract_with_grobid(pdf_path)

        if grobid_xml:
            try:
                return self.parse_grobid_xml(gorbid_xml)
            except Exception as e:
                self.logger.warning(f"Grobid parsing failed: {e}, falling back to PyMuPDF")

            text = self.extract_with_pymupdf(pdf_path)
            return ExtractedPaper(
                title = pdf_path.stem,
                abstract = "",
                authors = [],
                sections = []
                full_text = text,
                metadata = {"extraction_method": "pymupdf"}
            )