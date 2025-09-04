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
## PDF extraction with Grobid and PyMuPDF

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

    def parse_grobid_xml(self, xml_content: str) -> ExtractedPaper:
        root = etree.fromstring(xml_content.encode())
        
        # Define namespace for TEI documents
        namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # Extract title safely - check both locations
        title = "Unknown Title"
        
        # Try title in titleStmt first
        title_elems = root.xpath(".//tei:titleStmt/tei:title[@level='a'][@type='main']", namespaces=namespaces)
        if not title_elems:
            title_elems = root.xpath(".//tei:titleStmt/tei:title", namespaces=namespaces)
        if not title_elems:
            # Try title in analytic section
            title_elems = root.xpath(".//tei:analytic/tei:title[@level='a'][@type='main']", namespaces=namespaces)
        
        if title_elems and title_elems[0].text:
            title = title_elems[0].text

        # Extract abstract safely
        abstract_elems = root.xpath('.//tei:abstract//tei:p', namespaces=namespaces)
        abstract_parts = []
        for elem in abstract_elems:
            if elem.text:
                abstract_parts.append(elem.text)
            # Also get text from child elements
            for child in elem.itertext():
                if child.strip():
                    abstract_parts.append(child.strip())
        abstract = ' '.join(abstract_parts)

        # Extract authors safely
        authors = []
        author_elems = root.xpath('.//tei:author/tei:persName', namespaces=namespaces)
        for author in author_elems:
            forename_elems = author.xpath('.//tei:forename[@type="first"]', namespaces=namespaces)
            surname_elems = author.xpath('.//tei:surname', namespaces=namespaces)
            
            forename_text = forename_elems[0].text if forename_elems and forename_elems[0].text else ""
            surname_text = surname_elems[0].text if surname_elems and surname_elems[0].text else ""
            
            if forename_text or surname_text:
                name = f"{forename_text} {surname_text}".strip()
                if name:
                    authors.append(name)

        # Extract sections safely - from body
        sections = []
        div_elems = root.xpath('.//tei:body//tei:div[tei:head]', namespaces=namespaces)
        for div in div_elems:
            head_elems = div.xpath('.//tei:head', namespaces=namespaces)
            section_title = "Untitled"
            if head_elems and head_elems[0].text:
                section_title = head_elems[0].text
            
            # Extract all text content from paragraphs in this section
            p_elems = div.xpath('.//tei:p', namespaces=namespaces)
            section_parts = []
            for p in p_elems:
                # Get all text content including from child elements
                for text in p.itertext():
                    if text.strip():
                        section_parts.append(text.strip())
            
            section_content = ' '.join(section_parts)
            section_type = self._classify_section(section_title.lower())

            if section_content:  # Only add sections with content
                sections.append(PaperSection(
                    title=section_title,
                    content=section_content,
                    section_type=section_type
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
        # Extracting initially with Grobid
        grobid_xml = self.extract_with_grobid(pdf_path)

        if grobid_xml:
            try:
                return self.parse_grobid_xml(grobid_xml)
            except Exception as e:
                self.logger.warning(f"Grobid parsing failed: {e}, falling back to PyMuPDF")
            
        # If Grobid fails, fallback to PyMuPDF
        text = self.extract_with_pymupdf(pdf_path)
        return ExtractedPaper(
            title = pdf_path.stem,
            abstract = "",
            authors = [],
            sections = [],
            full_text = text,
            metadata = {"extraction_method": "pymupdf"}
        )