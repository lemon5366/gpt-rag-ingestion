import logging
import os
import re
from io import BytesIO

import PyPDF2

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker
from ..exceptions import UnsupportedFormatError

class PdfAnalysisChunker(BaseChunker):
    """
    PdfAnalysisChunker is a class designed to split PDF document content into chunks based on the format and specific chunking criteria. The class leverages various LangChain splitters tailored for different content formats, ensuring accurate and efficient processing.

    Initialization:
    ---------------
    The PdfAnalysisChunker is initialized with the following parameters:
    - data (str): The document content to be chunked.
    """
    def __init__(self, data):
        super().__init__(data)
        self.max_chunk_size = int(os.getenv("NUM_TOKENS", "2048"))
        self.token_overlap = int(os.getenv("TOKEN_OVERLAP", "100"))
        self.minimum_chunk_size = int(os.getenv("MIN_CHUNK_SIZE", "100"))
        self.supported_formats = ['pdf']

    def get_chunks(self):
        """
        Splits the PDF document content into chunks based on the specified format and criteria.
        """
        if self.extension not in self.supported_formats:
            raise UnsupportedFormatError(f"[pdf_analysis_chunker] {self.extension} format is not supported")

        logging.info(f"[pdf_analysis_chunker][{self.filename}] Running get_chunks.")

        # analyze document from bytes
        document = self._analyze_document()
        # process document chunks
        
        logging.info(f"[pdf_analysis_chunker][{self.filename}] process document chunks.")
        chunks = self._process_document_chunks(document)
        return chunks

    def _process_document_chunks(self, document):
        """
        Processes the analyzed document content into manageable chunks.

        Args:
            document : The analyzed document content .

        Returns:
            list: A list of dictionaries, where each dictionary represents a processed chunk of the document content.

        The method performs the following steps:
        1. Prepares the document content for chunking.
        2. Splits the content into chunks using a chosen splitting strategy.
        3. Iterates through the chunks, determining their page numbers and creating chunk representations.
        4. Skips chunks that do not meet the minimum size requirement.
        5. Logs the number of chunks created and skipped.
        """
        chunks = []
        text_chunks = self._chunk_content(document)
        chunk_id = 0
        skipped_chunks = 0
        current_page = 1

        for text_chunk, num_tokens in text_chunks:
            current_page = self._update_page(text_chunk, current_page)
            chunk_page = self._determine_chunk_page(text_chunk, current_page)
            if num_tokens >= self.minimum_chunk_size:
                chunk_id += 1
                chunk = self._create_chunk(
                    chunk_id=chunk_id,
                    content=text_chunk,
                    page=chunk_page
                )
                chunks.append(chunk)
            else:
                skipped_chunks += 1

        logging.debug(f"[pdf_analysis_chunker][{self.filename}] {len(chunks)} chunk(s) created")
        if skipped_chunks > 0:
            logging.debug(f"[pdf_analysis_chunker][{self.filename}] {skipped_chunks} chunk(s) skipped")
        return chunks

    def _determine_chunk_page(self, content, current_page):
        """
        Determines the chunk page number based on the position of the PageBreak element.
        
        Args:
            content (str): The content chunk being processed.
            current_page (int): The current page number.
        
        Returns:
            int: The page number for the chunk.
        """
        match = re.search(r'Page(\d{5})', content)
        if match:
            page_number = int(match.group(1))
            position = match.start() / len(content)
            # Determine the chunk_page based on the position of the PageBreak element
            if position < 0.5:
                chunk_page = page_number + 1
            else:
                chunk_page = page_number
        else:
            chunk_page = current_page
        return chunk_page

    def _update_page(self, content, current_page):
        """
        Updates the current page number based on the content.
        
        Args:
            content (str): The content chunk being processed.
            current_page (int): The current page number.
        
        Returns:
            int: The updated current page number.
        """
        matches = re.findall(r'Page(\d{5})', content)
        if matches:
            page_number = int(matches[-1])
            if page_number >= current_page:
                current_page = page_number + 1
        return current_page

    def _analyze_document(self):
        """
        Analyzes the document using the pypdf2 library.
        """
        pdf_content = ""
        with BytesIO(self.document_bytes) as file:
            pdf_reader = PyPDF2.PdfReader(file)
            logging.info(f"[pdf_analysis_chunker]PDF has {len(pdf_reader.pages)} pages")
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_content = page.extract_text()
                    if page_content.strip():
                        # add page break
                        pdf_content += f"\n--- Page{page_num + 1} ---\n"
                        pdf_content += page_content
                        logging.info(f"[pdf_analysis_chunker]Extracted page {page_num + 1}")
                except Exception as e:
                    logging.error(f"Error extracting page {page_num + 1}: {e}")
                    continue

        return pdf_content.strip()

    def _chunk_content(self, content):
        """
        Splits the document content into chunks based on the specified pdf format .
        
        Yields:
            tuple: A tuple containing the chunked content and the number of tokens in the chunk.
        """
        logging.info(f"[pdf_analysis_chunker][{self.filename}] Start to chunk document")
        splitter = self._get_pdf_splitter()

        chunks = splitter.split_text(content)
        logging.info(f"[pdf_analysis_chunker][{self.filename}] Complete  chunking document, chunk size:{len(chunks)}")
        for chunked_content in chunks:
            chunk_size = self.token_estimator.estimate_tokens(chunked_content)
            yield chunked_content, chunk_size

    
    def _get_pdf_splitter(self):
        """
        Get appropriate splitter based on pdf format.
        
        Returns:
            object: The splitter to use for chunking.
        """
        separators = [".", "!", "?"] + [" ", "\n", "\t"]
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                separators=separators,
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.token_overlap
            )
    