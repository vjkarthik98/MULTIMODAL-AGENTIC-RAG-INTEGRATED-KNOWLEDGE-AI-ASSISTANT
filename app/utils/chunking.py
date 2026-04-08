from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Logger
logger = logging.getLogger(__name__)


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )


def chunk_text(text):
    splitter = get_text_splitter()

    chunks = splitter.split_text(text)

    logger.debug(f"[Chunking] Text split into chunks | count={len(chunks)}")

    return chunks