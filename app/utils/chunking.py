from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=200,         # optimal for MiniLM
        chunk_overlap=100,      # preserves context
        separators=["\n\n", "\n", ".", " ",""]
    )

def chunk_text(text):
    splitter = get_text_splitter()
    chunks = splitter.split_text(text)

    return chunks