import os
import docx
import pdfplumber
import pandas as pd
import fitz

from datetime import datetime
from app.ingestion.schema import IngestedDocument
from app.utils.chunking import chunk_text
from app.ingestion.frame_captioner import generate_caption 
from app.utils.logger import get_logger
from docx import Document

# Logger
logger = get_logger(__name__)

# Table -> Text
def table_to_text(df):
    try:
        columns = " | ".join([str(col) for col in df.columns])
        rows = []

        for i, row in df.iterrows():
            row_text = " | ".join([str(cell) for cell in row])
            rows.append(f"Row {i+1}: {row_text}")

        table_text = "Table:\n"
        table_text += f"Columns: {columns}\n"
        table_text += "\n".join(rows)

        return table_text
    
    except Exception as e:
        logger.error(f"[TableParser] Failed to convert table | error ={str(e)}")
        return None

def ingest(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    source_type = ext.replace(".", "")

    text = ""
    structured_documents = []

    try:
        logger.info(f"[DocumentIngest] Starting ingestion | file={file_path}")

        # PDF
        if ext == ".pdf":
            

            doc = fitz.open(file_path)

            logger.info(
                f"[PDFIngest] Opened PDf | file={file_path} | pages={len(doc)}"
            )
           
            with pdfplumber.open(file_path) as pdf:

                for page_index in range(len(doc)):
                    page = doc[page_index]
                    logger.debug(f"[PDFIngest] Processing page = {page_index}")
                    
                    # 1. EXTRACT TEXT 
                    page_text = page.get_text()
                    if page_text:
                        text += page_text + "\n"
                        

                    # 2. EXTRACT IMAGES
                    image_list = page.get_images(full=True)
                    logger.info(
                        f"[PDFIngest] page = {page_index} | images_found={len(image_list)}"
                    )
                    

                    for img_index, img in enumerate(image_list):
                        try:
                            logger.debug(
                                f"[PDFIngest] Extracting image | page={page_index} | img_index={img_index}"
                            )
                            
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            

                            image_path = f"temp_pdf_image_{page_index}_{img_index}.png"

                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                            # Caption using BLIP pipeline
                            caption = generate_caption(image_path)

                            if not caption:
                                logger.warning(
                                    f"[PDFIngest] Empty caption | page={page_index} | img_index={img_index}"
                                )
                                continue


                            # CREATE IMAGE DOCUMENT
                            structured_documents.append(
                                IngestedDocument(
                                    text=caption,
                                    metadata={
                                        "source": os.path.basename(file_path),
                                        "source_type": source_type,
                                        "modality": "image",
                                        "page": page_index,
                                        "chunk_id": None,
                                        "element_index": img_index,
                                        "ingestion_time": datetime.utcnow().isoformat(),
                                    },
                                )
                            )
                            logger.info(
                                f"[PDFIngest] Image document created | page={page_index} | img_index={img_index}"
                            )
                            
                        except Exception:
                            logger.error(
                                f"[PDFIngest] Image extraction failed | page={page_index} | img_index={img_index} | error={str(e)}"
                            )
                            continue

                    # TABLES (PDF)
                    try:
                        page_plumber = pdf.pages[page_index]
                        tables = page_plumber.extract_tables()

                        if tables:
                            logger.info(
                                f"[PDFIngest] Tables found | page={page_index} | count={len(tables)}"
                            )

                        for table_index, table in enumerate(tables):
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                table_text = table_to_text(df)

                                if table_text:
                                    structured_documents.append(
                                        IngestedDocument(
                                            text=table_text,
                                            metadata={
                                                "source": os.path.basename(file_path),
                                                "source_type": source_type,
                                                "modality": "table",
                                                "page": page_index,
                                                "chunk_id": None,
                                                "element_index": table_index,
                                                "ingestion_time": datetime.utcnow().isoformat(),
                                            },
                                        )
                                    )
                                    logger.info(
                                        f"[PDFIngest] Table doc created | page={page_index} | table_index={table_index}"
                                    )
                            except Exception as e:
                                logger.error(
                                    f"[PDFIngest] Table parsing failed | error={str(e)}"
                                )
                    except Exception:
                        pass

                                
        # Word
        elif ext == ".docx":

            doc = docx.Document(file_path)

            # Text
            for para in doc.paragraphs:
                text += para.text + "\n"

            # Tables
            for table_index, table in enumerate(doc.tables):
                try:
                    data = []
                    for row in table.rows:
                        data.append([cell.text.strip() for cell in row.cells])

                    df = pd.DataFrame(data[1:], columns=data[0])
                    table_text = table_to_text(df)

                    if table_text:
                        structured_documents.append(
                            IngestedDocument(
                                text=table_text,
                                metadata={
                                    "source": os.path.basename(file_path),
                                    "source_type": source_type,
                                    "modality": "table",
                                    "page": None,
                                    "chunk_id": None,
                                    "element_index": table_index,
                                    "ingestion_time": datetime.utcnow().isoformat(),
                                },
                            )
                        )
                        logger.info(
                            f"[WordIngest] Table doc created | index={table_index}"
                        )

                except Exception as e:
                    logger.error(f"[WordIngest] Table parsing failed | error = {str(e)}")
            # WORD IMAGES
            try:
                rels = doc.part._rels
                image_index = 0

                for rel in rels:
                    rel = rels[rel]

                    if "image" in rel.target_ref:
                        try:
                            image_bytes = rel.target_part.blob
                            
                            image_path = f"temp_word_image_{image_index}.png"

                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                            caption = generate_caption(image_path)

                            if not caption:
                                image_index += 1
                                continue

                            structured_documents.append(
                                IngestedDocument(
                                    text=caption,
                                    metadata={
                                        "source": os.path.basename(file_path),
                                        "source_type": source_type,
                                        "modality": "image",
                                        "page": None,
                                        "chunk_id": None,
                                        "element_index": image_index,
                                        "ingestion_time": datetime.utcnow().isoformat(),
                                    },
                                )
                            )

                            logger.info(
                                f"[WordIngest] Image doc created | index={image_index}"
                            )

                            image_index +=1

                        except Exception as e:
                            logger.error(
                                f"[WordIngest] Image extraction failed | error= {str(e)}"
                            )
            except Exception as e:
                logger.error(
                    f"[WordIngest] Image extraction block failed | error={str(e)}"
                )

        # Excel
        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                df = pd.read_excel(file_path)

            table_text = table_to_text(df)

            if table_text:
                    structured_documents.append(
                    IngestedDocument(
                        text=table_text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "source_type": source_type,
                            "modality": "table",
                            "page": None,
                            "chunk_id": None,
                            "element_index": 0,
                            "ingestion_time": datetime.utcnow().isoformat(),
                        },
                    )
                )
                    logger.info("[ExcelIngest] Table doc created")
                    
        else:
            logger.error(f"[DocumentIngest] Unsupported file type | file={file_path}")
            raise ValueError("Unsupported document type")

    except Exception as e:
        logger.error(f"[DocumentIngest] Parsing failed | file={file_path} | error={str(e)}")
        raise ValueError(f"Document parsing failed: {str(e)}")

    # TEXT PROCESSING
    if not text.strip():
        logger.warning("[DocumentIngest] No text content extracted")
    
    metadata = {
        "source": os.path.basename(file_path),
        "source_type": source_type,
        "modality": "document",
        "page": None,
        "ingestion_time": datetime.utcnow().isoformat()
    }

    # Chunking
    chunks = chunk_text(text)

    text_documents =  [
        IngestedDocument(
            text=chunk,
            metadata={
                **metadata,
                  "chunk_id": i,
                  "element_index": None
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    # Combine 
    all_documents = text_documents + structured_documents

    logger.info(
        f"[DocumentIngest] Final documents | text={len(text_documents)} | structured={len(structured_documents)}"
    )

    return all_documents