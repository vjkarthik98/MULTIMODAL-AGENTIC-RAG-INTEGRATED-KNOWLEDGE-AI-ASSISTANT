from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "type": str(type(file))
    }