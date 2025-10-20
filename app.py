from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from utils.rmbg_model import RMBGRemover

app = FastAPI(title="Background Remover (RMBG-1.4)")
remover = RMBGRemover()

@app.get("/")
def root():
    return {"message": "RMBG-1.4 Background Remover is running 🚀"}

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    image_bytes = await file.read()
    output = remover.remove_background(image_bytes)
    return StreamingResponse(output, media_type="image/png")
