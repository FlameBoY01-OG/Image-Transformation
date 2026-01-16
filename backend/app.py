from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from nst_engine import run_style_transfer

app = FastAPI()

# allow requests from any origin (needed for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "NeuroBrush API is active"}

@app.post("/transform")
async def transform(content: UploadFile = File(...), style: UploadFile = File(...)):
    print("Received images. Starting processing...")
    
    # read the uploaded images
    content_bytes = await content.read()
    style_bytes = await style.read()
    
    # do the actual style transfer
    result_bytes = run_style_transfer(content_bytes, style_bytes)
    
    # send back the transformed image
    return Response(content=result_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)