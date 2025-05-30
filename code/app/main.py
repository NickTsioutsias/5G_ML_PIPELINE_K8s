from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import os

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": f"FROM: {os.environ.get('ENV', 'DEFAULT_ENV')}"}


@app.get("/download")
def stream_download():
    def generate():
        chunk = b"x" * 1024  # 1KB
        for _ in range(1024):  # 1024 * 1KB = 1MB total
            yield chunk

    return StreamingResponse(generate(), media_type="application/octet-stream")
