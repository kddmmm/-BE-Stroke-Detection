import sys
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.openapi.utils import get_openapi
from app.ws.connection import websocket_router
from app.utils.webcam import WebcamManager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    WebcamManager.instance()
    yield
    WebcamManager.instance().release()

app = FastAPI(
    title="Stroke Detection Server",
    description="FAST 법칙 기반 뇌졸중 조기 감지 시스템 (WebSocket 통신)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan 
)

@app.get("/debug-openapi")
def debug_openapi():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("\n===== FASTAPI EXCEPTION =====\n", tb_str)
    return PlainTextResponse(f"Internal server error:\n{tb_str}", status_code=500)

app.include_router(websocket_router)
