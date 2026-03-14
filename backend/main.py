from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from backend.api.routes import router
from backend.vector_store import vector_store
from config.settings import settings


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description='Enterprise AI Knowledge Assistant powered by RAG',
    docs_url='/docs',
    redoc_url='/redoc',
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.on_event('startup')
async def startup_event():
    logger.info(f'Starting {settings.app_name} v{settings.app_version}')
    vector_store.load()
    logger.success('Application startup complete')


@app.on_event('shutdown')
async def shutdown_event():
    logger.info('Shutting down application...')
    if vector_store.index is not None:
        vector_store.save()
    logger.info('Shutdown complete')


app.include_router(router, prefix='/api/v1')


@app.get('/')
async def root():
    return {
        'message': f'Welcome to {settings.app_name}',
        'version': settings.app_version,
        'docs': '/docs',
    }