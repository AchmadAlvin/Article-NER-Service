import sys
import os
import re
import html
from typing import List, Optional, Any


from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.ner_model import get_ner_pipeline


app = FastAPI(
    title="Article Named-Entity Recognition Service",
    description="Layanan ekstraksi entitas bernama dari judul dan isi berita dengan normalisasi dan confidence score.",
    version="1.1.0"
)


MAX_TITLE_LENGTH = 1000
MAX_BODY_LENGTH = 50000


class ArticleRequest(BaseModel):
    title: Any
    body: Any
    model: Optional[str] = "adapted"

class BatchArticleRequest(BaseModel):
    articles: Any


class EntityItem(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: Optional[float] = 0.90
    normalized: Optional[str] = None

class MetaInfo(BaseModel):
    entity_count: int
    model: str
    language: Optional[str] = "en"
    confidence_explanation: Optional[str] = (
        "Confidence score scale (0.0 to 1.0): "
        ">=0.95: deterministic domain/rule-based matches; "
        "0.85-0.94: high-probability in-vocabulary neural predictions; "
        "0.70-0.84: contextual out-of-vocabulary detections."
    )

class ExtractResponse(BaseModel):
    entities: List[EntityItem]
    meta: MetaInfo
    cleaned_text: Optional[str] = None

class BatchExtractResponse(BaseModel):
    total_articles: int
    results: List[ExtractResponse]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Bad Request",
            "detail": "Invalid request schema or malformed JSON data.",
            "errors": [err.get("msg") for err in exc.errors()]
        }
    )

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail
        }
    )


def strip_html_tags(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r'<[^>]+>', '', text)

    cleaned = html.unescape(cleaned)
    return cleaned

def validate_article_fields(title: Any, body: Any) -> tuple[str, str]:
    if not isinstance(title, str) or not isinstance(body, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both 'title' and 'body' must be string values."
        )


    clean_title = strip_html_tags(title).strip()
    clean_body = strip_html_tags(body).strip()


    if not clean_title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'title' cannot be empty or contain only whitespace or empty HTML tags."
        )


    if not clean_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'body' cannot be empty or contain only whitespace or empty HTML tags."
        )


    if len(clean_title) > MAX_TITLE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Field 'title' exceeds maximum length of {MAX_TITLE_LENGTH} characters."
        )


    if len(clean_body) > MAX_BODY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Field 'body' exceeds maximum length of {MAX_BODY_LENGTH} characters."
        )

    return clean_title, clean_body


@app.post("/extract-entities", response_model=ExtractResponse)
async def extract_entities_endpoint(req: ArticleRequest):
    clean_title, clean_body = validate_article_fields(req.title, req.body)


    model_name = req.model if req.model in ["baseline", "adapted"] else "adapted"


    combined_text = f"{clean_title}\n\n{clean_body}"


    pipeline = get_ner_pipeline()


    raw_entities = pipeline.extract_entities(combined_text, model_type=model_name)


    detected_lang = pipeline.detect_language(combined_text)


    entities = [EntityItem(**ent) for ent in raw_entities]


    if detected_lang == "zh":
        model_str = f"spacy-zh_core_web_sm-{model_name}"
    else:
        model_str = f"spacy-{pipeline.spacy_model_name}-{model_name}"


    meta = MetaInfo(
        entity_count=len(entities),
        model=model_str,
        language=detected_lang
    )


    return ExtractResponse(
        entities=entities,
        meta=meta,
        cleaned_text=combined_text
    )


@app.post("/extract-entities/batch", response_model=BatchExtractResponse)
async def extract_entities_batch_endpoint(req: BatchArticleRequest):
    if not isinstance(req.articles, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'articles' must be a list of article objects."
        )


    if len(req.articles) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'articles' list cannot be empty."
        )


    if len(req.articles) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded: maximum 50 articles per batch request."
        )

    pipeline = get_ner_pipeline()
    batch_results = []


    for item in req.articles:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each item in 'articles' must be a JSON object with 'title' and 'body'."
            )

        title = item.get("title")
        body = item.get("body")
        model = item.get("model", "adapted")


        valid_title, valid_body = validate_article_fields(title, body)
        model_name = model if model in ["baseline", "adapted"] else "adapted"
        combined_text = f"{valid_title}\n\n{valid_body}"


        raw_entities = pipeline.extract_entities(combined_text, model_type=model_name)
        detected_lang = pipeline.detect_language(combined_text)
        entities = [EntityItem(**ent) for ent in raw_entities]


        if detected_lang == "zh":
            batch_model_str = f"spacy-zh_core_web_sm-{model_name}"
        else:
            batch_model_str = f"spacy-{pipeline.spacy_model_name}-{model_name}"

        meta = MetaInfo(
            entity_count=len(entities),
            model=batch_model_str,
            language=detected_lang
        )
        batch_results.append(ExtractResponse(entities=entities, meta=meta, cleaned_text=combined_text))


    return BatchExtractResponse(
        total_articles=len(batch_results),
        results=batch_results
    )


static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)


app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()

    return "<h1>NER Service is Running.</h1>"
