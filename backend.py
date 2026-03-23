"""
AUDIBLE Backend — AI Play Import
FastAPI service that interprets flag football play diagrams via GPT-4o vision.
"""

import os
import io
import re
import base64
import json
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audible-backend")

app = FastAPI(
    title="AUDIBLE Backend",
    description="AI Play Import — interprets flag football play diagrams via GPT-4o vision",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set — /interpret-play will fail at runtime")

VISION_PROMPT = """You are a football play diagram interpreter. Analyze this flag football play diagram and extract:
1. Player positions (offense and defense) as normalized x,y coordinates (0.0-1.0) from top-left
2. Routes/assignments as sequences of points
3. Suggest a play name, type (pass/run/defense/special), and format (5v5/7v7/both)
4. Write a brief description of what the play is designed to do

Return ONLY valid JSON matching this schema:
{
  "name": "string",
  "type": "pass" | "run" | "defense" | "special",
  "format": "5v5" | "7v7" | "both",
  "notes": "string",
  "players": [{"x": float, "y": float, "team": "offense" | "defense", "label": "string"}],
  "routes": [{"fromIndex": int, "points": [[float, float]], "type": "route" | "block" | "coverage"}]
}"""


def image_bytes_to_base64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("utf-8")


def pdf_first_page_to_png(pdf_bytes: bytes) -> bytes:
    """Extract first page of a PDF and render it as PNG bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")
    page = doc.load_page(0)
    # Render at 2x scale for better OCR/vision quality
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def call_vision_api(client: OpenAI, image_b64: str, mime_type: str) -> dict:
    """Send image to GPT-4o and parse the JSON response."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
        max_tokens=2048,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    logger.info("GPT-4o raw response: %s", raw[:300])

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON from GPT-4o: %s", e)
        raise ValueError(f"GPT-4o returned non-JSON: {raw[:200]}")


def validate_and_normalize(data: dict) -> dict:
    """Validate and sanitize the structured play data."""
    valid_types = {"pass", "run", "defense", "special"}
    valid_formats = {"5v5", "7v7", "both"}
    valid_route_types = {"route", "block", "coverage"}
    valid_teams = {"offense", "defense"}

    result = {
        "name": str(data.get("name", "Imported Play"))[:80],
        "type": data.get("type", "pass") if data.get("type") in valid_types else "pass",
        "format": data.get("format", "both") if data.get("format") in valid_formats else "both",
        "notes": str(data.get("notes", ""))[:500],
        "players": [],
        "routes": [],
    }

    raw_players = data.get("players", []) or []
    for p in raw_players[:20]:  # cap at 20 players
        try:
            x = float(p.get("x", 0.5))
            y = float(p.get("y", 0.5))
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            team = p.get("team", "offense") if p.get("team") in valid_teams else "offense"
            label = str(p.get("label", ""))[:10]
            result["players"].append({"x": x, "y": y, "team": team, "label": label})
        except (TypeError, ValueError):
            continue

    raw_routes = data.get("routes", []) or []
    for r in raw_routes[:30]:  # cap at 30 routes
        try:
            from_idx = int(r.get("fromIndex", 0))
            rtype = r.get("type", "route") if r.get("type") in valid_route_types else "route"
            raw_pts = r.get("points", []) or []
            points = []
            for pt in raw_pts[:50]:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px = max(0.0, min(1.0, float(pt[0])))
                    py = max(0.0, min(1.0, float(pt[1])))
                    points.append([px, py])
            if points:
                result["routes"].append({"fromIndex": from_idx, "points": points, "type": rtype})
        except (TypeError, ValueError):
            continue

    return result


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "AUDIBLE Backend", "version": "1.0.0"}


@app.post("/interpret-play")
async def interpret_play(file: UploadFile = File(...)):
    """
    Accept an image or PDF upload and return structured play data.
    Supports: image/jpeg, image/png, image/gif, image/webp, application/pdf
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    content_type = file.content_type or ""
    filename = file.filename or ""

    allowed_image_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    is_image = content_type in allowed_image_types or any(
        filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    )

    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Please upload an image (JPEG, PNG, WebP) or PDF.",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20 MB.")

    try:
        if is_pdf:
            logger.info("Converting PDF to image: %s", filename)
            image_bytes = pdf_first_page_to_png(raw_bytes)
            mime_type = "image/png"
        else:
            image_bytes = raw_bytes
            # Normalize MIME
            if content_type in allowed_image_types:
                mime_type = content_type
            else:
                ext = filename.lower().rsplit(".", 1)[-1]
                mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
                mime_type = mime_map.get(ext, "image/jpeg")

    except Exception as e:
        logger.error("File processing error: %s", e)
        raise HTTPException(status_code=422, detail=f"Could not process file: {str(e)}")

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        image_b64 = image_bytes_to_base64(image_bytes)
        logger.info("Calling GPT-4o vision for play interpretation (image size: %d bytes)", len(image_bytes))
        raw_data = call_vision_api(client, image_b64, mime_type)
    except ValueError as e:
        logger.error("Vision API parse error: %s", e)
        raise HTTPException(status_code=422, detail="Could not interpret this image. Try a clearer photo.")
    except Exception as e:
        logger.error("OpenAI API error: %s", e)
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")

    try:
        result = validate_and_normalize(raw_data)
    except Exception as e:
        logger.error("Validation error: %s", e)
        raise HTTPException(status_code=422, detail="Could not interpret this image. Try a clearer photo.")

    logger.info(
        "Play interpreted: name=%s, type=%s, players=%d, routes=%d",
        result["name"], result["type"], len(result["players"]), len(result["routes"])
    )
    return JSONResponse(content=result)


@app.post("/import-roster")
async def import_roster(file: UploadFile = File(...)):
    """
    Accept CSV, XLSX, PDF, or image roster files.
    Extract player data using GPT-4o or direct parsing.
    Returns structured list of players.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    content = await file.read()
    filename = file.filename.lower()

    # Guard: 10MB max
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    players = []

    # ── CSV / plain text ──────────────────────────────
    if filename.endswith('.csv') or file.content_type in ('text/csv', 'text/plain'):
        import csv
        text = content.decode('utf-8', errors='ignore')
        reader = csv.DictReader(io.StringIO(text))
        raw_rows = list(reader)

        # Use GPT to normalize headers and extract fields
        prompt = f"""You are a sports roster parser. Given this CSV data, extract player information.
Return a JSON array where each player object has these fields (use null if unknown):
- name (string, full name)
- number (string, jersey number)
- position (string, one of: QB, WR, RB, C, DB, LB, TE, K, or best match)
- format (string: "5v5", "7v7", or "both")
- status (string: "active", "injured", or "active" as default)
- notes (string, any extra info, or "")

CSV data:
{text[:3000]}

Return ONLY valid JSON array. No explanation."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
        players = json.loads(raw)

    # ── XLSX ──────────────────────────────────────────
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
            ws = wb.active
            rows = []
            for row in ws.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    rows.append([str(c) if c is not None else '' for c in row])

            text = '\n'.join([','.join(row) for row in rows])
            prompt = f"""You are a sports roster parser. Given this spreadsheet data, extract player information.
Return a JSON array where each player has: name, number, position (QB/WR/RB/C/DB/LB/TE or best match), format ("5v5"/"7v7"/"both"), status ("active"/"injured"), notes.
Use null for unknown fields. Default status to "active", format to "both".

Data:
{text[:3000]}

Return ONLY valid JSON array."""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
            players = json.loads(raw)
        except ImportError:
            raise HTTPException(status_code=400, detail="Excel support not available. Please upload a CSV instead.")

    # ── PDF ───────────────────────────────────────────
    elif filename.endswith('.pdf') or file.content_type == 'application/pdf':
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        if text.strip():
            # Text-based PDF
            prompt = f"""You are a sports roster parser. Extract all player data from this text.
Return a JSON array where each player has: name, number, position (QB/WR/RB/C/DB/LB/TE or best match), format ("5v5"/"7v7"/"both"), status ("active"), notes ("").
Default format to "both". Return ONLY valid JSON array.

Text:
{text[:3000]}"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000, temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
            players = json.loads(raw)
        else:
            # Scanned PDF — render first page as image and use vision
            page = fitz.open(stream=content, filetype="pdf")[0]
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "This is a scanned sports roster. Extract all player data and return a JSON array where each player has: name, number, position (QB/WR/RB/C/DB/LB/TE), format ('both'), status ('active'), notes (''). Return ONLY valid JSON array."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
                ]}],
                max_tokens=2000
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
            players = json.loads(raw)

    # ── IMAGE (photo of roster) ────────────────────────
    elif file.content_type and file.content_type.startswith('image/'):
        b64 = base64.b64encode(content).decode()
        ext = filename.split('.')[-1] if '.' in filename else 'jpeg'
        mime = f"image/{ext}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "This is a photo of a sports roster list. Extract all visible player information and return a JSON array where each player has: name (string), number (string or null), position (QB/WR/RB/C/DB/LB/TE or null), format ('both'), status ('active'), notes (''). Return ONLY valid JSON array, no explanation."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}}
            ]}],
            max_tokens=2000
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
        players = json.loads(raw)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Upload CSV, XLSX, PDF, or an image.")

    # Normalize and validate player objects
    valid_positions = {'QB','WR','RB','C','DB','LB','TE','K','ATH'}
    normalized = []
    for p in players:
        if not isinstance(p, dict): continue
        name = str(p.get('name') or '').strip()
        if not name or name.lower() in ('null','none',''): continue
        pos = str(p.get('position') or 'ATH').upper().strip()
        if pos not in valid_positions: pos = 'ATH'
        normalized.append({
            'name': name,
            'number': str(p.get('number') or '').strip() or None,
            'position': pos,
            'format': p.get('format') or 'both',
            'status': p.get('status') or 'active',
            'notes': str(p.get('notes') or '').strip()
        })

    return {"players": normalized, "count": len(normalized)}
