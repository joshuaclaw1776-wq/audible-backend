"""
AUDIBLE Backend — AI Play Import + Multi-User Auth
FastAPI service that interprets flag football play diagrams via GPT-4o vision
and provides multi-user authentication/team management.
"""

import os
import io
import re
import base64
import json
import logging
import sqlite3
import hashlib
import time
import uuid
from typing import Optional

import jwt
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audible-backend")

app = FastAPI(
    title="AUDIBLE Backend",
    description="AI Play Import + Multi-User Auth — Flag Coach IQ",
    version="2.0.0",
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

JWT_SECRET = os.getenv("JWT_SECRET", "flagcoachiq-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30
DB_PATH = os.getenv("DB_PATH", "flagcoachiq.db")


# ─────────────────────────────────────────────
#  DATABASE INIT
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS teams (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            team_id TEXT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'assistant',
            created_at INTEGER,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        );

        CREATE TABLE IF NOT EXISTS invites (
            id TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL,
            token TEXT UNIQUE NOT NULL,
            used INTEGER DEFAULT 0,
            created_at INTEGER,
            expires_at INTEGER
        );

        CREATE TABLE IF NOT EXISTS activity_log (
            id TEXT PRIMARY KEY,
            team_id TEXT,
            user_id TEXT,
            user_name TEXT,
            action TEXT,
            item_type TEXT,
            item_name TEXT,
            timestamp INTEGER
        );

        CREATE TABLE IF NOT EXISTS team_data (
            id TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data_json TEXT NOT NULL,
            updated_by TEXT,
            updated_at INTEGER,
            UNIQUE(team_id, data_type)
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized: %s", DB_PATH)


init_db()


# ─────────────────────────────────────────────
#  JWT HELPERS
# ─────────────────────────────────────────────

def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + (JWT_EXPIRY_DAYS * 86400),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    payload = verify_token(token)
    user_id = payload.get("sub")
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(row)


def require_head_coach(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "head_coach":
        raise HTTPException(status_code=403, detail="Head Coach access required")
    return user


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def row_to_user(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"],
        "team_id": row["team_id"],
        "created_at": row["created_at"],
    }


def row_to_team(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
    }


# ─────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────

class RegisterBody(BaseModel):
    team_name: str
    name: str
    email: str
    password: str

class LoginBody(BaseModel):
    email: str
    password: str

class AcceptInviteBody(BaseModel):
    token: str
    name: str
    password: str

class SendInviteBody(BaseModel):
    email: str
    role: str  # 'coordinator' | 'assistant'

class TeamDataBody(BaseModel):
    data: list

class UpdateRoleBody(BaseModel):
    role: str

class UpdateTeamBody(BaseModel):
    name: str

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
#  ROUTES — HEALTH
# ─────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {"status": "ok", "service": "AUDIBLE Backend", "version": "2.0.0"}


# ─────────────────────────────────────────────
#  ROUTES — AUTH
# ─────────────────────────────────────────────

@app.post("/auth/register")
async def register(body: RegisterBody):
    """Create a new team and head coach account."""
    conn = get_db()
    # Check if email already exists
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (body.email.lower().strip(),)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="An account with that email already exists")

    now = int(time.time())
    team_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())

    conn.execute(
        "INSERT INTO teams (id, name, created_at) VALUES (?, ?, ?)",
        (team_id, body.team_name.strip(), now)
    )
    conn.execute(
        "INSERT INTO users (id, team_id, name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, team_id, body.name.strip(), body.email.lower().strip(), hash_password(body.password), "head_coach", now)
    )
    conn.commit()

    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    team_row = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,)).fetchone()
    conn.close()

    token = create_token(user_id)
    logger.info("New team registered: %s by %s", body.team_name, body.email)
    return {"token": token, "user": row_to_user(user_row), "team": row_to_team(team_row)}


@app.post("/auth/login")
async def login(body: LoginBody):
    """Login with email + password."""
    conn = get_db()
    user_row = conn.execute("SELECT * FROM users WHERE email = ?", (body.email.lower().strip(),)).fetchone()
    if not user_row or user_row["password_hash"] != hash_password(body.password):
        conn.close()
        raise HTTPException(status_code=401, detail="Invalid email or password")

    team_row = conn.execute("SELECT * FROM teams WHERE id = ?", (user_row["team_id"],)).fetchone()
    conn.close()

    token = create_token(user_row["id"])
    logger.info("Login: %s", body.email)
    return {"token": token, "user": row_to_user(user_row), "team": row_to_team(team_row) if team_row else None}


@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info."""
    conn = get_db()
    team_row = conn.execute("SELECT * FROM teams WHERE id = ?", (current_user["team_id"],)).fetchone()
    conn.close()
    return {"user": current_user, "team": row_to_team(team_row) if team_row else None}


# ─────────────────────────────────────────────
#  ROUTES — INVITES
# ─────────────────────────────────────────────

@app.post("/invites/send")
async def send_invite(body: SendInviteBody, current_user: dict = Depends(require_head_coach)):
    """Send an invite (Head Coach only). Returns the join URL."""
    if body.role not in ("coordinator", "assistant"):
        raise HTTPException(status_code=400, detail="Role must be 'coordinator' or 'assistant'")

    conn = get_db()
    now = int(time.time())
    invite_id = str(uuid.uuid4())
    token = str(uuid.uuid4()).replace("-", "")
    expires_at = now + (7 * 86400)  # 7 days

    conn.execute(
        "INSERT INTO invites (id, team_id, email, role, token, used, created_at, expires_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
        (invite_id, current_user["team_id"], body.email.lower().strip(), body.role, token, now, expires_at)
    )
    conn.commit()
    conn.close()

    join_url = f"https://flagcoachiq.app/join?token={token}"
    logger.info("Invite sent for %s (role: %s) by %s", body.email, body.role, current_user["email"])
    return {"invite_token": token, "join_url": join_url, "email": body.email, "role": body.role}


@app.post("/invites/accept")
async def accept_invite(body: AcceptInviteBody):
    """Accept an invite and create an account."""
    conn = get_db()
    invite = conn.execute(
        "SELECT * FROM invites WHERE token = ? AND used = 0",
        (body.token,)
    ).fetchone()

    if not invite:
        conn.close()
        raise HTTPException(status_code=404, detail="Invalid or expired invite token")

    now = int(time.time())
    if invite["expires_at"] < now:
        conn.close()
        raise HTTPException(status_code=410, detail="This invite has expired")

    # Check if email already registered
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (invite["email"],)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO users (id, team_id, name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, invite["team_id"], body.name.strip(), invite["email"], hash_password(body.password), invite["role"], now)
    )
    conn.execute("UPDATE invites SET used = 1 WHERE id = ?", (invite["id"],))
    conn.commit()

    user_row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    team_row = conn.execute("SELECT * FROM teams WHERE id = ?", (invite["team_id"],)).fetchone()
    conn.close()

    token = create_token(user_id)
    logger.info("Invite accepted by %s for team %s", invite["email"], invite["team_id"])
    return {"token": token, "user": row_to_user(user_row), "team": row_to_team(team_row)}


@app.get("/invites/list")
async def list_invites(current_user: dict = Depends(require_head_coach)):
    """List all pending invites for the team."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM invites WHERE team_id = ? AND used = 0 ORDER BY created_at DESC",
        (current_user["team_id"],)
    ).fetchall()
    conn.close()
    return {"invites": [dict(r) for r in rows]}


@app.delete("/invites/{invite_id}")
async def revoke_invite(invite_id: str, current_user: dict = Depends(require_head_coach)):
    """Revoke a pending invite."""
    conn = get_db()
    conn.execute(
        "DELETE FROM invites WHERE id = ? AND team_id = ?",
        (invite_id, current_user["team_id"])
    )
    conn.commit()
    conn.close()
    return {"ok": True}


# ─────────────────────────────────────────────
#  ROUTES — TEAM DATA
# ─────────────────────────────────────────────

VALID_DATA_TYPES = {"plays", "players", "games", "practices", "notes", "customDrills"}


@app.get("/data/{data_type}")
async def get_data(data_type: str, current_user: dict = Depends(get_current_user)):
    """Get team data for a given type."""
    if data_type not in VALID_DATA_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid data type. Must be one of: {', '.join(VALID_DATA_TYPES)}")

    conn = get_db()
    row = conn.execute(
        "SELECT data_json FROM team_data WHERE team_id = ? AND data_type = ?",
        (current_user["team_id"], data_type)
    ).fetchone()
    conn.close()

    if row:
        try:
            data = json.loads(row["data_json"])
        except Exception:
            data = []
    else:
        data = []

    return {"data": data, "data_type": data_type}


@app.post("/data/{data_type}")
async def save_data(data_type: str, body: TeamDataBody, current_user: dict = Depends(get_current_user)):
    """Save/overwrite team data for a given type. Logs activity."""
    if data_type not in VALID_DATA_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid data type. Must be one of: {', '.join(VALID_DATA_TYPES)}")

    now = int(time.time())
    conn = get_db()

    # Upsert team data
    conn.execute("""
        INSERT INTO team_data (id, team_id, data_type, data_json, updated_by, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_id, data_type) DO UPDATE SET
            data_json = excluded.data_json,
            updated_by = excluded.updated_by,
            updated_at = excluded.updated_at
    """, (str(uuid.uuid4()), current_user["team_id"], data_type, json.dumps(body.data), current_user["id"], now))

    # Log activity
    conn.execute(
        "INSERT INTO activity_log (id, team_id, user_id, user_name, action, item_type, item_name, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), current_user["team_id"], current_user["id"], current_user["name"],
         "saved", data_type, f"{len(body.data)} items", now)
    )
    conn.commit()
    conn.close()

    return {"ok": True, "data_type": data_type, "count": len(body.data)}


@app.get("/activity")
async def get_activity(current_user: dict = Depends(get_current_user)):
    """Get last 50 activity log entries for the team."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM activity_log WHERE team_id = ? ORDER BY timestamp DESC LIMIT 50",
        (current_user["team_id"],)
    ).fetchall()
    conn.close()
    return {"activity": [dict(r) for r in rows]}


# ─────────────────────────────────────────────
#  ROUTES — ADMIN
# ─────────────────────────────────────────────

@app.get("/admin/staff")
async def list_staff(current_user: dict = Depends(require_head_coach)):
    """List all team members (Head Coach only)."""
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM users WHERE team_id = ? ORDER BY created_at ASC",
        (current_user["team_id"],)
    ).fetchall()
    conn.close()
    return {"staff": [row_to_user(r) for r in rows]}


@app.patch("/admin/staff/{user_id}")
async def update_staff_role(user_id: str, body: UpdateRoleBody, current_user: dict = Depends(require_head_coach)):
    """Update a staff member's role (Head Coach only)."""
    if body.role not in ("head_coach", "coordinator", "assistant"):
        raise HTTPException(status_code=400, detail="Invalid role")
    if user_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot change your own role")

    conn = get_db()
    result = conn.execute(
        "UPDATE users SET role = ? WHERE id = ? AND team_id = ?",
        (body.role, user_id, current_user["team_id"])
    )
    conn.commit()
    conn.close()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Staff member not found")
    return {"ok": True}


@app.delete("/admin/staff/{user_id}")
async def remove_staff(user_id: str, current_user: dict = Depends(require_head_coach)):
    """Remove a staff member from the team (Head Coach only)."""
    if user_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="You cannot remove yourself")

    conn = get_db()
    result = conn.execute(
        "DELETE FROM users WHERE id = ? AND team_id = ?",
        (user_id, current_user["team_id"])
    )
    conn.commit()
    conn.close()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Staff member not found")
    return {"ok": True}


@app.patch("/admin/team")
async def update_team(body: UpdateTeamBody, current_user: dict = Depends(require_head_coach)):
    """Update team name (Head Coach only)."""
    conn = get_db()
    conn.execute("UPDATE teams SET name = ? WHERE id = ?", (body.name.strip(), current_user["team_id"]))
    conn.commit()
    team_row = conn.execute("SELECT * FROM teams WHERE id = ?", (current_user["team_id"],)).fetchone()
    conn.close()
    return {"team": row_to_team(team_row)}


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


@app.post("/analyze-play")
async def analyze_play(request: Request):
    """
    Analyze an offensive flag football play against defensive coverages.
    """
    import json as json_module
    body = await request.json()

    play_name = body.get('name', 'Unknown Play')
    play_type = body.get('type', 'pass')
    play_notes = body.get('notes', '')
    play_format = body.get('format', 'both')
    routes_desc = body.get('routes_description', '')

    system_prompt = """You are an expert flag football offensive coordinator with deep knowledge of all defensive coverages used in 5v5 and 7v7 flag football. 

You understand these defensive coverages:
- Man Coverage (Cover 0): Each defender locks onto one receiver. No help over top. Vulnerable to double moves, pick routes, rub routes, and speed mismatches.
- Cover 2: Two deep safeties splitting the field in half. 3-4 underneath zone defenders. Vulnerable to the seam route, post route, flood concepts, and corner routes. Strong against deep single routes.
- Cover 3: Three deep defenders (2 corners + 1 safety) each owning a third of the field. Underneath defenders play flat/hook zones. Vulnerable to flood concepts, 4-verticals, crossing routes, and curl-flat combos.
- Cover 4 (Quarters): Four defenders playing deep quarters coverage. Very deep prevent. Vulnerable to short and intermediate routes, screen passes, crossing routes, and anything underneath.
- Zone Blitz: One or more rushers sent while remaining defenders drop into zone. Vulnerable to hot routes, quick slants, screens, and bubble routes. Can disrupt timing.
- All-Out Blitz (Cover 0 Blitz): Max pressure, no help. Extremely vulnerable to quick releases, pre-snap reads, hot routes, and any receiver who beats their man.
- 2-3 Zone: Two deep, three underneath. Common in 5v5. Vulnerable to flood routes, verticals in the seams, and aggressive route running in the flats.
- 3-2 Zone: Three deep, two underneath. Vulnerable to short crossing routes, quick slants, and curl routes.
- Cover 1 (Man Free): Man coverage with one free safety in the middle. Vulnerable to pick routes, rub routes, crossers, and flood to the boundary away from the safety.
- Press Man: Defenders jam receivers at the line. Vulnerable to speed releases, fade routes, double moves, and any WR with a speed advantage.

**7v7-Specific Coverages:**
- Cover 2 Man (2 deep safeties + 5 man coverage underneath): Common 7v7 look. Vulnerable to crossing routes, rub/pick combos, and flood concepts that stress the safeties laterally.
- Tampa 2 (Cover 2 with MLB/center defender dropping to deep middle): Closes off the seam but leaves the flats and the boundary side open. Vulnerable to out routes, corner routes, and flooding to one side.
- Cover 6 (Quarter-Quarter-Half: Cover 4 to one side, Cover 2 to other): Disguise coverage that shows different looks pre-snap. Vulnerable to attacking the Cover 2 half with flood routes or the boundary corner route.
- 5-1 Zone (5 zone defenders, 1 rusher): Most common base 7v7 zone look. Vulnerable to flooding zones, clearing routes that occupy underneath defenders, and vertical routes into vacated zones.
- 5-2 Defense (5 zone defenders, 2 rushers): Aggressive with double rush. Vulnerable to hot routes, quick slants, bubble screens, and any pre-snap adjustments to attack the rushers' vacated zones.
- 4-3 Man (4 DBs in man, 3 rushers): Max pressure man. Extremely vulnerable to quick releases, pre-snap reads, pick routes, and screens. The triple rush creates openings if neutralized.
- 3-3-1 Defense (3 underneath zone, 3 mid zone, 1 deep safety): Popular 7v7 hybrid. Vulnerable to seam routes splitting the zones and deep over routes that stress the single safety.
- Cloud Coverage (corner plays off-zone, safety rotates to his side): Designed to protect against the deep ball. Vulnerable to fade routes, back-shoulder throws, and quick hitch routes in front of the soft corner.
- Bracket Coverage (two defenders bracketing the top WR): Takes away the star receiver. Vulnerable to leaving other receivers in single coverage — attack the other side of the field aggressively.

For each play analysis, consider:
- The play type (pass/run/defense/special)
- Route combinations and how they stress different coverage shells
- Formation strength/weakness vs coverage
- Flag football specific rules (no-run zones, QB can't run unless rusher crosses)
"""

    format_note = "This is a 7v7 flag football play — prioritize 7v7-specific coverages (Cover 2 Man, Tampa 2, Cover 6, 5-1 Zone, 5-2, 4-3 Man, 3-3-1, Cloud, Bracket) in your analysis." if play_format == "7v7" else "This is a 5v5 flag football play — focus on 5v5 coverages (Man, Cover 2, Cover 3, Zone Blitz, 2-3 Zone, 3-2 Zone, All-Out Blitz)." if play_format == "5v5" else "This play works in both 5v5 and 7v7 — analyze against the full range of flag football coverages including 7v7-specific looks."

    user_prompt = f"""Analyze this flag football offensive play:

Play Name: {play_name}
Play Type: {play_type}
Format: {play_format}
Format Note: {format_note}
Coach's Notes: {play_notes if play_notes else 'None provided'}
Route/Formation Description: {routes_desc if routes_desc else 'Standard formation'}

Provide a detailed coverage matchup analysis. Return ONLY valid JSON in this exact format:
{{
  "play_name": "{play_name}",
  "summary": "One sentence describing what this play is designed to do",
  "strong_against": [
    {{
      "coverage": "Coverage name",
      "rating": 5,
      "reason": "Specific explanation of why this play beats this coverage in flag football"
    }},
    {{
      "coverage": "Coverage name",
      "rating": 4,
      "reason": "Explanation"
    }},
    {{
      "coverage": "Coverage name",
      "rating": 3,
      "reason": "Explanation"
    }}
  ],
  "weak_against": [
    {{
      "coverage": "Coverage name",
      "rating": 2,
      "reason": "Why this coverage gives this play trouble"
    }},
    {{
      "coverage": "Coverage name",
      "rating": 1,
      "reason": "Why this is a difficult matchup"
    }}
  ],
  "adjustment": "Specific coaching tip: what route or formation tweak makes this play work against its weak coverages",
  "ideal_situation": "Best game situation to call this play (down, distance, field position, score scenario)",
  "danger_coverage": "The single most dangerous coverage to face with this play",
  "best_matchup": "The single best coverage to attack with this play"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE)
        analysis = json_module.loads(raw)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI analysis unavailable: {str(e)}")
