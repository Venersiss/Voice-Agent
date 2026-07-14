"""
Multi-Company AI Voice Agent Bridge
====================================
FastAPI server bridging Twilio voice calls to the Gemini Multimodal Live API
for real-time AI-powered phone receptionist conversations.

ARCHITECTURE:
  Twilio Voice Call → POST /voice (returns TwiML with WebSocket stream URL)
  Twilio Media Stream (mulaw/8kHz) ←WebSocket→ FastAPI ←WebSocket→ Gemini Live (PCM/16kHz)

KEY FEATURES:
  - Multi-company: resolves Twilio "To" number → company identity via n8n webhook
  - Real-time audio streaming with automatic audio codec conversion
  - Barge-in support: caller speech interrupts AI playback
  - Voice Activity Detection (VAD) to reduce phone echo
  - Function calling: check_availability, book_appointment, end_call
  - Knowledge-grounded responses from 15 curated Q&A pairs
  - Timezone-aware appointment slot building (8AM-4PM, on the hour)
  - Auto-hangup on idle/silence, max-call-duration watchdog
  - Identity prefetch with timeout fallback for fast call pickup

NDA NOTE: Several implementation details have been modified or removed
to comply with confidentiality obligations. Specific webhook URLs, API
endpoint configurations, company names, calendars, email templates, and
certain business-logic sections (post-booking SMS workflows, internal
routing rules, multi-tier fallback strategies) have been replaced with
commented placeholders. The architecture, tool-calling pattern, audio
pipeline, and knowledge-base injection remain structurally accurate.

Tech Stack: Python 3.12, FastAPI, websockets, httpx, Google Gemini
Multimodal Live API (gemini-2.5-flash-native-audio), Twilio Voice + Media Streams
"""

import asyncio
import audioop
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from time import perf_counter

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Audio codec constants
# ---------------------------------------------------------------------------
TWILIO_RATE = 8000
GEMINI_IN_RATE = 16000
GEMINI_OUT_RATE = 24000
TWILIO_FRAME_MS = 20
TWILIO_FRAME_BYTES = 160
TWILIO_OUTBOUND_CHUNK_MS = 100
TWILIO_OUTBOUND_CHUNK_BYTES = int(TWILIO_RATE * TWILIO_OUTBOUND_CHUNK_MS / 1000)
BARGE_IN_RMS_THRESHOLD = 700
BARGE_IN_CONSECUTIVE_FRAMES = 3
BARGE_IN_CLEAR_COOLDOWN_MS = 350
INPUT_VAD_RMS_THRESHOLD = 500
INPUT_VAD_HANGOVER_FRAMES = 8
GEMINI_AUDIO_STREAM_END_IDLE_MS = int(os.getenv("GEMINI_AUDIO_STREAM_END_IDLE_MS", "1200"))


def twilio_mulaw_to_gemini_pcm(mulaw_bytes: bytes, resample_state) -> tuple[bytes, object]:
    """Convert Twilio mulaw/8kHz -> linear PCM16/16kHz for Gemini."""
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    pcm_16k, new_state = audioop.ratecv(pcm_8k, 2, 1, TWILIO_RATE, GEMINI_IN_RATE, resample_state)
    return pcm_16k, new_state


def gemini_pcm_to_twilio_mulaw(pcm_24k_bytes: bytes, resample_state) -> tuple[bytes, object]:
    """Convert Gemini PCM16/24kHz -> Twilio mulaw/8kHz."""
    pcm_8k, new_state = audioop.ratecv(pcm_24k_bytes, 2, 1, GEMINI_OUT_RATE, TWILIO_RATE, resample_state)
    mulaw_8k = audioop.lin2ulaw(pcm_8k, 2)
    return mulaw_8k, new_state


def chunk_bytes(data: bytes, chunk_size: int) -> tuple[list[bytes], bytes]:
    """Split bytes into fixed-size chunks, returning (complete_chunks, remainder)."""
    if chunk_size <= 0 or not data:
        return [], data
    full_length = len(data) - (len(data) % chunk_size)
    chunks = [data[idx : idx + chunk_size] for idx in range(0, full_length, chunk_size)]
    return chunks, data[full_length:]


# ---------------------------------------------------------------------------
# Configuration (all from environment variables)
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-native-audio")
GEMINI_API_VERSION = os.environ.get("GEMINI_API_VERSION", "v1beta")
GEMINI_WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    f"google.ai.generativelanguage.{GEMINI_API_VERSION}.GenerativeService.BidiGenerateContent"
)


def gemini_ws_url() -> tuple[str, str]:
    api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")
    return f"{GEMINI_WS_BASE}?key={api_key}", api_key


# NDA NOTE: Actual webhook URLs replaced with placeholders.
# The original system calls n8n workflows for company identity lookup,
# calendar availability checking, appointment booking, and SMS fallback.
N8N_WEBHOOK_IDENTITY_URL = os.environ.get("N8N_WEBHOOK_IDENTITY_URL", "")
N8N_WEBHOOK_AVAILABILITY_URL = os.environ.get("N8N_WEBHOOK_AVAILABILITY_URL", "")
N8N_WEBHOOK_BOOKING_URL = os.environ.get("N8N_WEBHOOK_BOOKING_URL", "")
N8N_WEBHOOK_PENDING_BOOKING_URL = os.environ.get("N8N_WEBHOOK_PENDING_BOOKING_URL", "")

# NDA NOTE: Twilio credentials managed via env vars
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

AUTO_HANGUP_IDLE_SECONDS = float(os.environ.get("AUTO_HANGUP_IDLE_SECONDS", "2.0"))
IDENTITY_PREFETCH_TIMEOUT_MS = max(int(os.environ.get("IDENTITY_PREFETCH_TIMEOUT_MS", "5000")), 4000)
ENABLE_LOCAL_BARGE_IN = os.environ.get("ENABLE_LOCAL_BARGE_IN", "false").lower() in {"1", "true", "yes", "on"}

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice-agent")

# ---------------------------------------------------------------------------
# Shared HTTP client
# ---------------------------------------------------------------------------
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))
    api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY", "")
    logger.info("=== AI Voice Agent Bridge started ===")
    logger.info("Gemini model: %s | key set: %s", GEMINI_MODEL, bool(api_key))
    logger.info("Webhooks configured: identity=%s availability=%s booking=%s",
                bool(N8N_WEBHOOK_IDENTITY_URL), bool(N8N_WEBHOOK_AVAILABILITY_URL), bool(N8N_WEBHOOK_BOOKING_URL))
    if not api_key:
        logger.error("GOOGLE_AI_STUDIO_KEY is not set")
    yield
    await http_client.aclose()
    logger.info("AI Voice Agent Bridge stopped")


app = FastAPI(title="AI Voice Agent Bridge", lifespan=lifespan)
call_store: dict[str, dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Function declarations for Gemini tool calling
# ---------------------------------------------------------------------------
CHECK_AVAILABILITY_DECLARATION = {
    "name": "check_availability",
    "description": "Check available appointment slots for a requested day. Call before offering times.",
    "parameters": {
        "type": "object",
        "properties": {
            "requested_day": {
                "type": "string",
                "description": "Day the caller requested: 'tomorrow', 'Wednesday', or ISO-8601 date like '2026-04-24'."
            }
        },
        "required": ["requested_day"],
    },
}

BOOK_APPOINTMENT_DECLARATION = {
    "name": "book_appointment",
    "description": (
        "Create a confirmed appointment after the caller is qualified, "
        "has chosen a valid slot, and has provided name, email, phone, address, and notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "caller_name": {"type": "string", "description": "Full name of the caller."},
            "caller_email": {"type": "string", "description": "Caller email address."},
            "caller_phone": {"type": "string", "description": "Phone number for confirmation."},
            "caller_address": {"type": "string", "description": "Home address of the caller."},
            "preferred_datetime": {"type": "string", "description": "Confirmed appointment date/time in ISO-8601 format."},
            "notes": {"type": "string", "description": "Any additional notes."},
        },
        "required": ["caller_name", "caller_email", "caller_phone", "caller_address", "preferred_datetime"],
    },
}

END_CALL_DECLARATION = {
    "name": "end_call",
    "description": "Call after saying goodbye to end the conversation. System will hang up.",
    "parameters": {"type": "object", "properties": {}},
}

# ---------------------------------------------------------------------------
# Industry-specific knowledge base (15 curated Q&A pairs)
# ---------------------------------------------------------------------------
# NDA NOTE: The actual Q&A pairs covered industry-specific topics relevant to our
# client's domain. Below are structurally equivalent demonstrative entries.
# The original knowledge base contained 15 pairs covering: pricing, eligibility,
# installation, warranties, tax credits, maintenance, and common objections.
#
# This knowledge is injected into the system prompt and the AI answers using a
# "Helpful Expert" tone (not pushy salesperson), always pivoting back to
# qualification or booking.

KNOWLEDGE_GUIDANCE = """
Approved Q&A guidance — use these answers as a guide:

Q1: Is this really free? I keep seeing ads for no-cost programs.
A: There is no such thing as "free" equipment. However, there are programs where
   you can start for $0 down. We specialize in helping you swap your monthly bill
   for a lower payment so you see immediate savings.

Q2: What happens if I move in a few years?
A: The system adds value to your home. Most warranties and financing agreements
   are fully transferable to the new homeowner.
[... additional Q&A pairs covering common questions ...]
Q15: What's the absolute lowest price you can do?
A: The "cheapest" system often costs more in repairs. We focus on best
   value-to-savings ratio. Let's get you a custom report.

How to use this knowledge:
- Answer briefly in 1-3 sentences.
- Use "Helpful Expert" tone, not "Pushy Salesperson."
- After answering, pivot back to qualification or booking.
"""

# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(company: dict) -> str:
    """Build the live voice prompt from company data (resolved via n8n webhook)."""
    company_name = company.get("company_name", "our company")
    agent_name = company.get("agent_name") or "Alex"
    calendar_id = company.get("calendar_id", "primary")
    tz_name = (company.get("timezone") or "America/Los_Angeles").strip()

    now_utc = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        now_local = now_utc.astimezone(ZoneInfo(tz_name))
        current_dt = now_local.strftime("%A, %B %d, %Y at %I:%M %p")
    except Exception:
        current_dt = now_utc.strftime("%A, %B %d, %Y at %I:%M %p UTC")

    prompt = f"""\
You are {agent_name}, a warm, confident appointment setter for {company_name}.
Today's date and time in {tz_name}: {current_dt}
Calendar ID: {calendar_id}

Core rules:
- Speak like a real human receptionist. Never mention AI, prompts, or system messages.
- Keep every turn short and phone-friendly.
- Ask exactly one question at a time.
- Use natural fillers instead of silent pauses.
- If the caller interrupts, stop and respond to the new input.
- The caller called YOU. Never say "I am calling."

Qualification flow (ask one question at a time):
  1. Are you the homeowner?
  2. Is your monthly bill over $100?
  3. Does your property get direct/plenty of sunlight?
  4. Is your credit score above 650?

Before offering times, call check_availability.
- Only offer slots returned by check_availability, between 8 AM and 4 PM.
- Never invent appointment times.
- Appointments must be on the hour.

When collecting email, always ask the caller to spell it out. Confirm it back.
After successful booking, briefly confirm and say goodbye, then call end_call.
If booking returns an error saying it was already done, say goodbye and call end_call.

Knowledge base:
{KNOWLEDGE_GUIDANCE}
"""
    return prompt


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def now_ms() -> float:
    return perf_counter() * 1000.0


def elapsed_ms(start_ms: float | None) -> float | None:
    if start_ms is None:
        return None
    return round(now_ms() - start_ms, 1)


def resolve_requested_day(requested_day: str, tz_name: str) -> datetime.date:
    """Resolve natural-language day input ('tomorrow', 'Wednesday') to a concrete local date."""
    requested_day = (requested_day or "").strip().lower()
    try:
        from zoneinfo import ZoneInfo
        now_local = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name))
    except Exception:
        now_local = datetime.now(timezone.utc)

    target_date = now_local.date() + timedelta(days=1)
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    if requested_day == "today":
        target_date = now_local.date()
    elif requested_day == "tomorrow" or not requested_day:
        target_date = now_local.date() + timedelta(days=1)
    elif requested_day in weekdays:
        target_wd = weekdays.index(requested_day)
        days_ahead = (target_wd - now_local.weekday()) % 7 or 7
        target_date = now_local.date() + timedelta(days=days_ahead)
    else:
        try:
            target_date = datetime.fromisoformat(requested_day).date()
        except Exception:
            pass
    return target_date


def build_candidate_slots(requested_day: str, tz_name: str) -> list[dict]:
    """Build 8AM-4PM hourly candidate slots for availability checking."""
    target_date = resolve_requested_day(requested_day, tz_name)
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = None

    slots = []
    for hour in range(8, 17):
        if tz:
            start_dt = datetime(target_date.year, target_date.month, target_date.day, hour, 0, tzinfo=tz)
        else:
            start_dt = datetime(target_date.year, target_date.month, target_date.day, hour, 0)
        end_dt = start_dt + timedelta(hours=1)
        slots.append({
            "label": start_dt.strftime("%A, %B %d at %I:%M %p"),
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        })
    return slots


def validate_booking_args(args: dict, tz_name: str) -> dict | None:
    """Return error payload when required booking fields are missing or malformed."""
    required = ["caller_name", "caller_email", "caller_phone", "caller_address", "preferred_datetime"]
    missing = [f for f in required if not args.get(f)]
    if missing:
        return {"status": "error", "message": f"Missing fields: {', '.join(missing)}", "retryable": True}

    try:
        parsed = datetime.fromisoformat(args["preferred_datetime"].replace("Z", "+00:00"))
    except Exception:
        return {"status": "error", "message": "preferred_datetime must be valid ISO-8601", "retryable": True}

    if parsed.minute != 0 or parsed.second != 0 or parsed.microsecond != 0:
        return {"status": "error", "message": "Appointments must start on the hour", "retryable": True}
    if parsed.hour < 8 or parsed.hour > 16:
        return {"status": "error", "message": "Appointments must be between 8 AM and 4 PM", "retryable": True}
    if "@" not in args["caller_email"]:
        return {"status": "error", "message": "Invalid email address", "retryable": True}

    return None


# ---------------------------------------------------------------------------
# External API integrations
# ---------------------------------------------------------------------------
async def fetch_company_identity(to_number: str) -> dict:
    """Resolve a phone number to a company record via n8n webhook."""
    # NDA NOTE: The actual implementation calls an n8n webhook that queries
    # a Baserow database to find the company matching the dialed Twilio number.
    # The webhook returns company name, agent name, calendar ID, timezone, and
    # custom instructions. Falls back to a generic identity on failure.
    if not N8N_WEBHOOK_IDENTITY_URL:
        return {"company_name": "our company", "agent_name": "Alex"}

    try:
        resp = await http_client.post(N8N_WEBHOOK_IDENTITY_URL, json={"phone_number": to_number})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("Identity fetch failed: %s", exc)
        return {"company_name": "our company", "agent_name": "Alex"}


async def execute_check_availability(args: dict, company: dict) -> dict:
    """Return available appointment slots via n8n -> Google Calendar."""
    requested_day = args.get("requested_day", "").strip().lower()
    tz_name = (company.get("timezone") or "America/Los_Angeles").strip()
    candidate_slots = build_candidate_slots(requested_day, tz_name)

    if N8N_WEBHOOK_AVAILABILITY_URL:
        payload = {
            "requested_day": requested_day,
            "timezone": tz_name,
            "calendar_id": company.get("calendar_id", "primary"),
            "slots": candidate_slots,
        }
        try:
            resp = await http_client.post(N8N_WEBHOOK_AVAILABILITY_URL, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("Availability webhook failed: %s", exc)

    return {"available": True, "slots": [s["label"] for s in candidate_slots], "timezone": tz_name}


async def execute_book_appointment(args: dict, company: dict, call_sid: str = "") -> dict:
    """Forward booking payload to n8n (Google Calendar event + CRM + email confirmation)."""
    # NDA NOTE: The actual webhook validates, double-checks calendar availability,
    # creates a Google Calendar event, writes to CRM database, and sends confirmation
    # email via Gmail. Duplicate booking prevention and SMS fallback logic removed.
    tz_name = (company.get("timezone") or "America/Los_Angeles").strip()
    validation_error = validate_booking_args(args, tz_name)
    if validation_error:
        return validation_error

    if not N8N_WEBHOOK_BOOKING_URL:
        return {"status": "success", "message": "Appointment noted (webhook not configured)."}

    payload = {
        **args,
        "call_id": call_sid,
        "company_name": company.get("company_name", ""),
        "calendar_id": company.get("calendar_id", "primary"),
        "timezone": tz_name,
        "agent_name": company.get("agent_name", "Alex"),
    }
    try:
        resp = await http_client.post(N8N_WEBHOOK_BOOKING_URL, json=payload)
        return resp.json() if resp.status_code < 400 else {"status": "error", "message": "Booking failed"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "retryable": True}


async def send_twilio_clear(twilio_ws: WebSocket, stream_sid: str | None):
    """Flush Twilio's playback buffer to support barge-in."""
    if stream_sid:
        await twilio_ws.send_json({"event": "clear", "streamSid": stream_sid})


async def send_twilio_mark(twilio_ws: WebSocket, stream_sid: str | None, mark_name: str):
    """Tag sent audio so we know what was flushed by clear."""
    if stream_sid:
        await twilio_ws.send_json({"event": "mark", "streamSid": stream_sid, "mark": {"name": mark_name}})


async def twilio_hangup_call(call_sid: str) -> bool:
    """End an in-progress Twilio call via REST API."""
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not call_sid:
        return False
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls/{call_sid}.json"
    try:
        resp = await http_client.post(url, data={"Status": "completed"},
                                       auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        return 200 <= resp.status_code < 300
    except Exception as exc:
        logger.error("Twilio hangup exception for call %s: %s", call_sid, exc)
        return False


# ---------------------------------------------------------------------------
# Twilio inbound call handler
# ---------------------------------------------------------------------------
@app.post("/voice")
async def twilio_voice(request: Request):
    """Handle inbound Twilio voice call — return TwiML with WebSocket stream URL."""
    form = await request.form()
    to_number = form.get("To", "")
    from_number = form.get("From", "")
    call_sid = form.get("CallSid", "unknown")
    logger.info("Inbound call %s from %s to %s", call_sid, from_number, to_number)

    call_store[call_sid] = {"to_number": to_number, "from_number": from_number}

    host = request.headers.get("x-forwarded-host", request.headers.get("host", "localhost"))
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" else "ws"
    ws_url = f"{scheme}://{host}/ws/media/{call_sid}"

    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{ws_url}" />'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "ai-voice-agent-bridge"})


# ---------------------------------------------------------------------------
# WebSocket: Twilio Media Stream <-> Gemini Live (core real-time bridge)
# ---------------------------------------------------------------------------
@app.websocket("/ws/media/{call_sid}")
async def media_stream(ws: WebSocket, call_sid: str):
    """Bridge Twilio media to Gemini Live with low-latency audio streaming."""
    await ws.accept()
    call_context = call_store.pop(call_sid, {})
    to_number = call_context.get("to_number", "")
    from_number = call_context.get("from_number", "")
    logger.info("WS connected for call %s (from=%s to=%s)", call_sid, from_number, to_number)
    call_started_ms = now_ms()

    # Identity prefetch: look up company while WebSocket connects
    company = {"company_name": "our company", "agent_name": "Alex"}
    identity_task = asyncio.create_task(fetch_company_identity(to_number))
    identity_prefetched = False
    try:
        company = await asyncio.wait_for(asyncio.shield(identity_task),
                                          timeout=IDENTITY_PREFETCH_TIMEOUT_MS / 1000.0)
        identity_prefetched = True
        logger.info("Identity prefetched for call %s: %s", call_sid, company.get("company_name"))
    except (asyncio.TimeoutError, Exception) as exc:
        logger.info("Identity prefetch skipped for call %s: %s", call_sid, type(exc).__name__)

    gemini_ws = None
    try:
        ws_url, api_key = gemini_ws_url()
        gemini_ws = await websockets.connect(
            ws_url,
            additional_headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            compression=None,
            ping_interval=None,
        )

        # Gemini Live setup with system prompt, voice config, and function declarations
        setup_msg = {
            "setup": {
                "model": f"models/{GEMINI_MODEL}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}},
                },
                "realtimeInputConfig": {
                    "automaticActivityDetection": {
                        "disabled": False,
                        "prefixPaddingMs": 80,
                        "silenceDurationMs": 160,
                    },
                    "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                },
                "systemInstruction": {"parts": [{"text": build_system_prompt(company)}]},
                "tools": [{
                    "functionDeclarations": [
                        CHECK_AVAILABILITY_DECLARATION,
                        BOOK_APPOINTMENT_DECLARATION,
                        END_CALL_DECLARATION,
                    ]
                }],
            }
        }
        await gemini_ws.send(json.dumps(setup_msg))
        setup_resp = await gemini_ws.recv()
        logger.info("Gemini setup ACK for call %s in %sms", call_sid, elapsed_ms(call_started_ms))

        # Trigger initial greeting
        company_name = company.get("company_name", "")
        greeting_text = (f"The caller is on the line. Answer as {company.get('agent_name', 'Alex')} "
                         f"from {company_name} with a warm greeting. They called you.") if company_name else \
                        "The caller is on the line. Greet them warmly. They called you."
        await gemini_ws.send(json.dumps({"realtimeInput": {"text": greeting_text}}))

        # Call state tracker
        state = {
            "stream_sid": None,
            "tw_resample": None,
            "gem_resample": None,
            "assistant_speaking": False,
            "audio_marks_pending": set(),
            "outbound_audio_buffer": b"",
            "caller_speech_frames": 0,
            "input_speech_hangover_frames": 0,
            "last_barge_in_clear_ms": None,
            "identity_prefetched": identity_prefetched,
            "company": company,
            "to_number": to_number,
            "from_number": from_number,
            "call_started_ms": call_started_ms,
            "greeting_sent_ms": now_ms(),
            "first_twilio_audio_ms": None,
            "first_gemini_audio_ms": None,
            "tool_call_count": 0,
            "last_audio_forwarded_ms": None,
            "last_audio_stream_end_ms": None,
            "end_call_requested_ms": None,
            "last_caller_voice_ms": None,
            "booking_done": False,
        }

        # Max-call-duration watchdog (10 min safety net)
        asyncio.create_task(_call_duration_watchdog(call_sid, state, call_started_ms))

        # Bi-directional audio bridge
        await asyncio.gather(
            _twilio_to_gemini(ws, gemini_ws, call_sid, state),
            _gemini_to_twilio(ws, gemini_ws, call_sid, state),
        )

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected for call %s", call_sid)
    except Exception as exc:
        logger.error("Media stream error for call %s: %s", call_sid, exc)
    finally:
        if not identity_task.done():
            identity_task.cancel()
        if gemini_ws:
            await gemini_ws.close()
        logger.info("WS session ended for call %s (duration=%sms)", call_sid, elapsed_ms(call_started_ms))


CALL_MAX_DURATION_SECONDS = 600


async def _call_duration_watchdog(call_sid: str, state: dict, call_started_ms: float | None):
    """Hang up if call exceeds max duration (safety net)."""
    if call_started_ms is None:
        return
    remaining = max(0, CALL_MAX_DURATION_SECONDS * 1000.0 - (elapsed_ms(call_started_ms) or 0))
    await asyncio.sleep(remaining / 1000.0)
    logger.info("Max call duration reached for call %s", call_sid)
    await twilio_hangup_call(call_sid)


async def _force_hangup(call_sid: str, state: dict):
    """Wait for goodbye audio to finish, then hang up."""
    await asyncio.sleep(3.0)
    waited_ms = now_ms()
    while state.get("assistant_speaking") or state.get("audio_marks_pending"):
        if elapsed_ms(waited_ms) and elapsed_ms(waited_ms) >= 5000:
            break
        await asyncio.sleep(0.1)
    await twilio_hangup_call(call_sid)


async def _twilio_to_gemini(twilio_ws: WebSocket, gemini_ws, call_sid: str, state: dict):
    """Forward inbound Twilio audio to Gemini with VAD and barge-in support."""
    try:
        while True:
            try:
                data = await asyncio.wait_for(twilio_ws.receive_text(), timeout=0.25)
            except asyncio.TimeoutError:
                # Check if auto-hangup conditions met
                end_call_req = state.get("end_call_requested_ms")
                if (end_call_req and not state.get("assistant_speaking")
                        and not state.get("audio_marks_pending")):
                    idle_ms = elapsed_ms(end_call_req)
                    if idle_ms and idle_ms >= AUTO_HANGUP_IDLE_SECONDS * 1000.0:
                        logger.info("Auto-hangup for call %s (idle_ms=%s)", call_sid, idle_ms)
                        await twilio_hangup_call(call_sid)
                        break

                # Send audio stream end marker on silence
                last_audio = state.get("last_audio_forwarded_ms")
                if last_audio and elapsed_ms(last_audio) and elapsed_ms(last_audio) >= GEMINI_AUDIO_STREAM_END_IDLE_MS:
                    await gemini_ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                    state["last_audio_stream_end_ms"] = now_ms()
                continue

            msg = json.loads(data)
            event = msg.get("event")

            if event == "start":
                state["stream_sid"] = msg["start"]["streamSid"]

            elif event == "media":
                # Decode and convert audio
                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                if state["first_twilio_audio_ms"] is None:
                    state["first_twilio_audio_ms"] = now_ms()

                pcm_16k, state["tw_resample"] = twilio_mulaw_to_gemini_pcm(mulaw_bytes, state["tw_resample"])
                frame_rms = audioop.rms(pcm_16k, 2) if pcm_16k else 0

                # VAD with hangover
                if frame_rms >= INPUT_VAD_RMS_THRESHOLD:
                    state["input_speech_hangover_frames"] = INPUT_VAD_HANGOVER_FRAMES
                    state["last_caller_voice_ms"] = now_ms()
                elif state["input_speech_hangover_frames"] > 0:
                    state["input_speech_hangover_frames"] -= 1

                # Local barge-in (configurable)
                if ENABLE_LOCAL_BARGE_IN and state["assistant_speaking"]:
                    if frame_rms >= BARGE_IN_RMS_THRESHOLD:
                        state["caller_speech_frames"] += 1
                    else:
                        state["caller_speech_frames"] = 0
                    if state["caller_speech_frames"] >= BARGE_IN_CONSECUTIVE_FRAMES:
                        await send_twilio_clear(twilio_ws, state["stream_sid"])
                        state["assistant_speaking"] = False
                        state["audio_marks_pending"].clear()
                        state["outbound_audio_buffer"] = b""
                        state["caller_speech_frames"] = 0

                # Skip silent frames while assistant is speaking
                if (state["assistant_speaking"] and frame_rms < INPUT_VAD_RMS_THRESHOLD
                        and state["input_speech_hangover_frames"] <= 0):
                    continue

                # Forward to Gemini
                await gemini_ws.send(json.dumps({
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(pcm_16k).decode(),
                            "mimeType": f"audio/pcm;rate={GEMINI_IN_RATE}",
                        }
                    }
                }))
                state["last_audio_forwarded_ms"] = now_ms()

            elif event == "mark":
                mark = msg.get("mark", {}).get("name")
                if mark:
                    state["audio_marks_pending"].discard(mark)
                    if not state["audio_marks_pending"]:
                        state["assistant_speaking"] = False

            elif event == "stop":
                await gemini_ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                logger.info("Twilio stream stopped for call %s", call_sid)
                break

    except WebSocketDisconnect:
        logger.info("Twilio disconnected (twilio->gemini) for call %s", call_sid)


async def _gemini_to_twilio(twilio_ws: WebSocket, gemini_ws, call_sid: str, state: dict):
    """Forward Gemini audio to Twilio and handle function calls."""
    audio_chunks_sent = 0
    try:
        async for raw in gemini_ws:
            resp = json.loads(raw)

            # Handle audio output
            server_content = resp.get("serverContent")
            if server_content:
                if server_content.get("interrupted"):
                    await send_twilio_clear(twilio_ws, state["stream_sid"])
                    state["assistant_speaking"] = False
                    state["audio_marks_pending"].clear()
                    state["outbound_audio_buffer"] = b""

                for part in server_content.get("modelTurn", {}).get("parts", []):
                    inline_data = part.get("inlineData")
                    if not inline_data:
                        continue
                    audio_b64 = inline_data.get("data", "")
                    if not audio_b64 or not state["stream_sid"]:
                        continue

                    pcm_24k = base64.b64decode(audio_b64)
                    if state["first_gemini_audio_ms"] is None:
                        state["first_gemini_audio_ms"] = now_ms()
                        logger.info("TTFB for call %s: %sms", call_sid,
                                    elapsed_ms(state["greeting_sent_ms"]))

                    # Convert and chunk for Twilio
                    mulaw_8k, state["gem_resample"] = gemini_pcm_to_twilio_mulaw(pcm_24k, state["gem_resample"])
                    state["outbound_audio_buffer"] += mulaw_8k
                    chunks, state["outbound_audio_buffer"] = chunk_bytes(
                        state["outbound_audio_buffer"], TWILIO_OUTBOUND_CHUNK_BYTES)

                    for chunk in chunks:
                        await twilio_ws.send_json({
                            "event": "media",
                            "streamSid": state["stream_sid"],
                            "media": {"payload": base64.b64encode(chunk).decode()},
                        })
                        audio_chunks_sent += 1
                        state["assistant_speaking"] = True
                        state["audio_marks_pending"].add(f"audio-{call_sid}-{audio_chunks_sent}")
                        await send_twilio_mark(twilio_ws, state["stream_sid"],
                                              f"audio-{call_sid}-{audio_chunks_sent}")

            # Handle function calls from Gemini
            tool_call = resp.get("toolCall")
            if tool_call:
                for fn_call in tool_call.get("functionCalls", []):
                    fn_name = fn_call.get("name")
                    fn_args = fn_call.get("args", {})
                    fn_id = fn_call.get("id", "")
                    state["tool_call_count"] += 1

                    logger.info("Gemini function call: %s args=%s", fn_name, fn_args)

                    if fn_name == "check_availability":
                        result = await execute_check_availability(fn_args, state["company"])
                    elif fn_name == "book_appointment":
                        if state.get("booking_done"):
                            result = {"status": "error", "message": "Booking already completed"}
                        else:
                            result = await execute_book_appointment(fn_args, state["company"], call_sid)
                            if result.get("status") == "success":
                                state["booking_done"] = True
                    elif fn_name == "end_call":
                        state["end_call_requested_ms"] = now_ms()
                        result = {"status": "ok", "message": "Call closing. System will hang up shortly."}
                        asyncio.create_task(_force_hangup(call_sid, state))
                    else:
                        result = {"error": f"Unknown function: {fn_name}"}

                    await gemini_ws.send(json.dumps({
                        "toolResponse": {
                            "functionResponses": [{
                                "id": fn_id,
                                "name": fn_name,
                                "response": {"result": result},
                            }]
                        }
                    }))

    except WebSocketDisconnect:
        logger.info("Twilio disconnected (gemini->twilio) for call %s", call_sid)
    except websockets.exceptions.ConnectionClosed as exc:
        code = exc.rcvd.code if exc.rcvd else "?"
        logger.info("Gemini WS closed for call %s: code=%s", call_sid, code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8080")))
