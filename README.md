# Multi-Company AI Voice Agent Bridge

**Role:** Lead Developer & Systems Architect  
**Tech Stack:** Python 3.12, FastAPI, Google Gemini Multimodal Live API, Twilio Voice/Media Streams, n8n, Baserow, Google Calendar, Gmail, Twilio SMS

**Files:**
- `Solar-Voice-Agent-Bridge.json` — Sanitized n8n workflow export (3 webhooks, 22 nodes)
- `main.py` — FastAPI server bridging Twilio ↔ Gemini Live (see note below)
- `requirements.txt`, `Dockerfile` — Deployment artifacts

> **Note:** The `main.py`, `requirements.txt`, and `Dockerfile` files were removed from this showcase to comply with NDA restrictions. They are replaced by the `Solar-Voice-Agent-Bridge.json` n8n workflow file. The full FastAPI server (1459 lines) handled: WebSocket audio streaming, mulaw↔PCM codec conversion, VAD/barge-in, Gemini function calling, identity prefetch with timeout, auto-hangup, and call duration watchdog. The architecture is documented below.

## What It Does

An AI-powered phone receptionist that handles inbound calls for multiple companies simultaneously. When a customer calls, the system:

1. **Identifies the company** - Resolves the dialed phone number to a company profile (name, agent persona, calendar, timezone, custom instructions) via n8n webhook querying a Baserow CRM
2. **Answers naturally** - Gemini Live API streams real-time audio, responding as a human receptionist with the correct company identity
3. **Qualifies leads** - Asks 4 structured questions (homeowner? bill amount? sunlight? credit score?) one at a time
4. **Checks availability** - Calls Google Calendar via n8n to return real available slots (8AM-4PM, on the hour)
5. **Books appointments** - Collects name, email, phone, address, time → creates Google Calendar event + CRM record + sends confirmation email
6. **Handles edge cases** - Duplicate booking prevention, pending SMS fallback, auto-hangup on silence, max-duration cutoff

## Architecture

```
Phone Call → Twilio PSTN → POST /voice → TwiML (WebSocket stream URL)
                                                ↓
Twilio Media Stream (mulaw/8kHz) ←WebSocket→ FastAPI ←WebSocket→ Gemini Live (PCM/16kHz)
                                                ↓
                                     n8n Webhooks (HTTP)
                                     ├── /company-identity → Baserow CRM lookup
                                     ├── /check-availability → Google Calendar
                                     ├── /book-appointment → Calendar + CRM + Email
                                     └── /pending-booking → SMS fallback workflow
```

## Key Technical Details

### Real-Time Audio Pipeline
- Twilio sends audio as mulaw 8kHz → FastAPI converts to linear PCM 16kHz for Gemini
- Gemini returns audio as PCM 24kHz → FastAPI converts back to mulaw 8kHz for Twilio
- Audio chunks are split into 100ms segments for low-latency streaming
- Voice Activity Detection suppresses phone echo while agent is speaking

### Barge-In Support
- Caller speech above 700 RMS threshold for 3 consecutive frames triggers audio flush
- Clears Twilio playback buffer + Gemini interruption event
- Configurable cooldown (350ms) prevents false triggers from phone echo

### Function Calling
- Gemini natively calls `check_availability`, `book_appointment`, and `end_call`
- System validates all booking inputs: business hours (8AM-4PM), on the hour, valid email, ISO datetime
- Duplicate booking guard: once `booking_done` flag is set, subsequent `book_appointment` calls return error
- `end_call` function initiates graceful hangup: waits for goodbye audio → clears buffer → terminates call

### Identity Prefetch with Timeout
- Company identity lookup starts while WebSocket connects (asyncio task)
- 5-second timeout fallback: if lookup is slow, Gemini starts with generic greeting, then identity is injected mid-conversation
- Late identity injection sends a text instruction to Gemini to self-correct the company name naturally

### Knowledge-Grounded Responses
- 15 curated Q&A pairs injected into the system prompt
- Covers common customer questions (pricing, warranties, qualifications, installation, etc.)
- AI answers in "Helpful Expert" tone, always pivoting back to qualification/booking

## NDA Disclaimer

**Some implementation details have been modified or removed** to comply with confidentiality obligations. Specifically:
- Actual webhook endpoint URLs, API keys, and environment variable values have been replaced with placeholders
- Company names, phone numbers, email templates, and calendar IDs are generic
- Post-booking SMS confirmation workflows and internal routing logic have been summarized rather than shown
- Certain business-specific qualification criteria and pricing rules have been generalized

The architecture, audio pipeline, tool-calling pattern, VAD implementation, and system prompt structure remain structurally accurate representations of the production system.

## Relevance to Application Questions

- **Q12 (AI/automation systems I built):** This is a production AI voice agent handling real customer calls
- **Q13 (Multi-step LLM agent workflow):** Perfect example — Gemini Live model with 3 function declarations, real-time audio, multi-company identity resolution, calendar integration, and database persistence across 4+ n8n webhooks
- **Q14 (Internal knowledge system):** The 15 Q&A pairs form a curated knowledge base injected into the system prompt, grounding the AI's responses in verified information
- **Q15 (AI producing wrong answers):** The classifier sometimes misclassified companies based on domain name alone. Solution: implemented a 3-stage pipeline (keyword → scrape → LLM fallback) and added a "classifier_reason" field to track why each classification was made, enabling audit trails and prompt refinement
