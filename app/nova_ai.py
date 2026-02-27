import os
import io
import json
import time
import wave
import re
import tempfile
import threading
import asyncio
import struct
import math
import traceback
import sqlite3
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from flask import Blueprint

nova_ai = Blueprint('nova_ai', __name__)

# ── .env loader ──────────────────────────────────────────────────────────────
def _load_dotenv():
    for loc in [Path(".env"), Path(__file__).parent / ".env"]:
        if loc.exists():
            for raw in loc.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            break
_load_dotenv()

import gradio as gr
from groq import Groq

# ════════════════════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════════════════════
SEP  = "═" * 64
SEP2 = "─" * 64
ICONS = {
    "SYS":"⚙ ","MIC":"🎙 ","STT":"📝","LLM":"🧠","TTS":"🔊",
    "PIPE":"⚡","WARN":"⚠ ","ERROR":"✖ ","CMD":"📨","DB":"🗄 ",
}

def log(tag: str, msg: str, level: str = "INFO"):
    ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    icon = ICONS.get(tag, "  ")
    lvl  = f"[{level}]" if level != "INFO" else "     "
    print(f"  {ts} {icon} [{tag:<4}] {lvl} {msg}", flush=True)

def log_sep(title: str = ""):
    if title:
        pad = max(0, 60 - len(title)); l, r = pad // 2, pad - pad // 2
        print(f"\n{SEP}\n  {'─'*l} {title} {'─'*r}\n{SEP}", flush=True)
    else:
        print(f"  {SEP2}", flush=True)

def log_err(tag: str, e: Exception, extra: str = ""):
    log(tag, f"EXCEPTION {type(e).__name__}: {e}{' — '+extra if extra else ''}", "ERROR")
    for line in traceback.format_exc().splitlines():
        print(f"           {line}", flush=True)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

PA_RATE     = 16000
PA_CHUNK    = 1024
PA_CHANNELS = 1
PA_WIDTH    = 2
SILENCE_RMS = 300
SILENCE_SEC = 1.5
MIN_REC_SEC = 0.6
MAX_REC_SEC = 60.0

# 3 Female + 3 Male — confirmed edge-tts voices
VOICES = {
    "aria":    ("en-US-AriaNeural",    "Aria"),
    "sonia":   ("en-GB-SoniaNeural",   "Sonia"),
    "neerja":  ("en-IN-NeerjaNeural",  "Neerja"),
    "guy":     ("en-US-GuyNeural",     "Guy"),
    "ryan":    ("en-GB-RyanNeural",    "Ryan"),
    "william": ("en-AU-WilliamNeural", "William"),
}
VOICE_RATES = {
    "aria": "+5%", "sonia": "+0%", "neerja": "+3%",
    "guy":  "+0%", "ryan":  "-3%", "william": "+2%",
}
_active_voice = "aria"
_voice_lock   = threading.Lock()

SYSTEM_PROMPT = """You are NOVA — a brilliant AI voice assistant. Fast, warm, precise, emotionally intelligent.

EMOTION AWARENESS: Detect the user's emotional tone and mirror it subtly.

STRICT RULES:
- 1-3 sentences for simple queries; 4-5 max for complex.
- ZERO markdown: no bullets, asterisks, hashes, backticks.
- NEVER mention the current date, time, or day UNLESS the user explicitly asks.
- Never start with "I".
- Never say "Certainly!", "Great question!" — just answer.
- No unnecessary disclaimers or filler."""

# ════════════════════════════════════════════════════════════════════════════
# DATABASE (SQLite — WAL mode, context manager)
# ════════════════════════════════════════════════════════════════════════════
DB_PATH = Path(__file__).parent / "nova_conversations.db"
_db_lock = threading.Lock()

@contextmanager
def _db_conn():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-2000")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

def db_init():
    with _db_conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session   TEXT    NOT NULL,
                role      TEXT    NOT NULL CHECK(role IN ('user','assistant')),
                content   TEXT    NOT NULL,
                ts        TEXT    NOT NULL
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session)")
    log("DB", f"Database ready → {DB_PATH}")

def db_insert(session: str, role: str, content: str, ts: str):
    if not session or not role or not content:
        return
    try:
        with _db_lock:
            with _db_conn() as con:
                con.execute(
                    "INSERT INTO messages(session,role,content,ts) VALUES(?,?,?,?)",
                    (session[:64], role, content[:8192], ts)
                )
    except Exception as e:
        log_err("DB", e, "db_insert")

def db_export_session(session: str) -> list:
    try:
        with _db_conn() as con:
            rows = con.execute(
                "SELECT role,content,ts FROM messages WHERE session=? ORDER BY id",
                (session,)
            ).fetchall()
        return [{"role": r, "text": c, "ts": t} for r, c, t in rows]
    except Exception as e:
        log_err("DB", e, "db_export_session")
        return []

def db_all_sessions() -> list:
    try:
        with _db_conn() as con:
            rows = con.execute(
                "SELECT DISTINCT session FROM messages ORDER BY MIN(id) DESC"
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []

# ════════════════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ════════════════════════════════════════════════════════════════════════════
_client      = None
_client_lock = threading.Lock()

def get_client() -> Groq:
    global _client
    with _client_lock:
        if _client is None:
            key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY).strip()
            if not key:
                raise ValueError("GROQ_API_KEY not set — enter it in the sidebar")
            if not key.startswith("gsk_"):
                raise ValueError("Invalid API key format — must start with gsk_")
            _client = Groq(api_key=key, timeout=30.0)
            log("SYS", f"Groq client ready (****{key[-4:]})")
        return _client

def reset_client():
    global _client
    with _client_lock:
        _client = None

# ════════════════════════════════════════════════════════════════════════════
# CONVERSATION (in-memory + DB write-through)
# ════════════════════════════════════════════════════════════════════════════
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

class Conversation:
    MAX = 24
    def __init__(self):
        self._msgs: list = []
        self._log:  list = []
        self._lock  = threading.RLock()

    def add_user(self, text: str):
        text = text.strip()[:4096]
        if not text:
            return
        ts = _ts()
        with self._lock:
            self._msgs.append({"role": "user", "content": text})
            self._log.append({"role": "user", "text": text, "ts": ts})
            if len(self._msgs) > self.MAX * 2:
                self._msgs = self._msgs[-(self.MAX * 2):]
        db_insert(SESSION_ID, "user", text, ts)

    def add_nova(self, text: str):
        text = text.strip()[:4096]
        if not text:
            return
        ts = _ts()
        with self._lock:
            self._msgs.append({"role": "assistant", "content": text})
            self._log.append({"role": "assistant", "text": text, "ts": ts})
        db_insert(SESSION_ID, "assistant", text, ts)

    def history(self) -> list:
        with self._lock:
            return list(self._msgs)

    def log_data(self) -> list:
        with self._lock:
            return list(self._log)

    def export(self) -> str:
        with self._lock:
            return json.dumps(self._log, indent=2, ensure_ascii=False)

    def clear(self):
        with self._lock:
            n = len(self._log)
            self._msgs.clear()
            self._log.clear()
        log("SYS", f"Conversation cleared ({n} entries)")

conv = Conversation()
def _ts() -> str: return datetime.now().strftime("%H:%M:%S")

# ════════════════════════════════════════════════════════════════════════════
# RECORDER
# ════════════════════════════════════════════════════════════════════════════
class Recorder:
    def __init__(self):
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._running

    def start(self, on_done, on_status):
        with self._lock:
            if self._running:
                return
            self._running = True
        t = threading.Thread(target=self._loop, args=(on_done, on_status), daemon=True)
        self._thread = t
        t.start()

    def stop_early(self):
        with self._lock:
            self._running = False

    def _loop(self, on_done, on_status):
        pa = stream = None
        peak_rms = 0.0
        t_start = time.perf_counter()
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            # Find best input device
            dev_idx = None
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    dev_idx = i
                    break
            if dev_idx is None:
                on_status("error", "No microphone found — check system settings")
                return
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=PA_CHANNELS,
                rate=PA_RATE,
                input=True,
                input_device_index=dev_idx,
                frames_per_buffer=PA_CHUNK
            )
            frames: list = []
            t_last_speech = t_start
            grace = False
            on_status("listening", "Listening… auto-sends after 1.5s silence")

            while True:
                with self._lock:
                    if not self._running:
                        break
                try:
                    data = stream.read(PA_CHUNK, exception_on_overflow=False)
                except OSError as e:
                    log("MIC", f"Stream read error: {e}", "WARN")
                    break

                frames.append(data)
                samples = struct.unpack(f"{len(data)//2}h", data) if len(data) >= 2 else ()
                rms = math.sqrt(sum(s*s for s in samples) / len(samples)) if samples else 0.0
                peak_rms = max(peak_rms, rms)

                now = time.perf_counter()
                elapsed = now - t_start

                if not grace and elapsed >= MIN_REC_SEC:
                    grace = True
                if rms > SILENCE_RMS:
                    t_last_speech = now

                if grace:
                    sil = now - t_last_speech
                    if sil >= SILENCE_SEC:
                        on_status("silence_auto", "Auto-sending…")
                        break
                    elif sil >= 0.3:
                        on_status("counting_down", f"Auto-sending in {SILENCE_SEC - sil:.1f}s...")

                if elapsed >= MAX_REC_SEC:
                    on_status("silence_auto", "Max duration reached, sending…")
                    break

            if not frames or peak_rms < 50:
                on_status("error", "No audio captured — check your microphone")
                return

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(PA_CHANNELS)
                wf.setsampwidth(PA_WIDTH)
                wf.setframerate(PA_RATE)
                wf.writeframes(b"".join(frames))
            on_done(buf.getvalue())

        except ImportError:
            on_status("error", "PyAudio not installed — run: pip install pyaudio")
        except Exception as e:
            log_err("MIC", e)
            on_status("error", f"Microphone error: {str(e)[:80]}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass
            with self._lock:
                self._running = False

recorder = Recorder()

# ════════════════════════════════════════════════════════════════════════════
# STT
# ════════════════════════════════════════════════════════════════════════════
def transcribe(wav_bytes: bytes) -> str:
    if len(wav_bytes) < 1024:
        raise ValueError("Audio too short")
    f = io.BytesIO(wav_bytes)
    f.name = "audio.wav"
    r = get_client().audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
        response_format="text",
        language="en"
    )
    text = (r if isinstance(r, str) else getattr(r, "text", str(r))).strip()
    return text

# ════════════════════════════════════════════════════════════════════════════
# LLM — date/time injected only when user asks
# ════════════════════════════════════════════════════════════════════════════
TIME_KW = {
    "time", "date", "day", "today", "now", "when", "hour",
    "minute", "clock", "morning", "afternoon", "evening", "night"
}

def llm(user_text: str) -> str:
    conv.add_user(user_text)
    sys = SYSTEM_PROMPT
    words = set(re.findall(r'\b\w+\b', user_text.lower()))
    if words & TIME_KW:
        now = datetime.now()
        sys += f"\n\nCurrent: {now.strftime('%I:%M %p')}, {now.strftime('%A %B %d %Y')}."

    msgs = [{"role": "system", "content": sys}] + conv.history()
    t0 = time.perf_counter()

    r = get_client().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        max_tokens=320,
        temperature=0.7,
        top_p=0.9,
        stream=False
    )

    raw = r.choices[0].message.content
    if not raw:
        raise ValueError("Empty response from LLM")
    raw = raw.strip()
    ms = int((time.perf_counter() - t0) * 1000)
    log("LLM", f"{ms}ms | {len(raw)} chars")

    # Strip markdown artifacts
    reply = re.sub(r'\*\*(.+?)\*\*', r'\1', raw)
    reply = re.sub(r'\*(.+?)\*',     r'\1', reply)
    reply = re.sub(r'#+\s*',         '',    reply)
    reply = re.sub(r'`+',            '',    reply)
    reply = re.sub(r'\n+',           ' ',   reply).strip()

    conv.add_nova(reply)
    return reply

# ════════════════════════════════════════════════════════════════════════════
# TTS
# ════════════════════════════════════════════════════════════════════════════
_tts_lock = threading.Lock()

def tts(text: str, voice_key: str = None) -> str:
    with _voice_lock:
        vk = voice_key or _active_voice
    edge_voice = VOICES.get(vk, VOICES["aria"])[0]
    rate       = VOICE_RATES.get(vk, "+0%")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    path = tmp.name

    # Try edge-tts first
    try:
        import edge_tts

        async def _run():
            await edge_tts.Communicate(text, voice=edge_voice, rate=rate).save(path)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        if os.path.exists(path) and os.path.getsize(path) > 500:
            return path
        log("TTS", "edge-tts produced empty file, falling back", "WARN")
    except ImportError:
        log("TTS", "edge-tts not installed — using gTTS fallback", "WARN")
    except Exception as e:
        log_err("TTS", e, "edge-tts failed — trying gTTS")

    # gTTS fallback
    try:
        from gtts import gTTS
        tld_map = {
            "aria": "com", "sonia": "co.uk", "neerja": "co.in",
            "guy": "com", "ryan": "co.uk", "william": "com.au"
        }
        gTTS(text=text, lang="en", tld=tld_map.get(vk, "com"), slow=False).save(path)
        return path
    except Exception as e:
        log_err("TTS", e, "gTTS also failed")
        raise RuntimeError("All TTS engines failed") from e

# ════════════════════════════════════════════════════════════════════════════
# PIPELINE STATE QUEUE
# ════════════════════════════════════════════════════════════════════════════
_queue  = []
_qlock  = threading.Lock()
_qevent = threading.Event()

def _push(state, msg, you="", nova="", **kw):
    with _qlock:
        _queue.append({"state": state, "msg": msg, "you": you, "nova": nova, **kw})
    _qevent.set()

def _pop():
    with _qlock:
        return _queue.pop(0) if _queue else None

def _log_json():
    return json.dumps(conv.log_data())

def _stream_queue(done: threading.Event, timeout=90.0):
    audio_out = [None]
    t0 = time.perf_counter()
    while not done.is_set():
        _qevent.wait(timeout=0.10)
        _qevent.clear()
        while True:
            item = _pop()
            if item is None:
                break
            ap = item.pop("audio", None)
            if ap:
                audio_out[0] = ap
            yield _log_json(), audio_out[0], json.dumps(item)
        if time.perf_counter() - t0 > timeout:
            _push("error", "Request timed out — please try again")
            break
    # Drain remaining
    while True:
        item = _pop()
        if item is None:
            break
        ap = item.pop("audio", None)
        if ap:
            audio_out[0] = ap
        yield _log_json(), audio_out[0], json.dumps(item)

# ════════════════════════════════════════════════════════════════════════════
# PIPELINES
# ════════════════════════════════════════════════════════════════════════════
def run_voice_pipe(wav_bytes: bytes):
    t0 = time.perf_counter()
    _push("transcribing", "Transcribing…")
    try:
        you = transcribe(wav_bytes)
    except Exception as e:
        err = str(e)
        if "401" in err or "api_key" in err.lower():
            m = "Invalid API key — check sidebar"
        elif "429" in err:
            m = "Rate limited — please wait a moment"
        elif "audio too short" in err.lower():
            m = "Audio too short — speak longer"
        else:
            m = f"Transcription failed: {err[:60]}"
        _push("error", m)
        return

    if not you or len(you.strip()) < 2:
        _push("error", "Didn't catch that — please speak clearly")
        return

    stt_ms = int((time.perf_counter() - t0) * 1000)
    _push("thinking", "Thinking…", you=you)

    t1 = time.perf_counter()
    try:
        reply = llm(you)
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "401" in err:
            m = "API key error — check sidebar"
        elif "429" in err:
            m = "Rate limited — please wait"
        else:
            m = f"AI error: {err[:60]}"
        _push("error", m, you=you)
        return

    llm_ms = int((time.perf_counter() - t1) * 1000)
    _push("speaking", "Generating speech…", you=you, nova=reply)

    t2 = time.perf_counter()
    try:
        audio_path = tts(reply)
    except Exception as e:
        log_err("PIPE", e)
        # Still show reply even if TTS fails
        _push("done", reply, you=you, nova=reply,
              stt_ms=stt_ms, llm_ms=llm_ms, tts_ms=0,
              total_ms=int((time.perf_counter() - t0) * 1000))
        return

    tts_ms = int((time.perf_counter() - t2) * 1000)
    total  = int((time.perf_counter() - t0) * 1000)
    _push("done", f"Done {total}ms", you=you, nova=reply, audio=audio_path,
          stt_ms=stt_ms, llm_ms=llm_ms, tts_ms=tts_ms, total_ms=total)

def run_text_pipe(text: str):
    t0 = time.perf_counter()
    text = text.strip()[:2048]
    _push("thinking", "Thinking…", you=text)
    try:
        reply = llm(text)
    except Exception as e:
        err = str(e)
        if "api_key" in err.lower() or "401" in err:
            m = "API key error — check sidebar"
        elif "429" in err:
            m = "Rate limited — please wait"
        else:
            m = f"AI error: {err[:60]}"
        _push("error", m)
        return

    llm_ms = int((time.perf_counter() - t0) * 1000)
    _push("speaking", "Generating speech…", you=text, nova=reply)

    try:
        audio_path = tts(reply)
    except Exception as e:
        log_err("PIPE", e)
        _push("done", reply, you=text, nova=reply,
              llm_ms=llm_ms, total_ms=int((time.perf_counter() - t0) * 1000))
        return

    total = int((time.perf_counter() - t0) * 1000)
    _push("done", f"Done {total}ms", you=text, nova=reply, audio=audio_path,
          llm_ms=llm_ms, total_ms=total)

# ════════════════════════════════════════════════════════════════════════════
# COMMAND BUS
# ════════════════════════════════════════════════════════════════════════════
def handle_cmd(cmd_json: str):
    IDLE = json.dumps({"state": "idle", "msg": "", "you": "", "nova": ""})

    if not cmd_json or not cmd_json.strip():
        yield _log_json(), None, IDLE
        return

    try:
        cmd = json.loads(cmd_json)
    except (json.JSONDecodeError, ValueError):
        yield _log_json(), None, json.dumps({"state": "error", "msg": "Command parse error", "you": "", "nova": ""})
        return

    if not isinstance(cmd, dict):
        yield _log_json(), None, IDLE
        return

    c = str(cmd.get("cmd", "")).strip()
    log("CMD", f"cmd={c!r}")

    # ── set_voice ─────────────────────────────────────────────────────────
    if c == "set_voice":
        global _active_voice
        vk = str(cmd.get("voice", "aria")).strip()
        with _voice_lock:
            if vk in VOICES:
                _active_voice = vk
            else:
                vk = "aria"
                _active_voice = vk
        label = VOICES.get(vk, VOICES["aria"])[1]
        yield _log_json(), None, json.dumps({"state": "idle", "msg": f"Voice: {label}", "you": "", "nova": ""})
        return

    # ── set_key ───────────────────────────────────────────────────────────
    if c == "set_key":
        key = str(cmd.get("key", "")).strip()
        if not key:
            yield _log_json(), None, json.dumps({"state": "error", "msg": "Enter your API key first", "you": "", "nova": ""})
            return
        if not key.startswith("gsk_") or len(key) < 20:
            yield _log_json(), None, json.dumps({"state": "error", "msg": "Invalid key format (must start with gsk_)", "you": "", "nova": ""})
            return
        os.environ["GROQ_API_KEY"] = key
        reset_client()
        try:
            get_client()
            yield _log_json(), None, json.dumps({"state": "idle", "msg": "✓ API key applied!", "you": "", "nova": ""})
        except Exception as e:
            yield _log_json(), None, json.dumps({"state": "error", "msg": f"Key error: {str(e)[:60]}", "you": "", "nova": ""})
        return

    # ── clear ─────────────────────────────────────────────────────────────
    if c == "clear":
        conv.clear()
        with _qlock:
            _queue.clear()
        yield json.dumps([]), None, json.dumps({"state": "idle", "msg": "Conversation cleared", "you": "", "nova": ""})
        return

    # ── export ────────────────────────────────────────────────────────────
    if c == "export":
        data = conv.export()
        if data == "[]":
            yield _log_json(), None, json.dumps({"state": "idle", "msg": "Nothing to export yet", "you": "", "nova": ""})
            return
        fn  = f"nova_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False,
                encoding="utf-8", prefix="nova_export_"
            )
            tmp.write(data)
            tmp.close()
            yield _log_json(), None, json.dumps({"state": "idle", "msg": f"Exported: {fn}", "you": "", "nova": ""})
        except Exception as e:
            yield _log_json(), None, json.dumps({"state": "error", "msg": f"Export failed: {str(e)[:50]}", "you": "", "nova": ""})
        return

    # ── send_text ─────────────────────────────────────────────────────────
    if c == "send_text":
        text = str(cmd.get("text", "")).strip()
        if not text:
            yield _log_json(), None, IDLE
            return
        if len(text) > 2048:
            text = text[:2048]

        with _qlock:
            _queue.clear()
        _qevent.clear()

        done = threading.Event()
        def _run():
            try:
                run_text_pipe(text)
            except Exception as e:
                log_err("PIPE", e)
                _push("error", f"Unexpected error: {str(e)[:80]}")
            finally:
                done.set()

        threading.Thread(target=_run, daemon=True).start()
        yield from _stream_queue(done)
        return

    # ── mic_toggle ────────────────────────────────────────────────────────
    if c == "mic_toggle":
        if recorder.is_recording:
            recorder.stop_early()
            yield _log_json(), None, json.dumps({"state": "transcribing", "msg": "Processing…", "you": "", "nova": ""})
            return

        with _qlock:
            _queue.clear()
        _qevent.clear()

        wav_holder  = [None]
        rec_done    = threading.Event()
        err_holder  = [None]

        def on_done(wav):
            wav_holder[0] = wav
            rec_done.set()

        def on_status(state, msg):
            with _qlock:
                _queue.append({"state": state, "msg": msg, "you": "", "nova": ""})
            _qevent.set()
            if state == "error":
                err_holder[0] = msg
                rec_done.set()

        recorder.start(on_done=on_done, on_status=on_status)
        yield _log_json(), None, json.dumps({"state": "listening", "msg": "Listening… 1.5s silence auto-sends", "you": "", "nova": ""})

        t_start = time.perf_counter()
        last_out = ""
        while not rec_done.is_set():
            _qevent.wait(timeout=0.10)
            _qevent.clear()
            while True:
                item = _pop()
                if item is None:
                    break
                s = json.dumps(item)
                if s != last_out:
                    last_out = s
                    yield _log_json(), None, s
            if time.perf_counter() - t_start > MAX_REC_SEC + 5:
                recorder.stop_early()
                break

        # Drain any remaining status messages
        while True:
            item = _pop()
            if item is None:
                break
            yield _log_json(), None, json.dumps(item)

        if err_holder[0] or wav_holder[0] is None:
            return

        with _qlock:
            _queue.clear()
        _qevent.clear()

        pipe_done = threading.Event()
        def _run_voice():
            try:
                run_voice_pipe(wav_holder[0])
            except Exception as e:
                log_err("PIPE", e)
                _push("error", f"Pipeline error: {str(e)[:80]}")
            finally:
                pipe_done.set()

        threading.Thread(target=_run_voice, daemon=True).start()
        yield from _stream_queue(pipe_done)
        return

    # Unknown command
    yield _log_json(), None, json.dumps({"state": "error", "msg": f"Unknown command: {c[:20]}", "you": "", "nova": ""})


# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Tokens ── */
:root {
  --bg:      #0f0610;
  --bg2:     #170b14;
  --s1:      rgba(232,160,191,.06);
  --s2:      rgba(232,160,191,.11);
  --s3:      rgba(232,160,191,.18);
  --bd:      rgba(232,160,191,.10);
  --bd2:     rgba(232,160,191,.22);
  --rose:    #e8a0bf;
  --champagne: #f7d6e8;
  --crimson: #c9637c;
  --wine:    #8c3b5a;
  --glow:    #ff6b9d;
  --dim:     #6b2d47;
  --blush:   #ffb3d1;
  --ivory:   #fff0f8;
  --err:     #ff4d6d;
  --t95:     rgba(255,240,248,.95);
  --t80:     rgba(255,240,248,.80);
  --t60:     rgba(255,240,248,.60);
  --t35:     rgba(255,240,248,.35);
  --font:    'DM Sans', system-ui, sans-serif;
  --head:    'Syne', system-ui, sans-serif;
  --mono:    'JetBrains Mono', monospace;
  --ease:    cubic-bezier(.34,1.56,.64,1);
  --sb-w:    220px;
}

/* Light theme */
[data-theme="light"] {
  --bg:  #fdf6f9;
  --bg2: #f5e8ef;
  --s1:  rgba(140,59,90,.05);
  --s2:  rgba(140,59,90,.09);
  --s3:  rgba(140,59,90,.14);
  --bd:  rgba(140,59,90,.14);
  --bd2: rgba(140,59,90,.26);
  --t95: rgba(30,10,20,.92);
  --t80: rgba(30,10,20,.75);
  --t60: rgba(30,10,20,.55);
  --t35: rgba(30,10,20,.35);
}
[data-theme="light"] #nv-root::before {
  background:
    radial-gradient(ellipse 60% 40% at 50% -10%, rgba(201,99,124,.10) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 90% 90%,  rgba(140,59,90,.07)  0%, transparent 55%);
}
[data-theme="light"] .orb-sphere {
  box-shadow: 0 0 22px rgba(201,99,124,.45), 0 0 44px rgba(201,99,124,.16), inset 0 1px 0 rgba(255,255,255,.3);
}
[data-theme="light"] #sb,
[data-theme="light"] #panel {
  background: rgba(253,246,249,.96);
}
[data-theme="light"] .htitle {
  text-shadow: 0 0 30px rgba(201,99,124,.4), 0 0 60px rgba(201,99,124,.2);
}

*,*::before,*::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  background: var(--bg) !important;
  font-family: var(--font) !important;
  min-height: 100vh;
  overflow-x: hidden;
  color: var(--t95);
}

/* ── Hide Gradio chrome ── */
.gradio-container,
.gradio-container>.main,
.gradio-container>.main>.wrap,
.gradio-container>.main>.wrap>.padding {
  max-width: 100% !important;
  width: 100vw !important;
  padding: 0 !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  gap: 0 !important;
}
footer, .built-with { display: none !important; }
#_cmd, #_aout, #_txst, #_stst {
  position: fixed !important;
  left: -99999px !important;
  top: 0 !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
  opacity: 0 !important;
  pointer-events: none !important;
  z-index: -1 !important;
}

/* ── Layout ── */
#nv-root {
  display: grid;
  grid-template-columns: var(--sb-w) 1fr 260px;
  min-height: 100vh;
  width: 100vw;
  max-width: 100vw;
  position: relative;
}
#nv-root::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  background:
    radial-gradient(ellipse 60% 40% at 50% -10%, rgba(201,99,124,.14) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 90% 90%,  rgba(140,59,90,.10)  0%, transparent 55%),
    radial-gradient(ellipse 35% 25% at 5% 80%,   rgba(232,160,191,.06) 0%, transparent 50%);
}
#nv-root > * { position: relative; z-index: 1; }

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
#sb {
  border-right: 1px solid var(--bd);
  background: rgba(15,6,16,.94);
  backdrop-filter: blur(40px);
  -webkit-backdrop-filter: blur(40px);
  padding: 16px 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 100vh;
  overflow-y: auto;
  overflow-x: hidden;
}
#sb::-webkit-scrollbar { width: 2px; }
#sb::-webkit-scrollbar-thumb { background: rgba(232,160,191,.12); border-radius: 1px; }

/* Brand block */
.brand {
  display: flex;
  align-items: center;
  gap: 9px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--bd);
  flex-shrink: 0;
}
.bmark {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  flex-shrink: 0;
  background: linear-gradient(135deg, var(--crimson), var(--wine));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  box-shadow: 0 0 16px rgba(201,99,124,.40);
}
.brand-text { display: flex; flex-direction: column; gap: 1px; }
.bname {
font-family: var(--head);
font-size: 18px;
font-weight: 900;
letter-spacing: -.04em;
line-height: 1;
color: #ffffff;
}
.btag {
  font-size: 7px;
  font-weight: 600;
  letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--t35);
}

/* Section label */
.sbl {
  font-size: 7px;
  font-weight: 700;
  letter-spacing: .26em;
  text-transform: uppercase;
  color: var(--t35);
  margin-bottom: 3px;
}

/* Stat rows */
.sbg { display: flex; flex-direction: column; gap: 3px; }
.sr {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 5px 8px;
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 7px;
  transition: background .2s;
}
.sr:hover { background: var(--s2); }
.srk { font-size: 8px; color: var(--t60); }
.srv { font-size: 8px; font-weight: 600; color: var(--t80); font-family: var(--mono); }
.srv.live {
  color: var(--rose);
  display: flex;
  align-items: center;
  gap: 4px;
}
.ldot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--rose);
  animation: ldot-pulse 2.4s ease-in-out infinite;
  box-shadow: 0 0 6px var(--glow);
  flex-shrink: 0;
}
@keyframes ldot-pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.25; transform:scale(.7); }
}

/* Theme toggle */
#theme-btn {
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 7px;
  color: var(--t60);
  cursor: pointer;
  padding: 6px 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font);
  font-size: 9px;
  font-weight: 500;
  transition: all .2s;
  width: 100%;
  letter-spacing: .04em;
}
#theme-btn:hover { background: var(--s2); color: var(--t95); }

/* Voice grid — 3×2, just names, no region text */
.vgrid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 3px;
}
.vbtn {
  padding: 7px 4px;
  border-radius: 7px;
  text-align: center;
  background: var(--s1);
  border: 1px solid var(--bd);
  color: var(--t60);
  font-size: 9.5px;
  font-weight: 500;
  cursor: pointer;
  transition: all .18s;
  user-select: none;
  letter-spacing: .02em;
}
.vbtn:hover { background: var(--s2); color: var(--t80); }
.vbtn.on {
  background: rgba(201,99,124,.18);
  border-color: rgba(201,99,124,.50);
  color: var(--rose);
  font-weight: 700;
}

/* API key */
.kbox { display: flex; flex-direction: column; gap: 4px; }
#key-inp {
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 7px;
  color: var(--t80);
  font-family: var(--mono);
  font-size: 10px;
  padding: 6px 8px;
  width: 100%;
  outline: none;
  transition: border-color .2s;
}
#key-inp:focus { border-color: rgba(201,99,124,.50); }
#key-inp::placeholder { color: var(--t35); }
#key-apply {
  padding: 7px 0;
  background: linear-gradient(135deg, rgba(201,99,124,.18), rgba(140,59,90,.18));
  border: 1px solid rgba(201,99,124,.30);
  border-radius: 7px;
  color: var(--rose);
  font-family: var(--font);
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
  width: 100%;
  transition: all .2s;
  letter-spacing: .04em;
}
#key-apply:hover {
  background: linear-gradient(135deg, rgba(201,99,124,.30), rgba(140,59,90,.30));
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(201,99,124,.20);
}
#key-apply:active { transform: translateY(0); }
#key-status { font-size: 9px; color: var(--t60); text-align: center; min-height: 12px; }

/* Footer */
.sbfoot {
  margin-top: auto;
  font-size: 7.5px;
  color: var(--t35);
  text-align: center;
  line-height: 2;
  padding-top: 10px;
  border-top: 1px solid var(--bd);
}

/* ═══════════════════════════════════════════
   CENTER PANEL
═══════════════════════════════════════════ */
#ctr {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px 20px 60px;
  overflow-y: auto;
  min-height: 100vh;
}

.hbadge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(201,99,124,.08);
  border: 1px solid rgba(201,99,124,.22);
  border-radius: 100px;
  padding: 4px 14px;
  font-size: 7px;
  font-weight: 700;
  letter-spacing: .22em;
  color: var(--rose);
  text-transform: uppercase;
  margin-bottom: 10px;
}
.bdot {
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--rose);
  animation: ldot-pulse 2.8s ease-in-out infinite;
  box-shadow: 0 0 5px var(--glow);
}

/* NOVA heading — solid, always visible */
.htitle {
  font-family: var(--head);
  font-size: clamp(36px, 4.5vw, 62px);
  font-weight: 900;
  line-height: .88;
  letter-spacing: -.06em;
  text-align: center;
  color: var(--champagne);
  text-shadow: 0 0 40px rgba(255,107,157,.50), 0 0 80px rgba(201,99,124,.25);
  margin-bottom: 6px;
}
.hsub {
  font-size: 14px;
  font-weight: 300;
  color: var(--t60);
  text-align: center;
  line-height: 1.7;
  max-width: 360px;
  margin-bottom: 0;
}

/* ── Orb ── (REDUCED SIZE: 140px) */
.orb-wrap {
  position: relative;
  width: 140px;
  height: 140px;
  margin-top: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  margin-bottom: 12px;
  cursor: pointer;
  transition: transform .2s var(--ease);
}
.orb-wrap:active { transform: scale(.92); }

.orb-ring {
  position: absolute;
  border-radius: 50%;
  border: 1px solid transparent;
  pointer-events: none;
}
.or1 {
  inset: -10px;
  background:
    linear-gradient(var(--bg), var(--bg)) padding-box,
    conic-gradient(rgba(201,99,124,.60), transparent 42%, rgba(140,59,90,.50), transparent 85%) border-box;
  animation: spin-cw 8s linear infinite;
}
.or2 {
  inset: -20px;
  background:
    linear-gradient(var(--bg), var(--bg)) padding-box,
    conic-gradient(rgba(140,59,90,.30), transparent 52%, rgba(201,99,124,.24), transparent 96%) border-box;
  animation: spin-ccw 14s linear infinite;
}
.or3 {
  inset: -32px;
  background:
    linear-gradient(var(--bg), var(--bg)) padding-box,
    conic-gradient(rgba(232,160,191,.07), transparent 68%) border-box;
  animation: spin-cw 24s linear infinite;
}
@keyframes spin-cw  { to { transform: rotate(360deg);  } }
@keyframes spin-ccw { to { transform: rotate(-360deg); } }

.orb-proc {
  position: absolute;
  inset: -13px;
  border-radius: 50%;
  border: 2px solid transparent;
  border-top-color: var(--rose);
  border-right-color: var(--crimson);
  opacity: 0;
  pointer-events: none;
  animation: spin-cw .65s linear infinite;
  transition: opacity .3s;
}
.orb-proc.on { opacity: 1; }

.orb-sphere {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.40) 0%, transparent 42%),
    radial-gradient(circle at 70% 72%, rgba(0,0,0,.28) 0%, transparent 36%),
    linear-gradient(148deg, #f0a0c0 0%, #c9637c 42%, #6b1d3a 100%);
  box-shadow: 0 0 24px rgba(201,99,124,.60), 0 0 48px rgba(201,99,124,.22), inset 0 1px 0 rgba(255,255,255,.26);
  animation: orb-float 5.5s ease-in-out infinite;
  transition: all .5s var(--ease);
  display: flex;
  align-items: center;
  justify-content: center;
}
@keyframes orb-float {
  0%,100% { transform: translateY(0);   }
  50%      { transform: translateY(-5px); }
}
.orb-sphere.listening {
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.42) 0%, transparent 42%),
    linear-gradient(148deg, #ffb3d1 0%, #ff6b9d 45%, #cc0055 100%);
  box-shadow: 0 0 36px rgba(255,107,157,.85), 0 0 72px rgba(255,107,157,.38), inset 0 1px 0 rgba(255,255,255,.3);
  animation: orb-listen .35s ease-in-out infinite alternate;
}
@keyframes orb-listen { from { transform:scale(.95); } to { transform:scale(1.08); } }

.orb-sphere.countdown {
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.38) 0%, transparent 42%),
    linear-gradient(148deg, #f7d6e8 0%, #e8a0bf 45%, #8c3b5a 100%);
  box-shadow: 0 0 32px rgba(232,160,191,.75), 0 0 64px rgba(232,160,191,.30), inset 0 1px 0 rgba(255,255,255,.26);
  animation: orb-countdown .9s ease-in-out infinite;
}
@keyframes orb-countdown { 0%,100% { transform:scale(1); } 50% { transform:scale(1.04); } }

.orb-sphere.thinking {
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.34) 0%, transparent 42%),
    linear-gradient(148deg, #e8a0bf 0%, #c9637c 45%, #5a1a34 100%);
  box-shadow: 0 0 32px rgba(201,99,124,.78), 0 0 64px rgba(201,99,124,.30), inset 0 1px 0 rgba(255,255,255,.24);
  animation: orb-think 1.6s ease-in-out infinite;
}
@keyframes orb-think { 0%,100% { transform:scale(1); } 50% { transform:scale(1.05); } }

.orb-sphere.speaking {
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.44) 0%, transparent 42%),
    linear-gradient(148deg, #ffd4e8 0%, #ff6b9d 45%, #8c1a44 100%);
  box-shadow: 0 0 40px rgba(255,107,157,.88), 0 0 80px rgba(255,107,157,.42), inset 0 1px 0 rgba(255,255,255,.30);
  animation: orb-speak .22s ease-in-out infinite alternate;
}
@keyframes orb-speak { from { transform:scale(.96); } to { transform:scale(1.09); } }

.orb-sphere.error {
  background:
    radial-gradient(circle at 32% 28%, rgba(255,255,255,.26) 0%, transparent 42%),
    linear-gradient(148deg, #ffaaaa 0%, #ff4d6d 45%, #8b0000 100%);
  box-shadow: 0 0 28px rgba(255,77,109,.72), 0 0 56px rgba(255,77,109,.28), inset 0 1px 0 rgba(255,255,255,.20);
  animation: orb-err .40s ease;
}
@keyframes orb-err {
  0%,100% { transform:translateX(0); }
  20%     { transform:translateX(-6px); }
  40%     { transform:translateX(6px); }
  60%     { transform:translateX(-3px); }
  80%     { transform:translateX(3px); }
}

.orb-ico {
  font-size: 38px;
  user-select: none;
  pointer-events: none;
  filter: drop-shadow(0 2px 6px rgba(0,0,0,.55));
  transition: all .3s var(--ease);
  line-height: 1;
}

/* Waveform canvas */
#waveform-canvas {
  width: 100%;
  max-width: 280px;
  height: 40px;
  margin-top: 6px;
  opacity: 0;
  transition: opacity .4s;
  border-radius: 5px;
  display: block;
}
#waveform-canvas.on { opacity: 1; }

/* Status */
#stl {
  margin-top: 8px;
  font-size: 12px;
  font-weight: 400;
  color: var(--t60);
  text-align: center;
  min-height: 18px;
  letter-spacing: .03em;
  transition: color .3s;
}
#stl.cl { color: var(--glow); }
#stl.ct { color: var(--rose); }
#stl.cs { color: var(--champagne); }
#stl.cd { color: var(--rose); }
#stl.ce { color: var(--err); }

#sthin {
  margin-top: 3px;
  font-size: 7.5px;
  font-weight: 700;
  letter-spacing: .24em;
  text-transform: uppercase;
  color: var(--t35);
  text-align: center;
  transition: color .3s;
}
#sthin.hi { color: var(--t60); }

/* Wake badge */
#wake-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(140,59,90,.12);
  border: 1px solid rgba(140,59,90,.30);
  border-radius: 100px;
  padding: 3px 10px;
  font-size: 7.5px;
  color: var(--dim);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-top: 5px;
  transition: all .3s;
}
#wake-badge.active {
  background: rgba(255,107,157,.14);
  border-color: rgba(255,107,157,.45);
  color: var(--glow);
}

/* Silence bar */
#silw { width: 100%; max-width: 160px; margin-top: 8px; opacity: 0; transition: opacity .3s; }
#silw.on { opacity: 1; }
#siltrk { height: 2px; background: var(--s2); border-radius: 2px; overflow: hidden; }
#silfil {
  height: 100%; width: 100%;
  background: linear-gradient(90deg, var(--crimson), var(--glow));
  border-radius: 2px;
  transition: width .1s linear;
  box-shadow: 0 0 5px rgba(255,107,157,.5);
}
#sillbl { font-size: 8px; color: var(--rose); text-align: center; margin-top: 3px; font-family: var(--mono); }

/* Caption card */
#cap {
  margin-top: 14px;
  width: 100%;
  max-width: 460px;
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 12px;
  padding: 11px 13px;
  display: flex;
  flex-direction: column;
  gap: 7px;
  opacity: 0;
  transform: translateY(8px);
  transition: opacity .4s, transform .4s;
}
#cap.on { opacity: 1; transform: translateY(0); }
.crow { display: flex; align-items: flex-start; gap: 8px; }
.clbl {
  font-size: 7px;
  font-weight: 700;
  letter-spacing: .22em;
  text-transform: uppercase;
  min-width: 30px;
  padding-top: 3px;
  flex-shrink: 0;
}
.clbl.you { color: var(--crimson); }
.clbl.nova { color: var(--rose); }
.ctxt { font-size: 13.5px; font-weight: 300; color: var(--t80); line-height: 1.65; }
.csep { height: 1px; background: var(--bd); }

/* Latency chips */
#latr { display: flex; gap: 5px; margin-top: 6px; flex-wrap: wrap; justify-content: center; opacity: 0; transition: opacity .4s; }
#latr.on { opacity: 1; }
.lc {
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 100px;
  padding: 2px 9px;
  font-size: 8px;
  font-family: var(--mono);
  color: var(--t35);
}
.lc em { color: var(--rose); font-style: normal; font-weight: 600; }

/* Input bar */
#tbar {
  display: flex;
  gap: 8px;
  align-items: center;
  width: 100%;
  max-width: 520px;
  margin-top: 22px;
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 50px;
  padding: 7px 8px 7px 20px;
  transition: border-color .25s, box-shadow .25s;
}
#tbar:focus-within {
  border-color: rgba(201,99,124,.50);
  box-shadow: 0 0 0 3px rgba(201,99,124,.10), 0 0 18px rgba(201,99,124,.12);
}
#tinp {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--t95);
  font-family: var(--font);
  font-size: 14px;
  padding: 4px 0;
}
#tinp::placeholder { color: var(--t35); }
#mic-btn {
  width: 42px; height: 42px;
  border-radius: 50%;
  flex-shrink: 0;
  background: linear-gradient(135deg, rgba(201,99,124,.18), rgba(140,59,90,.22));
  border: 1px solid rgba(201,99,124,.32);
  color: var(--rose);
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .22s var(--ease);
}
#mic-btn:hover {
  background: linear-gradient(135deg, rgba(201,99,124,.32), rgba(140,59,90,.36));
  transform: scale(1.10);
  box-shadow: 0 0 14px rgba(201,99,124,.30);
}
#mic-btn:active { transform: scale(.90); }
#mic-btn.rec {
  background: linear-gradient(135deg, rgba(255,107,157,.28), rgba(204,0,85,.28));
  border-color: rgba(255,107,157,.55);
  color: var(--glow);
  animation: mic-pulse .55s ease-in-out infinite alternate;
}
@keyframes mic-pulse {
  from { box-shadow: 0 0 0 0 rgba(255,107,157,.40); }
  to   { box-shadow: 0 0 0 8px rgba(255,107,157,0); }
}
#tsnd {
  width: 42px; height: 42px;
  border-radius: 50%;
  flex-shrink: 0;
  background: linear-gradient(135deg, var(--crimson), var(--wine));
  color: var(--champagne);
  font-size: 13px;
  cursor: pointer;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform .18s, box-shadow .18s;
  box-shadow: 0 0 12px rgba(201,99,124,.28);
}
#tsnd:hover { transform: scale(1.10); box-shadow: 0 0 20px rgba(201,99,124,.45); }
#tsnd:active { transform: scale(.90); }

/* ═══════════════════════════════════════════
   TRANSCRIPT PANEL
═══════════════════════════════════════════ */
#panel {
  border-left: 1px solid var(--bd);
  background: rgba(15,6,16,.88);
  backdrop-filter: blur(40px);
  -webkit-backdrop-filter: blur(40px);
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.phead {
  padding: 12px 11px 9px;
  border-bottom: 1px solid var(--bd);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  gap: 6px;
}
.ptit {
  font-size: 7.5px;
  font-weight: 700;
  letter-spacing: .22em;
  text-transform: uppercase;
  color: var(--t35);
  display: flex;
  align-items: center;
  gap: 5px;
  white-space: nowrap;
}
.mcnt {
  background: var(--s2);
  border-radius: 100px;
  padding: 1px 7px;
  font-size: 8.5px;
  color: var(--t60);
}
.pact { display: flex; gap: 4px; flex-wrap: wrap; }
.pbtn {
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 6px;
  color: var(--t35);
  font-family: var(--font);
  font-size: 9px;
  font-weight: 500;
  padding: 4px 9px;
  cursor: pointer;
  transition: all .18s;
  white-space: nowrap;
}
.pbtn:hover {
  background: var(--s2);
  color: var(--t95);
  border-color: rgba(201,99,124,.35);
  transform: translateY(-1px);
}

/* Search bar */
#srchbar { padding: 7px 10px; border-bottom: 1px solid var(--bd); flex-shrink: 0; }
#srch {
  width: 100%;
  background: var(--s1);
  border: 1px solid var(--bd);
  border-radius: 18px;
  color: var(--t80);
  font-family: var(--font);
  font-size: 10px;
  padding: 5px 11px;
  outline: none;
  transition: border-color .2s;
}
#srch:focus { border-color: rgba(201,99,124,.45); }
#srch::placeholder { color: var(--t35); }

#txsc {
  flex: 1;
  overflow-y: auto;
  padding: 8px 10px;
  scroll-behavior: smooth;
}
#txsc::-webkit-scrollbar { width: 2px; }
#txsc::-webkit-scrollbar-thumb { background: rgba(232,160,191,.12); border-radius: 1px; }

.tempty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 140px;
  color: var(--t35);
  text-align: center;
  gap: 7px;
}
.tempty-i { font-size: 20px; opacity: .35; }
.tempty-t { font-size: 10px; font-weight: 300; line-height: 1.75; }

.mi { margin-bottom: 10px; animation: mi-pop .3s var(--ease); }
@keyframes mi-pop {
  from { opacity:0; transform:translateY(4px) scale(.97); }
  to   { opacity:1; transform:none; }
}
.mi.hidden { display: none; }

.mh { display: flex; align-items: center; gap: 5px; margin-bottom: 3px; padding: 0 2px; }
.mr { font-size: 7px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
.mr.u { color: var(--crimson); }
.mr.n { color: var(--rose); }
.mt { font-size: 7px; color: var(--t35); font-family: var(--mono); }

.bub {
  padding: 7px 10px;
  border-radius: 9px;
  font-size: 10.5px;
  font-weight: 300;
  line-height: 1.65;
  color: var(--t80);
  word-break: break-word;
}
.bub.u {
  background: rgba(201,99,124,.08);
  border: 1px solid rgba(201,99,124,.18);
  border-bottom-right-radius: 3px;
}
.bub.n {
  background: rgba(232,160,191,.06);
  border: 1px solid rgba(232,160,191,.14);
  border-bottom-left-radius: 3px;
}

/* Copy button */
.bub-wrap { position: relative; }
.copy-btn {
  position: absolute;
  top: 4px; right: 5px;
  background: var(--s2);
  border: 1px solid var(--bd2);
  border-radius: 4px;
  color: var(--t35);
  font-size: 8px;
  padding: 2px 5px;
  cursor: pointer;
  opacity: 0;
  transition: opacity .18s;
  font-family: var(--mono);
}
.bub-wrap:hover .copy-btn { opacity: 1; }
.copy-btn:hover { color: var(--rose); border-color: rgba(201,99,124,.40); }
.copy-btn.copied { color: #6dffb3; border-color: rgba(109,255,179,.40); }

/* ── Responsive ── */
@media (max-width: 960px) {
  :root { --sb-w: 190px; }
  #nv-root { grid-template-columns: var(--sb-w) 1fr 230px; }
}
@media (max-width: 740px) {
  #nv-root { grid-template-columns: 1fr; }
  #sb, #panel { display: none; }
}
"""

# ════════════════════════════════════════════════════════════════════════════
# JAVASCRIPT
# ════════════════════════════════════════════════════════════════════════════
JS = r"""
(function () {
"use strict";

var SILENCE_SEC = 1.5;
var lastSt = "", lastTx = "", txCnt = 0, curAudio = null, isRec = false;
var seenAudioSrcs = new Set();
var _theme = "dark";

/* ── Command bus ── */
function sendCmd(obj) {
  obj.ts = Date.now();
  var ta = findTA("_cmd");
  if (!ta) { console.warn("[NOVA] _cmd textarea not found"); return; }
  setTA(ta, JSON.stringify(obj));
  ta.dispatchEvent(new Event("input",  { bubbles: true }));
  ta.dispatchEvent(new Event("change", { bubbles: true }));
}

function findTA(id) {
  var w = document.getElementById(id);
  if (!w) return null;
  return w.querySelector("textarea") || w.querySelector("input") || null;
}

function setTA(el, val) {
  try {
    var proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    var d = Object.getOwnPropertyDescriptor(proto, "value");
    if (d && d.set) d.set.call(el, val); else el.value = val;
  } catch (e) { el.value = val; }
}

/* ── Theme toggle ── */
window.novaToggleTheme = function () {
  _theme = _theme === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", _theme);
  var btn = document.getElementById("theme-btn");
  if (btn) btn.textContent = _theme === "dark" ? "☀  Light mode" : "🌙  Dark mode";
};

/* ── Audio player ── */
function watchAudio() {
  function chk() {
    var roots = [document.getElementById("_aout"), document.body].filter(Boolean);
    roots.forEach(function (root) {
      root.querySelectorAll("audio").forEach(function (el) {
        var src = el.src || el.currentSrc || "";
        if (src && src !== "about:blank" && src.length > 12 && !seenAudioSrcs.has(src)) {
          seenAudioSrcs.add(src);
          playAudio(src);
        }
      });
    });
  }
  setInterval(chk, 140);
  new MutationObserver(chk).observe(document.body, {
    subtree: true, childList: true,
    attributes: true, attributeFilter: ["src", "currentSrc"]
  });
}

function playAudio(src) {
  if (curAudio) {
    try { curAudio.pause(); curAudio.src = ""; } catch (e) {}
    curAudio = null;
  }
  var a = new Audio(src);
  curAudio = a;
  a.onplay   = function () { orbSkin("speaking"); stl("NOVA is speaking…", "cs"); hin("RESPONDING", true); sbs("Speaking"); };
  a.onended  = function () { curAudio = null; idle(); };
  a.onerror  = function () { curAudio = null; idle(); };
  a.play().catch(function () {
    stl("Tap anywhere to hear NOVA", "cd");
    document.addEventListener("click", function once() {
      if (curAudio) curAudio.play().catch(function () {});
      document.removeEventListener("click", once);
    }, { once: true });
  });
}

/* ── Real-time waveform (Web Audio API) ── */
var _audioCtx = null, _analyser = null, _waveStream = null, _waveRaf = null;
var _waveCanvas = null, _waveCtx2d = null;

function startWaveform() {
  _waveCanvas = document.getElementById("waveform-canvas");
  if (!_waveCanvas) return;
  _waveCanvas.classList.add("on");
  _waveCtx2d = _waveCanvas.getContext("2d");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
  navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    .then(function (stream) {
      _waveStream = stream;
      _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      var src = _audioCtx.createMediaStreamSource(stream);
      _analyser = _audioCtx.createAnalyser();
      _analyser.fftSize = 256;
      src.connect(_analyser);
      drawWave();
    })
    .catch(function (e) { console.warn("[NOVA] Waveform mic:", e.message); });
}

function stopWaveform() {
  if (_waveRaf) { cancelAnimationFrame(_waveRaf); _waveRaf = null; }
  if (_waveStream) { _waveStream.getTracks().forEach(function (t) { t.stop(); }); _waveStream = null; }
  if (_audioCtx) { try { _audioCtx.close(); } catch (e) {} _audioCtx = null; }
  _analyser = null;
  if (_waveCanvas) {
    _waveCanvas.classList.remove("on");
    if (_waveCtx2d) _waveCtx2d.clearRect(0, 0, _waveCanvas.width, _waveCanvas.height);
  }
}

function drawWave() {
  if (!_analyser || !_waveCanvas || !_waveCtx2d) return;
  _waveRaf = requestAnimationFrame(drawWave);
  var W = _waveCanvas.width  = _waveCanvas.offsetWidth  || 280;
  var H = _waveCanvas.height = _waveCanvas.offsetHeight || 40;
  var buf = new Uint8Array(_analyser.frequencyBinCount);
  _analyser.getByteTimeDomainData(buf);
  _waveCtx2d.clearRect(0, 0, W, H);
  _waveCtx2d.beginPath();
  var sliceW = W / buf.length, x = 0;
  for (var i = 0; i < buf.length; i++) {
    var v = buf[i] / 128.0, y = v * H / 2;
    if (i === 0) _waveCtx2d.moveTo(x, y); else _waveCtx2d.lineTo(x, y);
    x += sliceW;
  }
  _waveCtx2d.lineTo(W, H / 2);
  _waveCtx2d.strokeStyle = "#ff6b9d";
  _waveCtx2d.lineWidth = 1.6;
  _waveCtx2d.shadowColor = "rgba(255,107,157,.5)";
  _waveCtx2d.shadowBlur = 5;
  _waveCtx2d.stroke();
}

/* ── Wake word (Web Speech API) ── */
var _wakeRec = null, _wakeActive = false;

function initWakeWord() {
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var badge = document.getElementById("wake-badge");
  if (!SR) {
    if (badge) badge.textContent = "🎤 Say \"Hey Nova\" unavailable";
    return;
  }

  var _wakeRec = null;
  var _wakeActive = false;
  var _restartTimer = null;
  var _lastTrigger = 0;
  var _consecutiveErrors = 0;

  function startWake() {
    if (_restartTimer) { clearTimeout(_restartTimer); _restartTimer = null; }
    try {
      _wakeRec = new SR();
      _wakeRec.continuous = true;
      _wakeRec.interimResults = true;
      _wakeRec.maxAlternatives = 2;
      _wakeRec.lang = "en-US";

      _wakeRec.onstart = function () {
        _consecutiveErrors = 0;
        _wakeActive = true;
        if (badge) badge.textContent = '🎤 Say "Hey Nova" to activate';
      };

      _wakeRec.onresult = function (e) {
        var t = "";
        for (var i = e.resultIndex; i < e.results.length; i++) {
          for (var j = 0; j < e.results[i].length; j++) {
            t += e.results[i][j].transcript.toLowerCase();
          }
        }
        var now = Date.now();
        var triggered = t.includes("hey nova") || t.includes("ok nova") ||
                        t.includes("hey nava") || t.includes("hey nover") ||
                        t.includes("a nova")   || t.includes("ey nova");
        if (triggered && !isRec && (now - _lastTrigger) > 3000) {
          _lastTrigger = now;
          if (badge) { badge.textContent = "✦ Wake word detected!"; badge.classList.add("active"); }
          setTimeout(function () {
            if (badge) { badge.textContent = '🎤 Say "Hey Nova" to activate'; badge.classList.remove("active"); }
          }, 2000);
          sendCmd({ cmd: "mic_toggle" });
        }
      };

      _wakeRec.onend = function () {
        _wakeActive = false;
        // Always restart unless page is unloading — use backoff on repeated errors
        var delay = Math.min(300 + _consecutiveErrors * 200, 2000);
        _restartTimer = setTimeout(startWake, delay);
      };

      _wakeRec.onerror = function (e) {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") {
          if (badge) badge.textContent = "🎤 Mic permission denied";
          _wakeActive = false;
          return; // Don't restart if permission denied
        }
        if (e.error !== "no-speech" && e.error !== "aborted") {
          _consecutiveErrors++;
          console.warn("[NOVA] Wake SR error:", e.error, "| consecutive:", _consecutiveErrors);
        }
        // onend will fire after this and handle restart
      };

      _wakeRec.onaudioend = function () {
        // Some browsers fire audioend without end — force a restart check
        setTimeout(function () {
          if (!_wakeActive && !isRec) {
            _restartTimer = _restartTimer || setTimeout(startWake, 400);
          }
        }, 500);
      };

      _wakeRec.start();
    } catch (e) {
      console.warn("[NOVA] Wake word init failed:", e.message);
      _consecutiveErrors++;
      var delay = Math.min(1000 + _consecutiveErrors * 500, 5000);
      _restartTimer = setTimeout(startWake, delay);
    }
  }

  startWake();
}

/* ── Poll Gradio outputs ── */
function poll() {
  var stEl = findTA("_stst");
  if (stEl && stEl.value && stEl.value !== lastSt && stEl.value.length > 2) {
    lastSt = stEl.value;
    try { onState(JSON.parse(stEl.value)); } catch (e) {}
  }
  var txEl = findTA("_txst");
  if (txEl && txEl.value && txEl.value !== lastTx && txEl.value.length > 2) {
    lastTx = txEl.value;
    try {
      var a = JSON.parse(txEl.value);
      if (a.length !== txCnt) { txCnt = a.length; renderTx(a); }
    } catch (e) {}
  }
}

/* ── State handler ── */
function onState(d) {
  var s   = d.state || "idle";
  var msg = d.msg   || "";
  var you = d.you   || "";
  var nov = d.nova  || "";

  var pr = document.getElementById("orb-proc");
  if (pr) pr.className = "orb-proc" + ((s === "transcribing" || s === "thinking") ? " on" : "");

  var skinMap = {
    idle: "", done: "", listening: "listening", counting_down: "countdown",
    silence_auto: "countdown", transcribing: "thinking", thinking: "thinking",
    speaking: "speaking", error: "error"
  };
  orbSkin(skinMap[s] || "");

  var clsMap = {
    idle: "", done: "cd", listening: "cl", counting_down: "ct", silence_auto: "ct",
    transcribing: "ct", thinking: "ct", speaking: "cs", error: "ce"
  };
  stl(msg, clsMap[s] || "");

  var icoMap = {
    idle: "◎", listening: "🎙", transcribing: "✦", thinking: "◌",
    speaking: "◉", done: "◎", error: "✕", counting_down: "⏱", silence_auto: "⏱"
  };
  ico(icoMap[s] || "◎");

  var hintMap = {
    idle: "TAP ORB OR 🎙 TO SPEAK", done: "TAP ORB OR 🎙 TO SPEAK",
    listening: "LISTENING — TAP 🎙 TO STOP", counting_down: "KEEP TALKING — TAP 🎙 TO STOP",
    transcribing: "TRANSCRIBING…", thinking: "THINKING…", speaking: "RESPONDING…", error: "TRY AGAIN"
  };
  hin(hintMap[s] || "TAP ORB OR 🎙 TO SPEAK", s !== "idle" && s !== "done");

  var sbMap = {
    idle: "Ready", done: "Ready", listening: "Listening…", counting_down: "Auto-sending…",
    transcribing: "Transcribing…", thinking: "Thinking…", speaking: "Speaking…", error: "Error"
  };
  sbs(sbMap[s] || "Ready");

  cap(you, nov);

  if (s === "counting_down") {
    var m = msg.match(/([\d.]+)s/);
    if (m) silBar(parseFloat(m[1]));
  } else {
    silHide();
  }

  var mb = document.getElementById("mic-btn");
  if (mb) mb.classList.toggle("rec", s === "listening" || s === "counting_down");

  if (s === "listening") { isRec = true; startWaveform(); }
  if (s === "done") { showLat(d); isRec = false; stopWaveform(); }
  if (s === "error") { isRec = false; silHide(); stopWaveform(); }
  if (s === "transcribing" || s === "thinking") { stopWaveform(); }
}

/* ── Transcript search ── */
window.novaSearch = function () {
  var q = (document.getElementById("srch") || {}).value || "";
  var kw = q.trim().toLowerCase();
  document.querySelectorAll(".mi").forEach(function (mi) {
    if (!kw) { mi.classList.remove("hidden"); return; }
    var txt = (mi.querySelector(".bub") || {}).textContent || "";
    mi.classList.toggle("hidden", !txt.toLowerCase().includes(kw));
  });
};

/* ── Copy bubble text ── */
window.novaCopyBubble = function (btn) {
  var wrap = btn.closest(".bub-wrap");
  var bub = wrap ? wrap.querySelector(".bub") : null;
  if (!bub) return;
  navigator.clipboard.writeText(bub.textContent.trim()).then(function () {
    btn.textContent = "✓"; btn.classList.add("copied");
    setTimeout(function () { btn.textContent = "copy"; btn.classList.remove("copied"); }, 1800);
  }).catch(function () {
    // Fallback for older browsers
    try {
      var range = document.createRange();
      range.selectNodeContents(bub);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      document.execCommand("copy");
      sel.removeAllRanges();
      btn.textContent = "✓"; btn.classList.add("copied");
      setTimeout(function () { btn.textContent = "copy"; btn.classList.remove("copied"); }, 1800);
    } catch (e) {}
  });
};

/* ── Public actions ── */
window.novaOrbClick = function () { sendCmd({ cmd: "mic_toggle" }); };
window.novaMicClick = function () { sendCmd({ cmd: "mic_toggle" }); };

window.novaSendText = function () {
  var inp = document.getElementById("tinp");
  if (!inp) return;
  var val = inp.value.trim();
  if (!val) return;
  if (val.length > 2048) val = val.slice(0, 2048);
  inp.value = "";
  sendCmd({ cmd: "send_text", text: val });
};

window.novaSetVoice = function (key) {
  document.querySelectorAll(".vbtn").forEach(function (b) {
    b.classList.toggle("on", b.dataset.voice === key);
  });
  sendCmd({ cmd: "set_voice", voice: key });
};

window.novaApplyKey = function () {
  var inp = document.getElementById("key-inp");
  var sta = document.getElementById("key-status");
  if (!inp) return;
  var val = inp.value.trim();
  if (!val) { if (sta) sta.textContent = "Enter your API key first"; return; }
  if (!val.startsWith("gsk_")) { if (sta) sta.textContent = "Key must start with gsk_"; return; }
  if (sta) sta.textContent = "Applying…";
  sendCmd({ cmd: "set_key", key: val });
  setTimeout(function () { if (sta && sta.textContent === "Applying…") sta.textContent = ""; }, 3000);
};

window.novaClear = function () {
  txCnt = 0; lastTx = "";
  var sc = document.getElementById("txsc");
  if (sc) sc.innerHTML = emptyTx();
  var b = document.querySelector(".mcnt");
  if (b) b.textContent = "0";
  var tc = document.getElementById("turn-count");
  if (tc) tc.textContent = "0";
  var c = document.getElementById("cap");
  if (c) c.classList.remove("on");
  var lr = document.getElementById("latr");
  if (lr) lr.classList.remove("on");
  idle();
  sendCmd({ cmd: "clear" });
};

window.novaExport = function () { sendCmd({ cmd: "export" }); };

/* ── Visuals ── */
function orbSkin(cls) {
  var s = document.querySelector(".orb-sphere");
  if (s) s.className = "orb-sphere" + (cls ? " " + cls : "");
}
function ico(ch) { var e = document.querySelector(".orb-ico"); if (e) e.textContent = ch; }
function stl(t, c) { var e = document.getElementById("stl"); if (e) { e.textContent = t; e.className = c || ""; } }
function hin(t, a) { var e = document.getElementById("sthin"); if (e) { e.textContent = t; e.className = a ? "hi" : ""; } }
function sbs(t) { var e = document.getElementById("sbst"); if (e) e.textContent = t; }

function cap(you, nova) {
  var c = document.getElementById("cap"); if (!c) return;
  if (!you && !nova) { c.classList.remove("on"); return; }
  c.classList.add("on");
  var yr  = document.getElementById("cyr");
  var yt  = document.getElementById("cyt");
  var sep = document.getElementById("csep");
  var nr  = document.getElementById("cnr");
  var nt  = document.getElementById("cnt");
  if (yt)  yt.textContent  = you || "";
  if (yr)  yr.style.display  = you ? "flex" : "none";
  if (sep) sep.style.display = (you && nova) ? "block" : "none";
  if (nt)  nt.textContent  = nova || "";
  if (nr)  nr.style.display  = nova ? "flex" : "none";
}

function showLat(d) {
  var row = document.getElementById("latr"); if (!row) return;
  var chips = [];
  if (d.stt_ms)   chips.push('<div class="lc">STT <em>'   + d.stt_ms   + 'ms</em></div>');
  if (d.llm_ms)   chips.push('<div class="lc">LLM <em>'   + d.llm_ms   + 'ms</em></div>');
  if (d.tts_ms)   chips.push('<div class="lc">TTS <em>'   + d.tts_ms   + 'ms</em></div>');
  if (d.total_ms) chips.push('<div class="lc">Total <em>' + d.total_ms + 'ms</em></div>');
  row.innerHTML = chips.join(""); row.classList.add("on");
}

function silBar(rem) {
  var w = document.getElementById("silw");
  var f = document.getElementById("silfil");
  var l = document.getElementById("sillbl");
  if (!w) return;
  w.classList.add("on");
  if (f) f.style.width = Math.max(0, Math.min(100, (rem / SILENCE_SEC) * 100)) + "%";
  if (l) l.textContent = "Auto-sending in " + rem.toFixed(1) + "s…";
}
function silHide() {
  var w = document.getElementById("silw"); if (w) w.classList.remove("on");
  var f = document.getElementById("silfil"); if (f) f.style.width = "100%";
  var l = document.getElementById("sillbl"); if (l) l.textContent = "";
}

function idle() {
  isRec = false; orbSkin(""); ico("◎");
  stl("Tap the orb or 🎙 to speak", "");
  hin("TAP ORB OR 🎙 TO SPEAK", false);
  sbs("Ready");
  var mb = document.getElementById("mic-btn"); if (mb) mb.classList.remove("rec");
}

function renderTx(items) {
  var sc = document.getElementById("txsc");
  var b  = document.querySelector(".mcnt");
  if (!sc) return;
  if (b) b.textContent = items.length;
  var tc = document.getElementById("turn-count");
  if (tc) tc.textContent = Math.floor(items.length / 2);
  if (!items.length) { sc.innerHTML = emptyTx(); return; }
  var srchVal = (document.getElementById("srch") || {}).value || "";
  sc.innerHTML = items.map(function (it) {
    var u   = it.role === "user";
    var txt = (it.text || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    var hidden = srchVal && !txt.toLowerCase().includes(srchVal.toLowerCase()) ? " hidden" : "";
    if (u) {
      return '<div class="mi' + hidden + '">'
           + '<div class="mh"><span class="mr u">YOU</span>'
           + '<span class="mt">' + (it.ts || "") + '</span></div>'
           + '<div class="bub u">' + txt + '</div></div>';
    } else {
      return '<div class="mi' + hidden + '">'
           + '<div class="mh"><span class="mr n">NOVA</span>'
           + '<span class="mt">' + (it.ts || "") + '</span></div>'
           + '<div class="bub-wrap"><div class="bub n">' + txt + '</div>'
           + '<button class="copy-btn" onclick="novaCopyBubble(this)">copy</button></div></div>';
    }
  }).join("");
  sc.scrollTop = sc.scrollHeight;
}

function emptyTx() {
  return '<div class="tempty">'
       + '<div class="tempty-i">✦</div>'
       + '<div class="tempty-t">Your conversation will appear here.<br>Tap the orb or say "Hey Nova" to begin.</div>'
       + '</div>';
}

/* ── Clock ── */
function clock() {
  var el = document.getElementById("sbclk"); if (!el) return;
  var d = new Date(), h = d.getHours(), m = d.getMinutes();
  var ap = h >= 12 ? "PM" : "AM"; h = h % 12 || 12;
  el.textContent = h + ":" + (m < 10 ? "0" : "") + m + " " + ap;
}

/* ── Keyboard shortcuts ── */
document.addEventListener("keydown", function (e) {
  if (e.code === "Space" && e.target === document.body) { e.preventDefault(); window.novaOrbClick(); }
  if (e.code === "Escape" && isRec) sendCmd({ cmd: "mic_toggle" });
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); window.novaSendText(); }
});

/* ── Auto-scroll transcript ── */
function autoScroll() {
  var sc = document.getElementById("txsc");
  if (sc && sc.scrollTop > sc.scrollHeight - sc.clientHeight - 100) {
    sc.scrollTop = sc.scrollHeight;
  }
}

/* ── Boot ── */
function boot() {
  console.log("[NOVA v12] Boot");
  setInterval(poll, 180);
  setInterval(autoScroll, 600);
  watchAudio();
  clock(); setInterval(clock, 10000);
  initWakeWord();

  var inp = document.getElementById("tinp");
  var snd = document.getElementById("tsnd");
  var mic = document.getElementById("mic-btn");
  var srch = document.getElementById("srch");

  if (inp && snd) {
    snd.addEventListener("click", window.novaSendText);
    inp.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); window.novaSendText(); }
    });
  }
  if (mic) mic.addEventListener("click", window.novaMicClick);
  if (srch) srch.addEventListener("input", window.novaSearch);

  console.log("[NOVA v12] Ready — SPACE to speak, ESC to cancel, Ctrl+Enter to send");
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", function () { setTimeout(boot, 600); });
else
  setTimeout(boot, 600);

})();
"""

# ════════════════════════════════════════════════════════════════════════════
# HTML
# ════════════════════════════════════════════════════════════════════════════
HTML = """
<div id="nv-root">

  <!-- ═══ SIDEBAR ═══ -->
  <aside id="sb">
    <!-- Brand / Title -->
    <div class="brand">
      <div class="bmark">✦</div>
      <div class="brand-text">
        <div class="bname">NOVA AI</div>
        <div class="btag">Voice Intelligence v12</div>
      </div>
    </div>

    <!-- System stats -->
    <div class="sbg">
      <div class="sbl">System</div>
      <div class="sr"><span class="srk">Engine</span><span class="srv">Groq Cloud</span></div>
      <div class="sr"><span class="srk">STT</span>   <span class="srv">Whisper-v3</span></div>
      <div class="sr"><span class="srk">LLM</span>   <span class="srv">Llama 3.3-70B</span></div>
      <div class="sr"><span class="srk">TTS</span>   <span class="srv">edge-tts</span></div>
      <div class="sr"><span class="srk">Silence</span><span class="srv">1.5 s auto</span></div>
      <div class="sr">
        <span class="srk">Status</span>
        <span class="srv live"><span class="ldot"></span><span id="sbst">Ready</span></span>
      </div>
      <div class="sr"><span class="srk">Time</span><span class="srv" id="sbclk">--:--</span></div>
    </div>

    <!-- Theme -->
    <div class="sbg">
      <div class="sbl">Appearance</div>
      <button id="theme-btn" onclick="novaToggleTheme()">☀  Light mode</button>
    </div>

    <!-- Voices — 3×2 grid, name only -->
    <div class="sbg">
      <div class="sbl">Voice</div>
      <div class="vgrid">
        <div class="vbtn on" data-voice="aria"    onclick="novaSetVoice('aria')">Aria</div>
        <div class="vbtn"    data-voice="sonia"   onclick="novaSetVoice('sonia')">Sonia</div>
        <div class="vbtn"    data-voice="neerja"  onclick="novaSetVoice('neerja')">Neerja</div>
        <div class="vbtn"    data-voice="guy"     onclick="novaSetVoice('guy')">Guy</div>
        <div class="vbtn"    data-voice="ryan"    onclick="novaSetVoice('ryan')">Ryan</div>
        <div class="vbtn"    data-voice="william" onclick="novaSetVoice('william')">William</div>
      </div>
    </div>

    <!-- API Key -->
    <div class="sbg">
      <div class="sbl">Groq API Key</div>
      <div class="kbox">
        <input id="key-inp" type="password" placeholder="gsk_…" autocomplete="off"
               onkeydown="if(event.key==='Enter') novaApplyKey()"/>
        <button id="key-apply" onclick="novaApplyKey()">Apply Key</button>
        <div id="key-status"></div>
      </div>
    </div>

    <div class="sbfoot">
      NOVA v12 &nbsp;·&nbsp; SPACE to speak · ESC to cancel<br>
      Ctrl+Enter to send · "Hey Nova" wake word<br>
      SQLite persistence · Web Audio waveform
    </div>
  </aside>

  <!-- ═══ CENTER ═══ -->
  <main id="ctr">
    <div class="hbadge"><div class="bdot"></div>AI Voice Assistant &nbsp;·&nbsp; Emotion-Aware &nbsp;·&nbsp; Always Listening</div>
    <h1 class="htitle">NOVA AI</h1>
    <p class="hsub">Speak naturally. NOVA listens, thinks, and responds in your voice.</p>

    <!-- Wake word badge -->
    <div id="wake-badge">Initialising wake word…</div>

    <!-- Orb (reduced size) -->
    <div class="orb-wrap" onclick="novaOrbClick()" title="Click to speak · or say Hey Nova">
      <div class="orb-ring or1"></div>
      <div class="orb-ring or2"></div>
      <div class="orb-ring or3"></div>
      <div class="orb-proc" id="orb-proc"></div>
      <div class="orb-sphere">
        <div class="orb-ico">◎</div>
      </div>
    </div>

    <!-- Waveform canvas -->
    <canvas id="waveform-canvas"></canvas>

    <div id="stl">Tap the orb or 🎙 to speak</div>
    <div id="sthin">TAP ORB OR 🎙 TO SPEAK</div>

    <!-- Silence countdown -->
    <div id="silw">
      <div id="siltrk"><div id="silfil"></div></div>
      <div id="sillbl"></div>
    </div>

    <!-- Live caption -->
    <div id="cap">
      <div class="crow" id="cyr" style="display:none">
        <span class="clbl you">You</span><span class="ctxt" id="cyt"></span>
      </div>
      <div class="csep" id="csep" style="display:none"></div>
      <div class="crow" id="cnr" style="display:none">
        <span class="clbl nova">Nova</span><span class="ctxt" id="cnt"></span>
      </div>
    </div>
    <div id="latr"></div>

    <!-- Input bar -->
    <div id="tbar">
      <input id="tinp" type="text" placeholder="Type here or tap 🎙 to speak…" autocomplete="off" maxlength="2048"/>
      <button id="mic-btn" title="Tap to record (SPACE)">🎙</button>
      <button id="tsnd"    title="Send (Enter)">➤</button>
    </div>
  </main>

  <!-- ═══ TRANSCRIPT PANEL ═══ -->
  <aside id="panel">
    <div class="phead">
      <div class="ptit">Transcript <span class="mcnt">0</span></div>
      <div style="font-size:7.5px;color:var(--t35);letter-spacing:.12em;white-space:nowrap">
        TURNS: <span id="turn-count">0</span>
      </div>
      <div class="pact">
        <button class="pbtn" onclick="novaClear()">✕ Clear</button>
        <button class="pbtn" onclick="novaExport()">↓ Save</button>
      </div>
    </div>
    <div id="srchbar">
      <input id="srch" type="text" placeholder="🔍  Search transcript…" maxlength="200"/>
    </div>
    <div id="txsc">
      <div class="tempty">
        <div class="tempty-i">✦</div>
        <div class="tempty-t">Your conversation will appear here.<br>Tap the orb or say "Hey Nova" to begin.</div>
      </div>
    </div>
  </aside>

</div>
"""


# ════════════════════════════════════════════════════════════════════════════
# BUILD APP
# ════════════════════════════════════════════════════════════════════════════
def build_app():
    with gr.Blocks(title="NOVA AI — Voice Assistant") as app:
        gr.HTML(HTML)
        cmd  = gr.Textbox(value="",   elem_id="_cmd",  label="cmd",  interactive=True)
        txst = gr.Textbox(value="[]", elem_id="_txst", label="tx",   interactive=False)
        stst = gr.Textbox(value="{}", elem_id="_stst", label="st",   interactive=False)
        aout = gr.Audio(autoplay=True, elem_id="_aout", label="audio")
        cmd.change(fn=handle_cmd, inputs=[cmd], outputs=[txst, aout, stst])
    return app


# ════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════
def startup():
    log_sep("NOVA v12 STARTUP")
    log("SYS", f"PID: {os.getpid()}")
    import sys
    log("SYS", f"Python: {sys.version.split()[0]}")

    db_init()

    checks = [
        ("pyaudio",   "PyAudio"),
        ("edge_tts",  "edge-tts"),
        ("gtts",      "gTTS"),
        ("groq",      "groq"),
        ("gradio",    "gradio"),
    ]
    for pkg, name in checks:
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", "?")
            log("SYS", f"  ✓ {name} {ver}")
        except ImportError:
            log("SYS", f"  ✗ {name} NOT installed", "WARN")

    k = os.environ.get("GROQ_API_KEY", "").strip()
    if not k:
        log("SYS", "No GROQ_API_KEY — enter it in the sidebar", "WARN")
    else:
        log("SYS", f"Groq key loaded (****{k[-4:]})")

    log_sep()
    log("SYS", "Install: pip install pyaudio groq gtts gradio edge-tts")
    log("SYS", "v12 changes:")
    log("SYS", "  • Orb reduced to 120px (was 190px)")
    log("SYS", "  • Sidebar: NOVA AI title prominently displayed")
    log("SYS", "  • Voice grid 3×2, name only (no region text)")
    log("SYS", "  • DB context manager, proper rollback")
    log("SYS", "  • API key format validation (gsk_ prefix)")
    log("SYS", "  • Input length capped (2048 chars)")
    log("SYS", "  • Text XSS-sanitized in transcript (& < > escaped)")
    log("SYS", "  • Clipboard fallback for older browsers")
    log("SYS", "  • Audio src cleanup on new playback")
    log("SYS", "  • Waveform null-checks prevent crashes")
    log("SYS", "  • Ctrl+Enter shortcut to send text")
    log("SYS", "  • TTS lock prevents concurrent synthesis")
    log("SYS", f"  • DB path: {DB_PATH}")
    log_sep("READY — http://127.0.0.1:7860")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# FLASK ROUTE (for integration with main server)
# ════════════════════════════════════════════════════════════════════════════

_gradio_launched = False
_gradio_lock = threading.Lock()

def launch_gradio():
    global _gradio_launched
    with _gradio_lock:
        if _gradio_launched:
            return
        _gradio_launched = True
    try:
        startup()
        gradio_app = build_app()
        gradio_app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            prevent_thread_lock=True,
            show_error=True,
            quiet=True,
            css=CSS,
            js=JS,
            allowed_paths=[],
        )
        log("SYS", "Gradio launched on http://127.0.0.1:7860")
    except Exception as e:
        log_err("SYS", e, "Gradio launch failed")

# Launch Gradio once when Flask imports this blueprint
launch_gradio()


@nova_ai.route('/')
def index():
    """Serve the Gradio app embedded in an iframe through Flask"""
    from flask import Response

    iframe_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NOVA AI - Voice Assistant</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { background: #0f0610; overflow: hidden; }
            iframe {
                position: fixed; top: 0; left: 0;
                width: 100vw; height: 100vh;
                border: none; display: block;
            }
            .loader {
                position: fixed; top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                color: #e8a0bf; font-size: 18px;
                text-align: center; z-index: 1000;
                font-family: system-ui, sans-serif;
            }
            .spinner {
                width: 50px; height: 50px;
                border: 3px solid rgba(232, 160, 191, 0.2);
                border-top-color: #e8a0bf;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="loader" id="loader">
            <div class="spinner"></div>
            Loading NOVA AI...
        </div>
        <iframe id="nova-frame" src="http://127.0.0.1:7860"
                onload="document.getElementById('loader').style.display='none'">
        </iframe>
    </body>
    </html>
    """
    return Response(iframe_html, mimetype='text/html')


# ════════════════════════════════════════════════════════════════════════════
# STANDALONE MODE
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    startup()
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        css=CSS,
        js=JS,
    )