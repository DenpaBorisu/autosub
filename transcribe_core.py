"""
Core transcription module with Bcut and Jianying ASR engines.

Supports auto-fallback between engines, audio chunking for long files,
and resume of interrupted transcriptions.
"""
import datetime
import hashlib
import hmac
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
import uuid
import zlib
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional, List, Tuple


def _no_window_kwargs() -> dict:
    """Return subprocess kwargs that suppress the console window on Windows.

    On Windows, launching a console-subsystem binary (ffmpeg.exe, ffprobe.exe)
    from a windowed (``--windowed`` PyInstaller) parent allocates a fresh
    console for every child, producing a rapidly flashing command window.
    ``CREATE_NO_WINDOW`` prevents that child console from being allocated.
    No-op on POSIX, where subprocesses never spawn a window.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


# =============================================================================
# HTTP Utilities (with automatic retry on transient errors)
# =============================================================================

# Retry only genuinely transient faults (resets / timeouts / 5xx). Retrying
# through 4xx blocks (412 risk-control, 429 rate-limit) tells the server we
# are an aggressive bot and turns soft throttles into hard IP bans, so those
# fail fast and trip the per-engine circuit breaker instead.
HTTP_RETRIES = 2
HTTP_RETRY_BASE_DELAY = 2.0

# Status codes that are terminal for the engine, never retried at HTTP level.
_TERMINAL_HTTP_CODES = frozenset({400, 401, 403, 412, 429})


class TerminalHttpError(RuntimeError):
    """Non-retryable HTTP rejection (auth/risk-control/rate-limit).

    Carries the HTTP status so the circuit breaker can decide how long to
    disable the engine.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _is_transient_error(e: Exception) -> bool:
    """Return True if the exception is likely a transient network/server error."""
    if isinstance(e, TerminalHttpError):
        return False
    if isinstance(e, urllib.error.URLError):
        return True
    if isinstance(e, TimeoutError):
        return True
    if isinstance(e, ConnectionError):
        return True
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500
    msg = str(e).lower()
    if any(k in msg for k in ("timeout", "timed out")):
        return True
    if "connection" in msg and any(k in msg for k in ("reset", "refused", "closed", "aborted")):
        return True
    if "temporarily" in msg or "unavailable" in msg:
        return True
    return False


def _retry(func: Callable, *args, retries: int = HTTP_RETRIES,
           base_delay: float = HTTP_RETRY_BASE_DELAY,
           retryable: Callable[[Exception], bool] = _is_transient_error,
           **kwargs):
    """Execute *func* with exponential backoff on retryable errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if retryable(e) and attempt < retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
                time.sleep(delay)
                continue
            raise
    raise last_exc  # pragma: no cover


def _raise_http_error(e: urllib.error.HTTPError, context: str) -> None:
    """Convert an HTTPError into a RuntimeError (or TerminalHttpError)."""
    body = e.read().decode("utf-8", errors="ignore")
    if e.code in _TERMINAL_HTTP_CODES:
        raise TerminalHttpError(e.code, f"{context}: HTTP {e.code}: {body}") from e
    raise RuntimeError(f"{context}: HTTP {e.code}: {body}") from e


def http_post(url: str, data: bytes = None, json_data: dict = None,
              headers: dict = None) -> dict:
    """Make HTTP POST request and return JSON response."""
    if headers is None:
        headers = {}

    body = None
    if json_data:
        body = json.dumps(json_data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif data:
        body = data

    def _do():
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            _raise_http_error(e, "Request failed")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    return _retry(_do)


def http_put(url: str, data: bytes, headers: dict = None) -> dict:
    """Make HTTP PUT request (for multipart upload)."""
    if headers is None:
        headers = {}

    def _do():
        req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return dict(resp.getheaders())
        except urllib.error.HTTPError as e:
            _raise_http_error(e, "Upload failed")

    return _retry(_do)


def http_get(url: str, params: dict = None, headers: dict = None) -> dict:
    """Make HTTP GET request and return JSON response."""
    if headers is None:
        headers = {}

    full_url = url
    if params:
        from urllib.parse import urlencode
        full_url = f"{url}?{urlencode(params)}"

    def _do():
        req = urllib.request.Request(full_url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            _raise_http_error(e, "Request failed")

    return _retry(_do)


def http_post_raw(url: str, data: bytes = None, json_data: dict = None,
                  headers: dict = None, timeout: int = 30):
    """Make HTTP POST and return raw response (dict or bytes depending on content-type)."""
    if headers is None:
        headers = {}

    body = None
    if json_data:
        body = json.dumps(json_data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif data:
        body = data

    def _do():
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    return json.load(resp)
                return resp.read()
        except urllib.error.HTTPError as e:
            _raise_http_error(e, "Request failed")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    return _retry(_do)


def http_put_json(url: str, data: bytes, headers: dict = None) -> dict:
    """Make HTTP PUT request, parse response as JSON."""
    if headers is None:
        headers = {}

    def _do():
        req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "json" in content_type:
                    return json.load(resp)
                return {"_http_ok": resp.status == 200}
        except urllib.error.HTTPError as e:
            _raise_http_error(
                e,
                f"Upload failed: {e.read().decode('utf-8', errors='ignore')}")

    return _retry(_do)


# =============================================================================
# Request Pacing & Engine Circuit Breaker
# =============================================================================
# These unofficial endpoints run risk control: metronomic 1 Hz polling and
# retry storms are the fastest way to get throttled (or 412'd). Everything
# here exists to keep total request volume low and non-uniform.

POLL_GRACE_SEC = 10.0          # wait after task creation before first poll
POLL_INTERVAL_START_SEC = 5.0  # adaptive poll interval, grows to the cap
POLL_INTERVAL_MAX_SEC = 15.0
INTER_CHUNK_DELAY_RANGE = (3.0, 6.0)  # jittered pause between chunk submissions

CIRCUIT_FAILURE_THRESHOLD = 3  # consecutive terminal failures trip the breaker
CIRCUIT_COOLDOWN_SEC = 20 * 60  # engine disabled for this long once tripped

_engine_state: dict = {}
_engine_state_lock = threading.Lock()


def _jitter(delay: float, spread: float = 0.2) -> float:
    """Scale a delay by ±spread so traffic is never metronomic."""
    return delay * random.uniform(1.0 - spread, 1.0 + spread)


def _next_poll_interval(current: float) -> float:
    """Grow the poll interval toward the cap, with jitter."""
    return min(_jitter(max(current, POLL_INTERVAL_START_SEC) * 1.5), POLL_INTERVAL_MAX_SEC)


def engine_available(name: str) -> bool:
    """True unless the engine's circuit breaker is open (in cooldown)."""
    with _engine_state_lock:
        state = _engine_state.get(name)
        if state is None:
            return True
        return time.time() >= state["disabled_until"]


def engine_cooldown_remaining(name: str) -> float:
    with _engine_state_lock:
        state = _engine_state.get(name)
        if state is None:
            return 0.0
        return max(0.0, state["disabled_until"] - time.time())


def record_engine_success(name: str) -> None:
    with _engine_state_lock:
        _engine_state.setdefault(name, {"consecutive": 0, "disabled_until": 0.0})
        _engine_state[name]["consecutive"] = 0


def record_engine_failure(name: str, error: Exception) -> None:
    """Count a terminal engine failure; trip the breaker at the threshold.

    An HTTP block (412/429/...) trips the breaker immediately — the endpoint
    has told us to stop, so we stop instead of retrying through it.
    """
    with _engine_state_lock:
        state = _engine_state.setdefault(name, {"consecutive": 0, "disabled_until": 0.0})
        state["consecutive"] += 1
        if isinstance(error, TerminalHttpError) or state["consecutive"] >= CIRCUIT_FAILURE_THRESHOLD:
            state["disabled_until"] = time.time() + CIRCUIT_COOLDOWN_SEC
            state["consecutive"] = 0


# =============================================================================
# Bcut API Implementation
# =============================================================================

class BcutASR:
    """Bilibili Bcut ASR API - Free, fast Chinese/English transcription"""

    API_BASE_URL = "https://member.bilibili.com/x/bcut/rubick-interface"
    API_REQ_UPLOAD = f"{API_BASE_URL}/resource/create"
    API_COMMIT_UPLOAD = f"{API_BASE_URL}/resource/create/complete"
    API_CREATE_TASK = f"{API_BASE_URL}/task"
    API_QUERY_RESULT = f"{API_BASE_URL}/task/result"

    HEADERS = {
        "User-Agent": "Bilibili/1.0.0 (https://www.bilibili.com)",
    }

    def __init__(self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None,
                 model_id: str = "8"):
        self.audio_path = audio_path
        self.task_id = None
        self.etags = []
        self.progress_callback = progress_callback
        self.model_id = model_id

    def _log(self, message: str) -> None:
        """Log progress message."""
        if self.progress_callback:
            self.progress_callback(message)

    def _load_audio(self) -> bytes:
        """Load audio file as binary data."""
        with open(self.audio_path, "rb") as f:
            return f.read()

    def upload(self) -> str:
        """Upload audio file and return download URL."""
        file_binary = self._load_audio()
        file_size = len(file_binary)

        self._log(f"Uploading ({file_size / 1024 / 1024:.1f} MB)...")

        # Request upload authorization
        upload_req = {
            "type": 2,
            "name": "audio.mp3",
            "size": file_size,
            "ResourceFileType": "mp3",
            "model_id": self.model_id,
        }

        resp = http_post(self.API_REQ_UPLOAD, json_data=upload_req, headers=self.HEADERS)
        resp_data = resp["data"]

        in_boss_key = resp_data["in_boss_key"]
        resource_id = resp_data["resource_id"]
        upload_id = resp_data["upload_id"]
        upload_urls = resp_data["upload_urls"]
        per_size = resp_data["per_size"]
        clips = len(upload_urls)

        # Upload parts
        for clip in range(clips):
            start_range = clip * per_size
            end_range = min((clip + 1) * per_size, file_size)
            chunk = file_binary[start_range:end_range]

            self._log(f"Uploading part {clip + 1}/{clips}...")
            resp_headers = http_put(upload_urls[clip], data=chunk, headers=self.HEADERS)

            etag = resp_headers.get("Etag")
            if etag:
                self.etags.append(etag)

        # Commit upload
        commit_req = {
            "InBossKey": in_boss_key,
            "ResourceId": resource_id,
            "Etags": ",".join(self.etags) if self.etags else "",
            "UploadId": upload_id,
            "model_id": self.model_id,
        }

        resp = http_post(self.API_COMMIT_UPLOAD, json_data=commit_req, headers=self.HEADERS)
        return resp["data"]["download_url"]

    def create_task(self, download_url: str) -> str:
        """Create ASR task and return task ID."""
        task_req = {
            "resource": download_url,
            "model_id": self.model_id,
        }

        resp = http_post(self.API_CREATE_TASK, json_data=task_req, headers=self.HEADERS)
        self.task_id = resp["data"]["task_id"]
        return self.task_id

    def get_result(self) -> dict:
        """Poll for transcription result with adaptive, jittered intervals.

        Server-side ASR on a several-minute chunk takes a while, so we sleep a
        grace period before the first poll and then grow the interval toward a
        cap — a handful of requests per chunk instead of hundreds.
        """
        self._log("Transcribing (may take 1-5 minutes)...")
        time.sleep(_jitter(POLL_GRACE_SEC))

        consecutive_errors = 0
        interval = POLL_INTERVAL_START_SEC
        deadline = time.time() + 15 * 60  # overall budget ~15 min
        i = 0
        while time.time() < deadline:
            try:
                resp = http_get(
                    self.API_QUERY_RESULT,
                    params={"model_id": self.model_id, "task_id": self.task_id},
                    headers=self.HEADERS
                )
                consecutive_errors = 0
            except TerminalHttpError:
                raise
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 6:
                    raise RuntimeError(
                        f"ASR polling failed after {consecutive_errors} consecutive errors: {e}"
                    ) from e
                self._log(f"Polling error (will retry): {e}")
                time.sleep(min(2 ** consecutive_errors, 30))
                continue

            task_resp = resp.get("data") or {}

            state = task_resp.get("state")
            if state == 4:  # Complete
                result = task_resp.get("result")
                if result is None:
                    raise RuntimeError("ASR completed but no result data returned")
                return json.loads(result) if isinstance(result, str) else result
            elif state in (3, 5):  # Failed or cancelled — stop immediately
                raise RuntimeError(f"ASR task failed with state: {state}")

            dots = i % 4
            if self.progress_callback:
                self._log(f"Transcribing{'.' * dots}{' ' * (3 - dots)}")
            time.sleep(interval)
            interval = _next_poll_interval(interval)
            i += 1

        raise RuntimeError("ASR task timeout")

    def transcribe(self) -> List[dict]:
        """Full transcription workflow. Returns list of utterances."""
        download_url = self.upload()
        self._log("Upload complete!")

        self.create_task(download_url)

        result = self.get_result()
        self._log("Transcription complete!")

        return result.get("utterances", [])


# =============================================================================
# Jianying (CapCut) ASR Implementation
# =============================================================================

def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(secret_key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    k_date = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    return _sign(k_service, "aws4_request")


def _aws_signature(secret_key: str, request_parameters: str, headers: dict,
                   method: str = "GET", payload: str = "",
                   region: str = "cn", service: str = "vod") -> str:
    canonical_uri = "/"
    canonical_querystring = request_parameters
    canonical_headers = "\n".join([f"{k}:{v}" for k, v in headers.items()]) + "\n"
    signed_headers = ";".join(headers.keys())
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    amzdate = headers["x-amz-date"]
    datestamp = amzdate.split("T")[0]
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{datestamp}/{region}/{service}/aws4_request"
    string_to_sign = f"{algorithm}\n{amzdate}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

    signing_key = _get_signature_key(secret_key, datestamp, region, service)
    return hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


# Per-run device id: a random 16-digit numeric string generated once per
# process and reused for the session. A permanently fixed tdid shared by
# every user of a tool is trivially blacklistable; per-request rotation
# looks even less like a real device.
_SESSION_TDID = "".join(str(random.randint(0, 9)) for _ in range(16))

# Client baseline the sign-ver:1 scheme is tied to. The submit endpoint now
# rejects genuinely old versions with ret 3510 ("block low version"), while
# newer values would enforce stronger native signing (x-argus/x-ladon) we
# cannot mimic — 6.6.0 is the known-good middle ground.
_JY_PF = "4"
_JY_APPVR = "6.6.0"
_JY_UA = ("Cronet/TTNetVersion:01594da2 2023-03-14 "
          "QuicVersion:46688bb4 2022-11-28")


class JianyingASR:
    """JianYing (CapCut) ASR API - Free Chinese/English transcription (fallback engine).

    Uses ByteDance's JianYing cloud ASR service with AWS S3-style upload.
    Request signing is computed locally (sign-ver: 1 legacy scheme reverse-
    engineered in the upstream bk_asr project) — no external sign service.
    """

    def __init__(self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None,
                 start_time: int = 0, end_time: int = 6000):
        self.audio_path = audio_path
        self.progress_callback = progress_callback
        self.start_time = start_time
        self.end_time = end_time
        self.tdid = _SESSION_TDID

        # Populated during upload
        self.session_token = None
        self.secret_key = None
        self.access_key = None
        self.store_uri = None
        self.auth = None
        self.upload_id = None
        self.upload_hosts = None

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _load_audio(self) -> bytes:
        with open(self.audio_path, "rb") as f:
            return f.read()

    @staticmethod
    def _generate_sign(url_path: str) -> Tuple[str, str]:
        """Compute the sign-ver:1 request signature locally.

        Scheme (from upstream bk_asr, reverse-engineered from the JianYing PC
        client): md5 over "9e2c|{last 7 chars of url path}|pf|appvr|time|tdid|11ac".
        Sensitive to system clock drift — keep the machine NTP-synced.
        """
        current_time = str(int(time.time()))
        sign_str = f"9e2c|{url_path[-7:]}|{_JY_PF}|{_JY_APPVR}|{current_time}|{_SESSION_TDID}|11ac"
        sign = hashlib.md5(sign_str.encode()).hexdigest()
        return sign.lower(), current_time

    def _build_headers(self, device_time: str, sign: str) -> dict:
        return {
            "User-Agent": _JY_UA,
            "appvr": _JY_APPVR,
            "device-time": str(device_time),
            "pf": _JY_PF,
            "sign": sign,
            "sign-ver": "1",
            "tdid": self.tdid,
        }

    def _upload_sign(self) -> None:
        url_path = "/lv/v1/upload_sign"
        sign, device_time = self._generate_sign(url_path)
        headers = self._build_headers(device_time, sign)
        payload = json.dumps({"biz": "pc-recognition"})
        resp = http_post_raw(
            "https://lv-pc-api-sinfonlinec.ulikecam.com/lv/v1/upload_sign",
            data=payload.encode("utf-8"), headers=headers
        )
        login_data = resp if isinstance(resp, dict) else json.loads(resp)
        self.access_key = login_data["data"]["access_key_id"]
        self.secret_key = login_data["data"]["secret_access_key"]
        self.session_token = login_data["data"]["session_token"]

    def _upload_auth(self) -> None:
        file_binary = self._load_audio()
        file_size = len(file_binary)
        request_parameters = (
            f"Action=ApplyUploadInner&FileSize={file_size}&FileType=object"
            f"&IsInner=1&SpaceName=lv-mac-recognition&Version=2020-11-19&s=5y0udbjapi"
        )

        t = datetime.datetime.now(datetime.timezone.utc)
        amz_date = t.strftime("%Y%m%dT%H%M%SZ")
        headers = {"x-amz-date": amz_date, "x-amz-security-token": self.session_token}

        signature = _aws_signature(self.secret_key, request_parameters, headers, region="cn", service="vod")
        authorization = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{amz_date[:8]}/cn/vod/aws4_request, "
            f"SignedHeaders=x-amz-date;x-amz-security-token, Signature={signature}"
        )
        headers["authorization"] = authorization

        resp = http_get(
            f"https://vod.bytedanceapi.com/?{request_parameters}",
            params=None, headers=headers
        )
        store_infos = resp

        infos = store_infos["Result"]["UploadAddress"]["StoreInfos"][0]
        self.store_uri = infos["StoreUri"]
        self.auth = infos["Auth"]
        self.upload_id = infos["UploadID"]
        self.upload_hosts = store_infos["Result"]["UploadAddress"]["UploadHosts"][0]

    def _upload_file(self) -> None:
        url = f"https://{self.upload_hosts}/{self.store_uri}?partNumber=1&uploadID={self.upload_id}"
        file_binary = self._load_audio()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Authorization": self.auth,
            "Content-CRC32": format(zlib.crc32(file_binary) & 0xFFFFFFFF, "08x"),
        }
        resp = http_put_json(url, data=file_binary, headers=headers)
        # ByteDance upload API convention: success=0 means OK.
        # http_put_json synthesises {"_http_ok": True} when the response body
        # is not JSON (HTTP status already validated at that layer), so treat
        # both as success.  Only a numeric non-zero ``success`` is a real error.
        success_val = resp.get("success")
        if isinstance(success_val, int) and not isinstance(success_val, bool) and success_val != 0:
            raise RuntimeError(f"File upload failed: {resp}")

    def _upload_check(self) -> None:
        url = f"https://{self.upload_hosts}/{self.store_uri}?uploadID={self.upload_id}"
        file_binary = self._load_audio()
        crc = format(zlib.crc32(file_binary) & 0xFFFFFFFF, "08x")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Authorization": self.auth,
            "Content-CRC32": crc,
        }
        http_post_raw(url, data=f"1:{crc}".encode("utf-8"), headers=headers)

    def submit(self) -> str:
        url = "https://lv-pc-api-sinfonlinec.ulikecam.com/lv/v1/audio_subtitle/submit"
        payload = {
            "adjust_endtime": 200,
            "audio": self.store_uri,
            "caption_type": 2,
            "client_request_id": "45faf98c-160f-4fae-a649-6d89b0fe35be",
            "max_lines": 1,
            "songs_info": [{"end_time": self.end_time, "id": "", "start_time": self.start_time}],
            "words_per_line": 16,
        }
        sign, device_time = self._generate_sign("/lv/v1/audio_subtitle/submit")
        headers = self._build_headers(device_time, sign)
        resp = http_post_raw(url, json_data=payload, headers=headers)
        resp_data = resp if isinstance(resp, dict) else json.loads(resp)
        if resp_data.get("ret") != "0":
            raise RuntimeError(f"Jianying API error: {resp_data.get('errmsg', 'Unknown')} (ret: {resp_data.get('ret')})")
        return resp_data["data"]["id"]

    def query(self, query_id: str) -> dict:
        url = "https://lv-pc-api-sinfonlinec.ulikecam.com/lv/v1/audio_subtitle/query"
        payload = {"id": query_id, "pack_options": {"need_attribute": True}}
        sign, device_time = self._generate_sign("/lv/v1/audio_subtitle/query")
        headers = self._build_headers(device_time, sign)
        resp = http_post_raw(url, json_data=payload, headers=headers)
        resp_data = resp if isinstance(resp, dict) else json.loads(resp)
        ret = resp_data.get("ret")
        if ret is not None and str(ret) != "0":
            data = resp_data.get("data") or {}
            if data.get("utterances") is None:
                raise RuntimeError(
                    f"Jianying query error: {resp_data.get('errmsg', 'Unknown')} (ret: {ret})"
                )
        return resp_data

    def transcribe(self) -> List[dict]:
        """Full Jianying transcription workflow."""
        self._log("Jianying: Getting upload credentials...")
        self._upload_sign()
        self._upload_auth()

        self._log("Jianying: Uploading audio...")
        self._upload_file()
        self._upload_check()
        self._log("Jianying: Upload complete!")

        query_id = self.submit()

        self._log("Jianying: Transcribing...")
        time.sleep(_jitter(POLL_GRACE_SEC))
        consecutive_errors = 0
        interval = POLL_INTERVAL_START_SEC
        deadline = time.time() + 15 * 60
        i = 0
        while time.time() < deadline:
            try:
                resp_data = self.query(query_id)
                consecutive_errors = 0
            except TerminalHttpError:
                raise
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 6:
                    raise RuntimeError(
                        f"Jianying polling failed after {consecutive_errors} errors: {e}"
                    ) from e
                self._log(f"Jianying: polling error (will retry): {e}")
                time.sleep(min(2 ** consecutive_errors, 30))
                continue

            data = resp_data.get("data") or {}
            utterances = data.get("utterances")
            if utterances is not None:
                self._log("Jianying: Transcription complete!")
                return utterances
            # Explicit failure status — stop instead of polling to timeout.
            if str(resp_data.get("ret", "0")) not in ("0", "None"):
                if data.get("progress") in ("failed", "fail", "-1"):
                    raise RuntimeError(
                        f"Jianying task failed: {resp_data.get('errmsg', 'Unknown')}")
            dots = i % 4
            self._log(f"Jianying: Waiting{'.' * dots}{' ' * (3 - dots)}")
            time.sleep(interval)
            interval = _next_poll_interval(interval)
            i += 1

        raise RuntimeError("Jianying transcription timeout")


# =============================================================================
# KuaiShou ASR Implementation
# =============================================================================

def _multipart_body(field_name: str, filename: str, content_type: str,
                    payload: bytes, fields: dict) -> Tuple[bytes, str]:
    """Build a multipart/form-data body with urllib (no requests dependency)."""
    boundary = f"----AutoSubBoundary{uuid.uuid4().hex}"
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; "
         f"filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").encode()
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class KuaiShouASR:
    """KuaiShou ASR - currently DISABLED and not wired into any engine chain.

    The reverse-engineered endpoint returns "effect disabled" (code 501)
    server-side, so every call wastes an upload and retries. Kept here,
    unreferenced, for the day the endpoint works again.

    One synchronous multipart POST; the response already contains the
    utterances, so there is no polling and almost no rate-limit surface.
    Ported from the upstream bk_asr project.
    """

    API_URL = "https://ai.kuaishou.com/api/effects/subtitle_generate"

    def __init__(self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None):
        self.audio_path = audio_path
        self.progress_callback = progress_callback

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def transcribe(self) -> List[dict]:
        self._log("KuaiShou: uploading audio (synchronous engine)...")
        with open(self.audio_path, "rb") as f:
            payload = f.read()
        body, content_type = _multipart_body(
            "file", "audio.mp3", "audio/mpeg", payload, {"typeId": "1"})
        headers = {
            "Content-Type": content_type,
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
            "Origin": "https://ai.kuaishou.com",
            "Referer": "https://ai.kuaishou.com/",
        }

        def _do():
            req = urllib.request.Request(self.API_URL, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                _raise_http_error(e, "KuaiShou request failed")

        resp_data = _retry(_do)
        # 200-with-error-code responses: 501 means the effect is disabled
        # server-side (observed 2026-08) — terminal, trip the circuit breaker
        # so the fallback chain skips KuaiShou instead of re-probing per chunk.
        if isinstance(resp_data, dict) and resp_data.get("code") not in (None, 0, 200):
            code = resp_data.get("code")
            if code == 501:
                raise TerminalHttpError(
                    501, f"KuaiShou effect disabled server-side: {resp_data.get('msg')}")
            raise RuntimeError(
                f"KuaiShou error {code}: {resp_data.get('msg', json.dumps(resp_data)[:200])}")
        data = (resp_data or {}).get("data") or {}
        utterances = data.get("text")
        if utterances is None:
            raise RuntimeError(
                f"KuaiShou returned no transcript: {json.dumps(resp_data)[:300]}")
        # Normalize 'text' -> 'transcript' via _normalize_utterances downstream.
        self._log("KuaiShou: transcription complete!")
        return utterances


# =============================================================================
# Sherpa-ONNX Local ASR Implementation
# =============================================================================

_local_recognizer = None
_local_recognizer_lock = threading.Lock()
_local_vad = None

_LOCAL_MAX_CHARS_PER_LINE = 30
_LOCAL_NUM_THREADS = 8

# Feed audio to the VAD in 0.1 s windows so its internal circular buffer never
# overflows on multi-hour files.
_VAD_WINDOW_SAMPLES = 1600


def _get_sherpa_recognizer():
    """Build (once) and return the cached local OfflineRecognizer.

    Uses the FireRedASR2 CTC bilingual zh-en model (fully offline). Unlike the
    old streaming transducer this is a non-streaming recognizer, so a fresh
    stream is created per chunk/audio file.
    """
    global _local_recognizer
    if _local_recognizer is not None:
        return _local_recognizer
    with _local_recognizer_lock:
        if _local_recognizer is None:
            try:
                import sherpa_onnx
            except ImportError as e:
                raise RuntimeError(
                    "sherpa-onnx is not installed. Run 'uv sync' to install it."
                ) from e
            try:
                from local_model import model_paths
                paths = model_paths()
            except FileNotFoundError as e:
                raise RuntimeError(
                    "Local model not downloaded. Use the 'Download local model' "
                    "button in the app first."
                ) from e
            _local_recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
                model=paths["model"],
                tokens=paths["tokens"],
                num_threads=_LOCAL_NUM_THREADS,
            )
        return _local_recognizer


def _get_sherpa_vad():
    """Build (once) and return the cached silero VoiceActivityDetector."""
    global _local_vad
    if _local_vad is not None:
        return _local_vad
    with _local_recognizer_lock:
        if _local_vad is None:
            try:
                import sherpa_onnx
            except ImportError as e:
                raise RuntimeError(
                    "sherpa-onnx is not installed. Run 'uv sync' to install it."
                ) from e
            try:
                from local_model import vad_model_path
                vad_path = vad_model_path()
            except FileNotFoundError as e:
                raise RuntimeError(
                    "VAD model missing. Use the 'Download local model' button "
                    "in the app first."
                ) from e
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = vad_path
            vad_config.silero_vad.threshold = 0.5
            vad_config.silero_vad.min_silence_duration = 0.5
            vad_config.silero_vad.min_speech_duration = 0.25
            vad_config.silero_vad.max_speech_duration = 20.0
            vad_config.silero_vad.window_size = 512
            vad_config.sample_rate = 16000
            vad_config.num_threads = 1
            vad_config.validate()
            _local_vad = sherpa_onnx.VoiceActivityDetector(vad_config)
        return _local_vad


def _vad_segments(samples, sample_rate: int) -> List[Tuple[int, int]]:
    """Split *samples* into speech segments using silero VAD.

    Returns a list of ``(start_sample, end_sample)`` 16 kHz ranges (exclusive
    end). Each segment is short enough for the non-streaming FireRedASR2 CTC
    encoder to decode without exhausting memory.
    """
    vad = _get_sherpa_vad()
    segments: List[Tuple[int, int]] = []
    for i in range(0, len(samples), _VAD_WINDOW_SAMPLES):
        vad.accept_waveform(samples[i:i + _VAD_WINDOW_SAMPLES])
        while not vad.empty():
            # seg.start is already relative to the stream start (global offset).
            seg = vad.front
            start = seg.start
            end = start + len(seg.samples)
            segments.append((start, end))
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        start = seg.start
        end = start + len(seg.samples)
        segments.append((start, end))
        vad.pop()
    vad.reset()
    return segments


def _local_split_utterances(tokens: List[str], timestamps: List[float]) -> List[dict]:
    """Group ASR tokens into subtitle lines with start/end times (ms).

    The bilingual model emits Chinese character tokens plus English BPE pieces
    where a leading '▁' (U+2581) marks a word boundary (becomes a space); the
    FireRedASR2 CTC model instead emits English pieces with an already-present
    leading space. Special tokens like <blk>/<sos/eos>/<unk> are dropped.
    """
    utterances: List[dict] = []
    buf: List[str] = []
    buf_len = 0
    buf_start: Optional[int] = None

    def flush(end_ts: float) -> None:
        nonlocal buf, buf_len, buf_start
        if not buf:
            return
        text = "".join(buf)
        if text:
            utterances.append({
                "transcript": text,
                "start_time": int(timestamps[buf_start] * 1000),
                "end_time": int(end_ts * 1000),
            })
        buf = []
        buf_len = 0
        buf_start = None

    for i, tok in enumerate(tokens):
        if tok.startswith("<"):  # skip <blk>, <sos/eos>, <unk>
            continue
        if tok.startswith("▁"):  # English word-start marker -> space
            tok = " " + tok[1:]
        if buf_start is None:
            buf_start = i
        buf.append(tok)
        buf_len += len(tok)
        if buf_len >= _LOCAL_MAX_CHARS_PER_LINE:
            flush(timestamps[i])
    if buf:
        flush(timestamps[-1] + 0.3)
    return utterances


class SherpaLocalASR:
    """Local sherpa-onnx transcription — fully offline, Chinese + English.

    Uses the sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25 (FireRedASR2
    CTC) offline model via OfflineRecognizer. Outputs real Chinese characters
    and English words with per-token timestamps.
    """

    def __init__(self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None,
                 ffmpeg_path: str = "ffmpeg", num_threads: int = 2):
        self.audio_path = audio_path
        self.progress_callback = progress_callback
        self.ffmpeg_path = ffmpeg_path
        self.num_threads = num_threads

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _load_audio(self):
        """Decode audio to 16 kHz mono float32 samples via ffmpeg."""
        cmd = [
            self.ffmpeg_path, "-v", "error", "-i", self.audio_path,
            "-ac", "1", "-ar", "16000", "-f", "f32le", "-",
        ]
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, **_no_window_kwargs()
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Local: failed to decode audio: {e}") from e
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError("Local engine requires numpy (run 'uv sync')") from e
        return np.frombuffer(result.stdout, dtype=np.float32), 16000

    def transcribe(self) -> List[dict]:
        """Transcribe the audio file offline. Returns list of utterances.

        The non-streaming FireRedASR2 CTC encoder can only handle bounded
        durations, so the audio is first split into speech segments with a tiny
        silero VAD model. Each segment is decoded independently and its
        timestamps are offset by the segment's start time.
        """
        self._log("Local: Transcribing offline...")
        recognizer = _get_sherpa_recognizer()
        samples, sample_rate = self._load_audio()

        segments = _vad_segments(samples, sample_rate)
        self._log(f"Local: VAD found {len(segments)} speech segments")
        total_ms = int(len(samples) / sample_rate * 1000)

        utterances: List[dict] = []
        for seg_index, (start_sample, end_sample) in enumerate(segments, start=1):
            seg_audio = samples[start_sample:end_sample]
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, seg_audio)
            recognizer.decode_stream(stream)

            result = stream.result
            tokens = list(result.tokens)
            timestamps = list(result.timestamps)
            seg_uts = _local_split_utterances(tokens, timestamps)
            offset_ms = int(start_sample / sample_rate * 1000)
            for u in seg_uts:
                u["start_time"] += offset_ms
                u["end_time"] += offset_ms
                if u["end_time"] > total_ms:
                    u["end_time"] = total_ms
            utterances.extend(seg_uts)
            if seg_index % 25 == 0 or seg_index == len(segments):
                self._log(f"Local: {seg_index}/{len(segments)} segments")

        self._log(f"Local: done ({len(utterances)} segments)")
        return utterances


# =============================================================================
# Utterance Normalization & Auto-Fallback
# =============================================================================

def _normalize_utterances(utterances: List[dict]) -> List[dict]:
    """Normalize utterance format across engines (Bcut uses 'transcript', Jianying uses 'text')."""
    for u in utterances:
        if "transcript" not in u and "text" in u:
            u["transcript"] = u["text"]
    return utterances


class AutoASR:
    """Auto-fallback ASR: Bcut first, falls back to JianYing on failure,
    with a per-engine circuit breaker.

    Once an engine trips its breaker (terminal HTTP block or repeated
    failures), subsequent chunks skip it for the cooldown instead of
    hammering a blocked endpoint chunk after chunk. (KuaiShou is disabled —
    its endpoint is dead server-side; see KuaiShouASR.)
    """

    def __init__(self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None,
                 model_id: str = "8", start_time: int = 0, end_time: int = 6000):
        self.audio_path = audio_path
        self.progress_callback = progress_callback
        self.model_id = model_id
        self.start_time = start_time
        self.end_time = end_time

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _engine_factories(self):
        # KuaiShou disabled: endpoint returns "effect disabled" (code 501)
        # and just wastes time. Chain is Bcut → JianYing.
        return [
            ("Bcut", lambda: BcutASR(self.audio_path, self.progress_callback, self.model_id)),
            ("JianYing", lambda: JianyingASR(
                self.audio_path, self.progress_callback, self.start_time, self.end_time)),
        ]

    def transcribe(self) -> List[dict]:
        errors: List[str] = []
        for name, factory in self._engine_factories():
            if not engine_available(name):
                cooldown = engine_cooldown_remaining(name)
                self._log(f"{name}: skipped (circuit breaker open, {cooldown / 60:.0f} min left)")
                errors.append(f"{name}: in cooldown")
                continue
            try:
                self._log(f"Trying {name} ASR...")
                utterances = factory().transcribe()
                record_engine_success(name)
                self._log(f"{name} ASR succeeded")
                return _normalize_utterances(utterances)
            except Exception as e:
                record_engine_failure(name, e)
                errors.append(f"{name}: {e}")
                self._log(f"{name} failed: {e}")

        raise RuntimeError("All ASR engines failed. " + " | ".join(errors))


# =============================================================================
# SRT Generation
# =============================================================================

def milliseconds_to_srt(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm)."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def utterances_to_srt(utterances: List[dict]) -> str:
    """Convert Bcut utterances to SRT format."""
    srt_lines = []

    index = 0
    for utterance in utterances:
        text = utterance.get("transcript", "").strip()
        if not text:
            continue

        index += 1
        start_time = utterance.get("start_time", 0)
        end_time = utterance.get("end_time", 0)

        timestamp = f"{milliseconds_to_srt(int(start_time))} --> {milliseconds_to_srt(int(end_time))}"
        srt_lines.append(f"{index}\n{timestamp}\n{text}\n")

    return "\n".join(srt_lines)


# =============================================================================
# Audio Extraction & Conversion
# =============================================================================

def get_ffmpeg_path() -> str:
    """Resolve the ffmpeg executable path.

    When running as a frozen executable (PyInstaller/Nuitka), bundled
    ffmpeg/ffprobe binaries are extracted to a temp dir (sys._MEIPASS)
    or sit next to the executable.  When running from source, returns
    "ffmpeg" so the system PATH is used.
    """
    if getattr(sys, "frozen", False):
        exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        # PyInstaller onefile extracts to _MEIPASS
        base = getattr(sys, "_MEIPASS", None)
        if base:
            candidate = os.path.join(base, exe_name)
            if os.path.isfile(candidate):
                return candidate
        # Nuitka / PyInstaller onedir: next to executable
        exe_dir = os.path.dirname(sys.executable)
        candidate = os.path.join(exe_dir, exe_name)
        if os.path.isfile(candidate):
            return candidate
    return "ffmpeg"


def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            **_no_window_kwargs()
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_to_mp3(input_path: str, output_path: str,
                         ffmpeg_path: str = "ffmpeg") -> bool:
    """Extract audio from video file and convert to MP3. Validates output."""
    try:
        subprocess.run(
            [
                ffmpeg_path, "-y", "-i", input_path,
                "-vn", "-acodec", "libmp3lame", "-b:a", "48k",
                "-ar", "16000", "-ac", "1",
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            **_no_window_kwargs()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    # Verify the output is a valid, non-empty audio file
    if not Path(output_path).exists() or Path(output_path).stat().st_size < 1024:
        return False
    if get_audio_duration(output_path, ffmpeg_path) <= 0:
        return False
    return True


def analyze_loudness(audio_path: str,
                     ffmpeg_path: str = "ffmpeg") -> Tuple[Optional[float], Optional[float]]:
    """Measure integrated loudness (LUFS) and loudness range (LU) via ebur128.

    Single cheap analysis pass (no audio output). Returns (integrated_lufs,
    lra) or (None, None) if measurement fails.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-v", "info", "-i", audio_path,
             "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
            **_no_window_kwargs()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None

    integrated: Optional[float] = None
    lra: Optional[float] = None
    for line in (result.stderr or "").splitlines():
        m = re.search(r"\bI:\s+(-?[\d.]+)\s+LUFS", line)
        if m:
            integrated = float(m.group(1))
        m = re.search(r"\bLRA:\s+([\d.]+)\s+LU", line)
        if m:
            lra = float(m.group(1))
    return integrated, lra


class NormalizeStatus(IntEnum):
    """Result of the audio normalization step."""
    NORMALIZED = 0  # a normalized output file was written
    SKIPPED = 1     # already well-leveled — no output file produced
    FAILED = 2


def normalize_audio_loudness(input_path: str, output_path: str,
                             ffmpeg_path: str = "ffmpeg") -> NormalizeStatus:
    """Normalize audio loudness before upload so quiet-but-audible regions are
    not misclassified as silence by the ASR's voice-activity detector.

    Two-step strategy (much faster than a full loudnorm pass):
      1. Measure integrated loudness + range with ebur128 (cheap scan).
      2. If already well-leveled, skip entirely (SKIPPED — no output written).
         Otherwise apply a single-pass dynaudnorm (dynamic compression) plus a
         highpass to strip sub-bass rumble.

    Never raises; FAILED lets the caller fall back to the un-normalized source.
    """
    integrated, lra = analyze_loudness(input_path, ffmpeg_path)
    if integrated is not None:
        if (WELL_LEVELED_MIN_LUFS <= integrated <= WELL_LEVELED_MAX_LUFS
                and (lra is None or lra <= WELL_LEVELED_MAX_LRA)):
            return NormalizeStatus.SKIPPED

    # Fall back to a cheap single-pass compressor even if measurement failed.
    try:
        subprocess.run(
            [
                ffmpeg_path, "-y", "-i", input_path,
                "-vn", "-af", NORMALIZE_FILTER,
                "-acodec", "libmp3lame", "-b:a", "48k",
                "-ar", "16000", "-ac", "1",
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            **_no_window_kwargs()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return NormalizeStatus.FAILED

    if not Path(output_path).exists() or Path(output_path).stat().st_size < 1024:
        return NormalizeStatus.FAILED
    if get_audio_duration(output_path, ffmpeg_path) <= 0:
        return NormalizeStatus.FAILED
    return NormalizeStatus.NORMALIZED


# =============================================================================
# Audio Chunking (for long files)
# =============================================================================

CHUNK_DURATION_SEC = 540  # 9 minutes
CHUNK_OVERLAP_SEC = 10
MAX_CHUNK_RETRIES = 3
MIN_SEGMENTS_PER_MIN = 2  # Sanity threshold: <2 segments/min likely means gaps

# Bumped when chunk-processing behavior changes (e.g. added normalization). A
# stale cache from an older version is discarded so the new behavior actually
# takes effect instead of reusing old chunk SRTs.
# v4: chunks re-encoded to 16 kHz mono 48 kbps (speech-optimized, ~6x smaller
# uploads); cloud pacing/retry behavior also changed but that doesn't affect
# cached SRTs.
CACHE_VERSION = 4

# Loudness targets used by the measure-then-decide normalizer. highpass strips
# sub-bass rumble that inflates the noise floor and confuses the ASR's
# voice-activity detector; dynaudnorm dynamically lifts quiet-but-audible
# regions (single-pass, ~5x faster than loudnorm). Files already within the
# well-leveled range are skipped entirely — no expensive re-encode.
LOUDNESS_TARGET_LUFS = -16
WELL_LEVELED_MIN_LUFS = -19
WELL_LEVELED_MAX_LUFS = -13
WELL_LEVELED_MAX_LRA = 12
NORMALIZE_FILTER = "highpass=f=80,dynaudnorm"


def _apply_config_tunables() -> None:
    """Override chunking defaults from config.json when present."""
    global CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC, MAX_CHUNK_RETRIES
    try:
        from config import Config
        cfg = Config.load()
        CHUNK_DURATION_SEC = int(cfg.chunk_duration_sec)
        CHUNK_OVERLAP_SEC = int(cfg.chunk_overlap_sec)
        MAX_CHUNK_RETRIES = int(cfg.max_chunk_retries)
    except Exception:
        pass  # fall back to compiled-in defaults


_apply_config_tunables()


def get_audio_duration(audio_path: str, ffmpeg_path: str = "ffmpeg") -> float:
    """Get audio duration in seconds using ffprobe."""
    ffprobe = os.path.join(
        os.path.dirname(ffmpeg_path),
        os.path.basename(ffmpeg_path).replace("ffmpeg", "ffprobe")
    )
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             audio_path],
            capture_output=True, text=True, check=True,
            **_no_window_kwargs()
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0


def _validate_chunk(chunk_path: Path, expected_duration: float,
                    ffmpeg_path: str = "ffmpeg", tolerance: float = 1.5) -> bool:
    """Verify a chunk file is a valid audio file of approximately the expected duration."""
    if not chunk_path.exists() or chunk_path.stat().st_size < 1024:
        return False
    actual = get_audio_duration(str(chunk_path), ffmpeg_path)
    if actual <= 0:
        return False
    return abs(actual - expected_duration) <= tolerance


def split_audio_ffmpeg(audio_path: str, output_dir: Path,
                       chunk_duration: int = CHUNK_DURATION_SEC,
                       overlap: int = CHUNK_OVERLAP_SEC,
                       ffmpeg_path: str = "ffmpeg") -> List[Tuple[Path, float]]:
    """Split audio into overlapping chunks using ffmpeg.

    Returns list of (chunk_path, offset_seconds) tuples.
    Existing chunk files are validated and re-extracted if corrupt or incomplete.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_audio_duration(audio_path, ffmpeg_path)
    if duration <= 0:
        return []

    chunks = []
    step = chunk_duration - overlap
    idx = 0

    start = 0.0
    while start < duration:
        actual_duration = min(chunk_duration, duration - start)
        chunk_path = output_dir / f"chunk_{idx:03d}.mp3"

        if not _validate_chunk(chunk_path, actual_duration, ffmpeg_path):
            if chunk_path.exists():
                try:
                    chunk_path.unlink()
                except OSError:
                    pass
            subprocess.run(
                [ffmpeg_path, "-y", "-i", audio_path,
                 "-ss", str(start), "-t", str(actual_duration),
                 "-acodec", "libmp3lame", "-b:a", "48k",
                 "-ar", "16000", "-ac", "1",
                 str(chunk_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                **_no_window_kwargs()
            )

        chunks.append((chunk_path, start))
        idx += 1
        start += step

    return chunks


def parse_srt(srt_path: Path) -> List[dict]:
    """Parse an SRT file into utterances with transcript, start_time, end_time."""
    entries = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            timestamp_line = lines[1]
            text = "\n".join(lines[2:])
            try:
                start_str, end_str = timestamp_line.split(" --> ")
                start_ms = _srt_to_ms(start_str.strip())
                end_ms = _srt_to_ms(end_str.strip())
                if text.strip():
                    entries.append({
                        "transcript": text.strip(),
                        "start_time": start_ms,
                        "end_time": end_ms,
                    })
            except (ValueError, IndexError):
                continue

    return entries


def _srt_to_ms(ts: str) -> int:
    """Parse SRT timestamp HH:MM:SS,mmm to milliseconds."""
    time_str, ms_str = ts.split(",")
    h, m, s = time_str.split(":")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms_str)


def merge_chunk_results(chunk_results: List[Tuple[float, List[dict]]],
                        overlap_ms: int = CHUNK_OVERLAP_SEC * 1000) -> List[dict]:
    """Merge utterances from multiple chunks, adjusting timestamps and deduplicating overlaps.

    Args:
        chunk_results: List of (offset_ms, utterances) tuples
        overlap_ms: Overlap duration in milliseconds

    For overlapping regions, we keep the earlier chunk's version up to the midpoint
    of the overlap, then switch to the next chunk's version.
    """
    if not chunk_results:
        return []

    merged = []

    for i, (offset_ms, utterances) in enumerate(chunk_results):
        # Adjust timestamps for this chunk
        adjusted_entries = [{
            "transcript": u.get("transcript", u.get("text", "")),
            "start_time": u["start_time"] + offset_ms,
            "end_time": u["end_time"] + offset_ms,
        } for u in utterances]

        if i == 0:
            # First chunk: add all entries
            merged.extend(adjusted_entries)
        else:
            # Subsequent chunks: handle overlap with previous chunk
            overlap_mid = offset_ms + overlap_ms // 2

            # Remove entries from previous result that are past the midpoint
            while merged and merged[-1]["start_time"] >= overlap_mid:
                merged.pop()

            # Add entries from this chunk that start at or after the midpoint
            for entry in adjusted_entries:
                if entry["start_time"] >= overlap_mid:
                    merged.append(entry)

    return merged


def _write_srt_atomic(srt_path: Path, content: str) -> None:
    """Write SRT content atomically — temp file then rename. Prevents partial files."""
    tmp_path = srt_path.with_suffix(srt_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(srt_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def validate_coverage(utterances: List[dict], expected_duration_ms: int,
                      max_gap_ms: int = 60000) -> Tuple[bool, List[str]]:
    """Check that utterances cover the expected duration without suspicious gaps.

    Returns (ok, warnings). Gaps between speech segments are advisory only —
    they almost always correspond to legitimate pauses, music, or quiet
    interludes rather than ASR failures, and forcing the whole job to fail on
    them throws away good work. *ok* is therefore True as long as any
    utterances were produced at all; *warnings* lists gaps for the log.
    """
    warnings: List[str] = []

    if not utterances:
        return False, ["No utterances produced"]

    sorted_u = sorted(utterances, key=lambda u: u.get("start_time", 0))

    # Detect large gaps between consecutive entries
    big_gaps = []
    for i in range(1, len(sorted_u)):
        gap = sorted_u[i].get("start_time", 0) - sorted_u[i - 1].get("end_time", 0)
        if gap > max_gap_ms:
            big_gaps.append(
                f"Gap of {gap / 1000:.0f}s at {_srt_timestamp(sorted_u[i-1].get('end_time', 0))}"
            )

    if big_gaps:
        warnings.extend(big_gaps[:5])
        if len(big_gaps) > 5:
            warnings.append(f"... and {len(big_gaps) - 5} more gaps")

    # Check that the last entry reaches near the end of the audio
    last_end = max(u.get("end_time", 0) for u in sorted_u)
    if expected_duration_ms > 0 and last_end < expected_duration_ms * 0.7:
        warnings.append(
            f"Subtitles end at {_srt_timestamp(last_end)} but audio is "
            f"{_srt_timestamp(expected_duration_ms)} long"
        )

    # Gaps and end-of-audio shortfalls are advisory: they often correspond to
    # legitimately silent regions (music, pauses, quiet interludes) rather than
    # ASR failures. Only a total lack of utterances is fatal.
    return True, warnings


def _srt_timestamp(ms: int) -> str:
    """Format milliseconds as a human-readable HH:MM:SS timestamp."""
    seconds = ms / 1000
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ChunkedTranscriber:
    """Split long audio, transcribe chunks, merge results with resume support."""

    def __init__(self, audio_path: str, output_srt: Path,
                 engine: str = "auto",
                 progress_callback: Optional[Callable[[str], None]] = None,
                 ffmpeg_path: str = "ffmpeg",
                 model_id: str = "8",
                 parallel: bool = False,
                 source_name: Optional[str] = None):
        self.audio_path = audio_path
        self.output_srt = output_srt
        self.engine = engine
        self.progress_callback = progress_callback
        self.ffmpeg_path = ffmpeg_path
        self.model_id = model_id
        # Name of the ORIGINAL media file (audio_path may be a cached
        # extraction whose name says nothing about the source).
        self.source_name = source_name or Path(audio_path).name
        self.chunk_dir = Path(str(output_srt) + ".chunks")
        # Parallel workers only make sense for the multi-engine "auto" chain;
        # a single pinned engine gains nothing from a pool.
        self._parallel = parallel and engine == "auto"
        self._log_lock = threading.Lock()

    def _log(self, message: str) -> None:
        # Called from engine worker threads; the GUI callback routes into a
        # Qt signal (thread-safe emit), but serialize anyway so multi-line
        # engine logs never interleave mid-line.
        if self.progress_callback:
            with self._log_lock:
                self.progress_callback(message)

    def _create_asr(self, audio_path: str):
        """Create the appropriate ASR instance for the selected engine."""
        if self.engine == "bcut":
            return BcutASR(audio_path, self.progress_callback, self.model_id)
        elif self.engine == "jianying":
            duration_ms = get_audio_duration(audio_path, self.ffmpeg_path) * 1000
            return JianyingASR(audio_path, self.progress_callback,
                               start_time=0, end_time=int(duration_ms))
        elif self.engine == "local":
            return SherpaLocalASR(audio_path, self.progress_callback, self.ffmpeg_path)
        else:  # "auto"
            duration_ms = get_audio_duration(audio_path, self.ffmpeg_path) * 1000
            return AutoASR(audio_path, self.progress_callback, self.model_id,
                           start_time=0, end_time=int(duration_ms))

    def _create_pinned_asr(self, engine_name: str, audio_path: str):
        """Create a single-engine ASR instance by display name (worker path)."""
        if engine_name == "Bcut":
            return BcutASR(audio_path, self.progress_callback, self.model_id)
        if engine_name == "JianYing":
            duration_ms = get_audio_duration(audio_path, self.ffmpeg_path) * 1000
            return JianyingASR(audio_path, self.progress_callback,
                               start_time=0, end_time=int(duration_ms))
        raise ValueError(f"Unknown engine: {engine_name}")

    def _transcribe_chunk_pinned(self, i: int, total: int, chunk_path: Path,
                                 engine_name: str) -> List[dict]:
        """Transcribe one chunk with a pinned engine (parallel worker path).

        No per-chunk fallback chain — the pool itself is the fallback
        dimension: another worker picks the chunk up if this engine dies.
        Chunk-level retries still apply and failures feed the breaker.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_CHUNK_RETRIES + 1):
            if not engine_available(engine_name):
                raise RuntimeError(f"{engine_name} circuit breaker open")
            try:
                asr = self._create_pinned_asr(engine_name, str(chunk_path))
                utterances = _normalize_utterances(asr.transcribe())
                record_engine_success(engine_name)
                if not utterances:
                    self._log(
                        f"Chunk {i + 1}/{total} [{engine_name}]: no speech detected "
                        f"(likely silent segment)")
                else:
                    self._log(
                        f"Chunk {i + 1}/{total} [{engine_name}]: done "
                        f"({len(utterances)} segments)"
                        + (f" on attempt {attempt}" if attempt > 1 else ""))
                return utterances
            except Exception as e:
                last_error = e
                record_engine_failure(engine_name, e)
                if attempt < MAX_CHUNK_RETRIES:
                    delay = 5 * (2 ** (attempt - 1))
                    self._log(
                        f"Chunk {i + 1}/{total} [{engine_name}]: attempt "
                        f"{attempt}/{MAX_CHUNK_RETRIES} failed — {e}; retrying in {delay}s")
                    time.sleep(delay)
                else:
                    self._log(
                        f"Chunk {i + 1}/{total} [{engine_name}]: FAILED after "
                        f"{MAX_CHUNK_RETRIES} attempts — {e}")

        raise RuntimeError(
            f"Chunk {i + 1}/{total} failed on {engine_name}: {last_error}") from last_error

    def _run_parallel(self, pending: List[Tuple[int, Path, float]],
                      total: int) -> Optional[List[int]]:
        """Transcribe pending chunks on engine-pinned workers.

        All chunks sit in one shared queue; each worker takes the next chunk
        the moment it is free, so fast engines naturally process more chunks
        and no worker idles waiting for a slow one. Each engine still handles
        its own chunks strictly serially (unchanged per-server pacing).

        Returns the list of failed chunk numbers, or None if the pool could
        not be used (fewer than two engines available).
        """
        # KuaiShou is disabled: its endpoint returns "effect disabled
        # server-side" (code 501) and never recovers within a run, so it
        # only wastes upload bandwidth and retry time. The class stays for
        # the day the endpoint works again.
        worker_engines = [n for n in ("Bcut", "JianYing")
                          if engine_available(n)]
        if len(worker_engines) < 2:
            return None

        self._log(f"Parallel mode: {len(worker_engines)} engine workers "
                  f"({', '.join(worker_engines)})")
        q: "queue.Queue[Tuple[int, Path, float]]" = queue.Queue()
        for item in pending:
            q.put(item)

        failed: List[int] = []
        failed_lock = threading.Lock()

        def run_worker(name: str) -> None:
            chunks_done = 0
            while True:
                try:
                    i, chunk_path, offset_sec = q.get_nowait()
                except queue.Empty:
                    return
                if not engine_available(name):
                    # Breaker tripped mid-run — hand the chunk back and exit;
                    # surviving workers (or the serial fallback) take over.
                    q.put((i, chunk_path, offset_sec))
                    self._log(f"{name} worker stopping (circuit breaker open)")
                    return
                if chunks_done:
                    time.sleep(random.uniform(*INTER_CHUNK_DELAY_RANGE))
                chunks_done += 1
                self._log(f"Chunk {i + 1}/{total} [{name}]: transcribing "
                          f"(offset {offset_sec:.0f}s)...")
                try:
                    utterances = self._transcribe_chunk_pinned(i, total, chunk_path, name)
                    _write_srt_atomic(self.chunk_dir / f"chunk_{i:03d}.srt",
                                      utterances_to_srt(utterances))
                except Exception as e:
                    with failed_lock:
                        failed.append(i + 1)
                    try:
                        (self.chunk_dir / f"chunk_{i:03d}.failed").write_text(str(e))
                    except OSError:
                        pass

        threads = [threading.Thread(target=run_worker, args=(name,),
                                    name=f"asr-{name}", daemon=True)
                   for name in worker_engines]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return failed

    def _transcribe_single_chunk(self, i: int, total: int,
                                 chunk_path: Path, offset_sec: float) -> List[dict]:
        """Transcribe one chunk with retry. Raises on terminal failure."""
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_CHUNK_RETRIES + 1):
            try:
                asr = self._create_asr(str(chunk_path))
                utterances = asr.transcribe()
                utterances = _normalize_utterances(utterances)

                if not utterances:
                    # ASR legitimately returns nothing for silent audio. Treat
                    # this as a successful 0-segment result rather than an error
                    # so a single silent stretch can't abort the whole file.
                    self._log(
                        f"Chunk {i + 1}/{total}: no speech detected "
                        f"(likely silent segment)"
                    )
                    return utterances

                self._log(
                    f"Chunk {i + 1}/{total}: done ({len(utterances)} segments)"
                    + (f" on attempt {attempt}" if attempt > 1 else "")
                )
                return utterances

            except Exception as e:
                last_error = e
                if attempt < MAX_CHUNK_RETRIES:
                    delay = 5 * (2 ** (attempt - 1))
                    self._log(
                        f"Chunk {i + 1}/{total}: attempt {attempt}/{MAX_CHUNK_RETRIES} "
                        f"failed — {e}; retrying in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    self._log(f"Chunk {i + 1}/{total}: FAILED after {MAX_CHUNK_RETRIES} attempts — {e}")

        raise RuntimeError(f"Chunk {i + 1}/{total} failed: {last_error}") from last_error

    def transcribe(self) -> tuple[bool, str, int]:
        """Full workflow: split, transcribe chunks, merge. Supports resume.

        ALL chunks must succeed for the result to be marked successful.
        """
        self._log("Audio duration: checking...")
        duration = get_audio_duration(self.audio_path, self.ffmpeg_path)

        # Short file — no chunking needed
        if duration <= CHUNK_DURATION_SEC:
            self._log(f"Audio is {duration:.0f}s — no chunking needed")

            utterances: List[dict] = []
            last_error: Optional[Exception] = None
            for attempt in range(1, MAX_CHUNK_RETRIES + 1):
                try:
                    asr = self._create_asr(self.audio_path)
                    utterances = asr.transcribe()
                    utterances = _normalize_utterances(utterances)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < MAX_CHUNK_RETRIES:
                        delay = 5 * (2 ** (attempt - 1))
                        self._log(f"Attempt {attempt}/{MAX_CHUNK_RETRIES} failed — {e}; retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        return False, f"Transcription failed after {MAX_CHUNK_RETRIES} attempts: {e}", 0

            if not utterances:
                self._log("No speech detected (audio may be silent)")

            srt_content = utterances_to_srt(utterances)
            _write_srt_atomic(self.output_srt, srt_content)
            return True, f"Created {self.output_srt.name} ({len(utterances)} segments)", len(utterances)

        self._log(f"Audio is {duration / 60:.1f} min — splitting into chunks")

        # Check for existing chunk directory (resume)
        manifest_path = self.chunk_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("source_file") != self.source_name:
                    self._log("Source file changed, restarting from scratch")
                    shutil.rmtree(self.chunk_dir)
                elif manifest.get("cache_version") != CACHE_VERSION:
                    self._log("Processing changed since last run, restarting from scratch")
                    shutil.rmtree(self.chunk_dir)
                else:
                    self._log("Resuming previous transcription")
            except (json.JSONDecodeError, KeyError):
                shutil.rmtree(self.chunk_dir)

        # Split audio
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_audio_ffmpeg(self.audio_path, self.chunk_dir,
                                    CHUNK_DURATION_SEC, CHUNK_OVERLAP_SEC,
                                    self.ffmpeg_path)

        if not chunks:
            return False, "Failed to split audio", 0

        self._log(f"Split into {len(chunks)} chunks")

        # Save manifest (preserving keys like source size/mtime written by
        # transcribe_file's audio-cache validation)
        manifest_data: dict = {}
        try:
            manifest_data = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
        manifest_data.update({
            "source_file": self.source_name,
            "total_chunks": len(chunks),
            "chunk_duration": CHUNK_DURATION_SEC,
            "overlap": CHUNK_OVERLAP_SEC,
            "engine": self.engine,
            "model_id": self.model_id,
            "cache_version": CACHE_VERSION,
        })
        manifest_path.write_text(json.dumps(manifest_data, indent=2))

        # Chunks that still need transcription (no cached SRT yet)
        pending: List[Tuple[int, Path, float]] = []
        for i, (chunk_path, offset_sec) in enumerate(chunks):
            if (self.chunk_dir / f"chunk_{i:03d}.srt").exists():
                continue
            chunk_failed = self.chunk_dir / f"chunk_{i:03d}.failed"
            if chunk_failed.exists():
                chunk_failed.unlink()
            pending.append((i, chunk_path, offset_sec))

        failed_chunks: List[int] = []

        if pending:
            self._log(f"{len(pending)} chunk(s) to transcribe "
                      f"({len(chunks) - len(pending)} cached)")

            # Parallel: engine-pinned workers sharing one queue. A failed or
            # breaker-blocked worker's chunks fall through to the serial pass.
            if self._parallel and len(pending) > 1:
                par_failed = self._run_parallel(pending, len(chunks))
                if par_failed is not None:
                    failed_chunks = par_failed
                    # Recompute from disk: workers wrote SRTs out of order,
                    # so membership in `pending` no longer means "not done".
                    pending = [(i, p, o) for (i, p, o) in pending
                               if not (self.chunk_dir / f"chunk_{i:03d}.srt").exists()]

            # Serial pass: normal path when parallel is off, single-engine
            # runs, or cleanup of chunks the parallel workers could not
            # finish (full AutoASR fallback chain per chunk).
            if pending:
                for idx, (i, chunk_path, offset_sec) in enumerate(pending):
                    if idx:
                        time.sleep(random.uniform(*INTER_CHUNK_DELAY_RANGE))
                    self._log(f"Chunk {i + 1}/{len(chunks)}: transcribing "
                              f"(offset {offset_sec:.0f}s)...")
                    try:
                        utterances = self._transcribe_single_chunk(
                            i, len(chunks), chunk_path, offset_sec)
                        _write_srt_atomic(self.chunk_dir / f"chunk_{i:03d}.srt",
                                          utterances_to_srt(utterances))
                    except Exception as e:
                        try:
                            (self.chunk_dir / f"chunk_{i:03d}.failed").write_text(str(e))
                        except OSError:
                            pass
                        failed_chunks.append(i + 1)

        # Collect results from cached chunk SRTs. Reading back from disk (in
        # chunk order) makes the merge independent of completion order, which
        # the parallel workers do not preserve.
        chunk_results: List[Tuple[int, List[dict]]] = []
        missing_chunks: List[int] = []
        for i, (chunk_path, offset_sec) in enumerate(chunks):
            chunk_srt = self.chunk_dir / f"chunk_{i:03d}.srt"
            if chunk_srt.exists():
                chunk_results.append((int(offset_sec * 1000), parse_srt(chunk_srt)))
            else:
                missing_chunks.append(i + 1)

        # STRICT: any chunk failure means the whole result is incomplete
        if missing_chunks:
            msg = (
                f"{len(missing_chunks)}/{len(chunks)} chunk(s) failed (indices {missing_chunks}). "
                f"Output NOT written. Re-run to resume from cached chunks."
            )
            return False, msg, 0

        if not chunk_results:
            return False, "All chunks failed", 0

        # Merge results
        self._log("Merging chunks...")
        merged = merge_chunk_results(chunk_results)

        if not merged:
            return False, "Merge produced no results", 0

        # Validate coverage. Gaps are advisory only — a long pause, music bed,
        # or quiet interlude between speech segments is a normal subtitle, not
        # an ASR failure, and must not discard the rest of the transcription.
        ok, warnings = validate_coverage(merged, int(duration * 1000))
        if not ok:
            # Only reachable if merging produced nothing, which is already
            # guarded above. Keep as a defensive no-op rather than failing.
            self._log("Coverage validation: no utterances (skipped)")
        for w in warnings:
            self._log(f"Note: {w}")

        srt_content = utterances_to_srt(merged)
        _write_srt_atomic(self.output_srt, srt_content)

        msg = (
            f"Created {self.output_srt.name} ({len(merged)} segments). "
            f"Chunks saved in {self.chunk_dir.name}/ (safe to delete)"
        )
        return True, msg, len(merged)


# =============================================================================
# Main Processing Function
# =============================================================================

def transcribe_file(
    file_path: Path,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    ffmpeg_path: str = "ffmpeg",
    model_id: str = "8",
    engine: str = "auto",
    normalize_audio: bool = True,
    parallel: bool = False,
) -> tuple[bool, str, int]:
    """
    Transcribe a single audio/video file.

    Args:
        file_path: Path to input file
        output_path: Path for output SRT (default: same as input with .srt extension)
        overwrite: Overwrite existing SRT file
        progress_callback: Optional callback for progress updates
        ffmpeg_path: Path to ffmpeg executable
        model_id: Bcut model ID
        engine: ASR engine — "bcut", "jianying", "local", or "auto" (default)
        normalize_audio: Apply loudness normalization before upload (default True).
            Helps the ASR hear quiet-but-audible regions instead of misclassifying
            them as silence. Already well-leveled files are skipped automatically.
        parallel: Run cloud engines concurrently on chunks (engine "auto" only).

    Returns:
        Tuple of (success: bool, message: str, segment_count: int)
    """
    if output_path is None:
        output_path = file_path.with_suffix(".srt")

    if output_path.exists() and not overwrite:
        return False, "SRT file already exists", 0

    # For chunked (long) files the extracted + normalized working audio is
    # cached inside the chunk dir so a resumed run skips re-extraction. The
    # cache is validated against the source file's name/size/mtime plus the
    # processing version; any mismatch discards the whole chunk dir.
    SOURCE_AUDIO_NAME = "source_audio.mp3"
    chunk_dir = Path(str(output_path) + ".chunks")
    manifest_path = chunk_dir / "manifest.json"
    cached_audio = chunk_dir / SOURCE_AUDIO_NAME
    use_audio_cache = get_audio_duration(str(file_path), ffmpeg_path) > CHUNK_DURATION_SEC

    working_path: Path
    temp_extracted: Optional[Path] = None
    temp_normalized: Optional[Path] = None

    if use_audio_cache:
        src_stat = file_path.stat()
        reuse = False
        if manifest_path.exists() and cached_audio.is_file():
            try:
                m = json.loads(manifest_path.read_text())
                reuse = (
                    m.get("source_file") == file_path.name
                    and m.get("source_size") == src_stat.st_size
                    and m.get("source_mtime") == int(src_stat.st_mtime)
                    and m.get("cache_version") == CACHE_VERSION
                )
            except (json.JSONDecodeError, OSError, AttributeError):
                reuse = False

        if reuse:
            if progress_callback:
                progress_callback("Reusing cached audio (extraction skipped)")
            working_path = cached_audio
        else:
            # Invalidate the whole cache (source changed or version bump)
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)

            working_path, temp_extracted, temp_normalized = _extract_working_audio(
                file_path, cached_audio, normalize_audio, progress_callback, ffmpeg_path)
            if working_path is None:
                return False, "Failed to extract audio", 0
            chunk_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({
                "source_file": file_path.name,
                "source_size": src_stat.st_size,
                "source_mtime": int(src_stat.st_mtime),
                "cache_version": CACHE_VERSION,
            }, indent=2))
    else:
        # Short file — no chunk dir; plain temp-file flow.
        target = file_path.with_suffix(".mp3")
        working_path, temp_extracted, temp_normalized = _extract_working_audio(
            file_path, target, normalize_audio, progress_callback, ffmpeg_path)
        if working_path is None:
            return False, "Failed to extract audio", 0

    try:
        transcriber = ChunkedTranscriber(
            audio_path=str(working_path),
            output_srt=output_path,
            engine=engine,
            progress_callback=progress_callback,
            ffmpeg_path=ffmpeg_path,
            model_id=model_id,
            parallel=parallel,
            source_name=file_path.name,
        )
        return transcriber.transcribe()

    except Exception as e:
        return False, str(e), 0

    finally:
        # Clean up temp files only; the cached source audio stays in the
        # chunk dir so failed runs can resume without re-extracting.
        for temp in (temp_normalized, temp_extracted):
            if temp is not None and temp.exists():
                try:
                    temp.unlink()
                except OSError:
                    pass


def _extract_working_audio(file_path: Path, final_target: Path,
                           normalize_audio: bool,
                           progress_callback: Optional[Callable[[str], None]],
                           ffmpeg_path: str) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Extract (and optionally normalize) audio into *final_target*.

    Returns (final_path, temp_extracted, temp_normalized) — the temps must be
    deleted by the caller once the ASR run is over.
    """
    temp_extracted: Optional[Path] = None
    if file_path.suffix.lower() != ".mp3":
        temp_extracted = file_path.with_suffix(".mp3")
        if progress_callback:
            progress_callback(f"Extracting audio from {file_path.name}...")
        if not extract_audio_to_mp3(str(file_path), str(temp_extracted), ffmpeg_path):
            return None, None, None
        source = temp_extracted
    else:
        source = file_path

    temp_normalized: Optional[Path] = None
    result = source
    if normalize_audio:
        if progress_callback:
            progress_callback(f"Measuring audio loudness for {file_path.name}...")
        norm_tmp = source.with_suffix(source.suffix + ".norm.mp3")
        status = normalize_audio_loudness(str(source), str(norm_tmp), ffmpeg_path)
        if status == NormalizeStatus.NORMALIZED:
            temp_normalized = norm_tmp
            result = norm_tmp
            if progress_callback:
                progress_callback(f"Normalized audio for {file_path.name}")
        elif status == NormalizeStatus.SKIPPED:
            if progress_callback:
                progress_callback(
                    f"Audio for {file_path.name} already well-leveled — skipping normalization")
        elif progress_callback:
            progress_callback("Loudness normalization failed — using original audio")

    if result != final_target:
        final_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(result), str(final_target))
    return final_target, temp_extracted, temp_normalized


SUPPORTED_EXTENSIONS = frozenset({
    # Video
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".flv", ".f4v",
    ".wmv", ".asf", ".mpg", ".mpeg", ".vob", ".ts", ".m2ts", ".mts",
    ".3gp", ".3g2", ".ogv", ".divx", ".rm", ".rmvb",
    # Audio
    ".mp3", ".mp2", ".mpga", ".wav", ".flac", ".m4a", ".aac", ".ac3",
    ".eac3", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".amr", ".mka",
    ".au", ".dts", ".caf", ".ra",
})


def get_audio_files(directory: str = ".") -> List[Path]:
    """Get all audio/video files in directory."""
    files = []
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)
