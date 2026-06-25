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
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import uuid
import zlib
from pathlib import Path
from typing import Callable, Optional, List, Tuple


# =============================================================================
# HTTP Utilities (with automatic retry on transient errors)
# =============================================================================

HTTP_RETRIES = 4
HTTP_RETRY_BASE_DELAY = 2.0


def _is_transient_error(e: Exception) -> bool:
    """Return True if the exception is likely a transient network/server error."""
    if isinstance(e, urllib.error.URLError):
        return True
    if isinstance(e, TimeoutError):
        return True
    if isinstance(e, ConnectionError):
        return True
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500 or e.code == 429
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
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code}: {error_body}") from e
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
            raise RuntimeError(f"Upload failed: HTTP {e.code}") from e

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
            raise RuntimeError(f"HTTP {e.code}") from e

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
            error_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code}: {error_body}") from e
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
            raise RuntimeError(
                f"Upload failed: HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}"
            ) from e

    return _retry(_do)


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
        """Poll for transcription result."""
        self._log("Transcribing (may take 1-5 minutes)...")

        consecutive_errors = 0
        for i in range(600):  # Max ~10 minutes
            try:
                resp = http_get(
                    self.API_QUERY_RESULT,
                    params={"model_id": 7, "task_id": self.task_id},
                    headers=self.HEADERS
                )
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 10:
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
            elif state in (3, 5):  # Failed or cancelled
                raise RuntimeError(f"ASR task failed with state: {state}")

            # Animated loading
            dots = i % 4
            if self.progress_callback:
                self._log(f"Transcribing{'.' * dots}{' ' * (3 - dots)}")
            time.sleep(1)

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


def _get_tid() -> str:
    i = str(datetime.datetime.now().year)[3]
    fr = 390 + int(i)
    ed = "3278516897751" if int(i) % 2 != 0 else f"{uuid.getnode():013d}"
    return f"{fr}{ed}"


class JianyingASR:
    """JianYing (CapCut) ASR API - Free Chinese/English transcription (fallback engine).

    Uses ByteDance's JianYing cloud ASR service with AWS S3-style upload.
    Depends on an external sign service for request authentication.
    """

    def __init__(self, audio_path: str, progress_callback: Optional[Callable[[str], None]] = None,
                 start_time: int = 0, end_time: int = 6000):
        self.audio_path = audio_path
        self.progress_callback = progress_callback
        self.start_time = start_time
        self.end_time = end_time
        self.tdid = _get_tid()

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

    def _generate_sign(self, url_path: str) -> Tuple[str, str]:
        """Get request signature from external sign service."""
        current_time = str(int(time.time()))
        data = {
            "url": url_path,
            "current_time": current_time,
            "pf": "4",
            "appvr": "6.6.0",
            "tdid": self.tdid,
        }
        headers = {
            "User-Agent": "SubtitleTools/1.0",
            "tdid": self.tdid,
            "t": current_time,
        }
        try:
            resp = http_post_raw(
                "https://asrtools-update.bkfeng.top/sign",
                json_data=data, headers=headers, timeout=10
            )
            if isinstance(resp, dict):
                sign = resp.get("sign")
            else:
                resp_data = json.loads(resp)
                sign = resp_data.get("sign")
            if not sign:
                raise RuntimeError("No 'sign' in response from sign service")
            return sign.lower(), current_time
        except Exception as e:
            raise RuntimeError(f"Jianying sign service unavailable: {e}") from e

    def _build_headers(self, device_time: str, sign: str) -> dict:
        return {
            "User-Agent": "Cronet/TTNetVersion:d4572e53 2024-06-12 QuicVersion:4bf243e0 2023-04-17",
            "appvr": "6.6.0",
            "device-time": str(device_time),
            "pf": "4",
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

    def _upload_commit(self) -> str:
        url = (
            f"https://{self.upload_hosts}/{self.store_uri}"
            f"?uploadID={self.upload_id}&partNumber=1&x-amz-security-token={self.session_token}"
        )
        file_binary = self._load_audio()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Authorization": self.auth,
        }
        req = urllib.request.Request(url, data=file_binary, headers=headers, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                pass
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Upload commit failed: HTTP {e.code}") from e
        return self.store_uri

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
        self._upload_commit()
        self._log("Jianying: Upload complete!")

        query_id = self.submit()

        # Jianying processes quickly, poll every 2s for up to ~10 min
        self._log("Jianying: Transcribing...")
        consecutive_errors = 0
        for i in range(300):
            try:
                resp_data = self.query(query_id)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    raise RuntimeError(
                        f"Jianying polling failed after {consecutive_errors} errors: {e}"
                    ) from e
                self._log(f"Jianying: polling error (will retry): {e}")
                time.sleep(min(2 ** consecutive_errors, 30))
                continue

            utterances = resp_data.get("data", {}).get("utterances")
            if utterances is not None:
                self._log("Jianying: Transcription complete!")
                return utterances
            dots = i % 4
            self._log(f"Jianying: Waiting{'.' * dots}{' ' * (3 - dots)}")
            time.sleep(2)

        raise RuntimeError("Jianying transcription timeout")


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
    """Auto-fallback ASR: tries Bcut first, falls back to Jianying on failure."""

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

    def transcribe(self) -> List[dict]:
        bcut_error = None
        try:
            self._log("Trying Bcut ASR...")
            asr = BcutASR(self.audio_path, self.progress_callback, self.model_id)
            utterances = asr.transcribe()
            self._log("Bcut ASR succeeded")
            return _normalize_utterances(utterances)
        except Exception as e:
            bcut_error = e
            self._log(f"Bcut failed: {e}")
            self._log("Falling back to Jianying ASR...")

        try:
            asr = JianyingASR(self.audio_path, self.progress_callback,
                              self.start_time, self.end_time)
            utterances = asr.transcribe()
            self._log("Jianying ASR succeeded")
            return _normalize_utterances(utterances)
        except Exception as jianying_error:
            raise RuntimeError(
                f"All ASR engines failed. Bcut: {bcut_error}. Jianying: {jianying_error}"
            ) from jianying_error


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

def check_ffmpeg(ffmpeg_path: str = "ffmpeg") -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
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
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                "-ar", "44100", "-ac", "1",
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    # Verify the output is a valid, non-empty audio file
    if not Path(output_path).exists() or Path(output_path).stat().st_size < 1024:
        return False
    if get_audio_duration(output_path, ffmpeg_path) <= 0:
        return False
    return True


# =============================================================================
# Audio Chunking (for long files)
# =============================================================================

CHUNK_DURATION_SEC = 540  # 9 minutes
CHUNK_OVERLAP_SEC = 10
MAX_CHUNK_RETRIES = 3
MIN_SEGMENTS_PER_MIN = 2  # Sanity threshold: <2 segments/min likely means gaps


def get_audio_duration(audio_path: str, ffmpeg_path: str = "ffmpeg") -> float:
    """Get audio duration in seconds using ffprobe."""
    ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe")
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             audio_path],
            capture_output=True, text=True, check=True
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
                 "-acodec", "libmp3lame", "-q:a", "2",
                 "-ar", "44100", "-ac", "1",
                 str(chunk_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
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

    Returns (ok, warnings) where *ok* is True when the output looks complete and
    *warnings* is a list of human-readable descriptions of any problems found.
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

    return len(warnings) == 0, warnings


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
                 model_id: str = "8"):
        self.audio_path = audio_path
        self.output_srt = output_srt
        self.engine = engine
        self.progress_callback = progress_callback
        self.ffmpeg_path = ffmpeg_path
        self.model_id = model_id
        self.chunk_dir = Path(str(output_srt) + ".chunks")

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _create_asr(self, audio_path: str):
        """Create the appropriate ASR instance for the selected engine."""
        if self.engine == "bcut":
            return BcutASR(audio_path, self.progress_callback, self.model_id)
        elif self.engine == "jianying":
            duration_ms = get_audio_duration(audio_path, self.ffmpeg_path) * 1000
            return JianyingASR(audio_path, self.progress_callback,
                               start_time=0, end_time=int(duration_ms))
        else:  # "auto"
            duration_ms = get_audio_duration(audio_path, self.ffmpeg_path) * 1000
            return AutoASR(audio_path, self.progress_callback, self.model_id,
                           start_time=0, end_time=int(duration_ms))

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
                    raise RuntimeError("ASR returned no transcription results")

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
                    if not utterances:
                        raise RuntimeError("ASR returned no transcription results")
                    break
                except Exception as e:
                    last_error = e
                    if attempt < MAX_CHUNK_RETRIES:
                        delay = 5 * (2 ** (attempt - 1))
                        self._log(f"Attempt {attempt}/{MAX_CHUNK_RETRIES} failed — {e}; retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        return False, f"Transcription failed after {MAX_CHUNK_RETRIES} attempts: {e}", 0

            srt_content = utterances_to_srt(utterances)
            _write_srt_atomic(self.output_srt, srt_content)
            return True, f"Created {self.output_srt.name} ({len(utterances)} segments)", len(utterances)

        self._log(f"Audio is {duration / 60:.1f} min — splitting into chunks")

        # Check for existing chunk directory (resume)
        manifest_path = self.chunk_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("source_file") != Path(self.audio_path).name:
                    self._log("Source file changed, restarting from scratch")
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

        # Save manifest
        manifest_path.write_text(json.dumps({
            "source_file": Path(self.audio_path).name,
            "total_chunks": len(chunks),
            "chunk_duration": CHUNK_DURATION_SEC,
            "overlap": CHUNK_OVERLAP_SEC,
            "engine": self.engine,
            "model_id": self.model_id,
        }, indent=2))

        # Transcribe each chunk — ALL must succeed
        chunk_results: List[Tuple[int, List[dict]]] = []
        failed_chunks: List[int] = []

        for i, (chunk_path, offset_sec) in enumerate(chunks):
            chunk_srt = self.chunk_dir / f"chunk_{i:03d}.srt"
            chunk_failed = self.chunk_dir / f"chunk_{i:03d}.failed"

            # Load from cache if a valid SRT already exists
            if chunk_srt.exists():
                cached = parse_srt(chunk_srt)
                if cached:
                    self._log(f"Chunk {i + 1}/{len(chunks)}: already transcribed ({len(cached)} segments)")
                    chunk_results.append((int(offset_sec * 1000), cached))
                    continue
                else:
                    chunk_srt.unlink()

            if chunk_failed.exists():
                chunk_failed.unlink()

            self._log(f"Chunk {i + 1}/{len(chunks)}: transcribing (offset {offset_sec:.0f}s)...")

            try:
                utterances = self._transcribe_single_chunk(i, len(chunks), chunk_path, offset_sec)
                # Save chunk SRT immediately (for resume)
                srt_content = utterances_to_srt(utterances)
                _write_srt_atomic(chunk_srt, srt_content)
                chunk_results.append((int(offset_sec * 1000), utterances))
            except Exception as e:
                chunk_failed.write_text(str(e))
                failed_chunks.append(i + 1)

        # STRICT: any chunk failure means the whole result is incomplete
        if failed_chunks:
            msg = (
                f"{len(failed_chunks)}/{len(chunks)} chunk(s) failed (indices {failed_chunks}). "
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

        # Validate coverage before declaring success
        ok, warnings = validate_coverage(merged, int(duration * 1000))
        if not ok:
            detail = "; ".join(warnings)
            self._log(f"Coverage validation FAILED: {detail}")
            return False, f"Output incomplete — {detail}", 0
        elif warnings:
            for w in warnings:
                self._log(f"Warning: {w}")

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
    engine: str = "auto"
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
        engine: ASR engine — "bcut", "jianying", or "auto" (default)

    Returns:
        Tuple of (success: bool, message: str, segment_count: int)
    """
    if output_path is None:
        output_path = file_path.with_suffix(".srt")

    if output_path.exists() and not overwrite:
        return False, "SRT file already exists", 0

    # Convert to MP3 if needed
    mp3_path = file_path.with_suffix(".mp3")
    needs_conversion = file_path.suffix.lower() != ".mp3"

    if needs_conversion:
        if progress_callback:
            progress_callback(f"Extracting audio from {file_path.name}...")
        if not extract_audio_to_mp3(str(file_path), str(mp3_path), ffmpeg_path):
            return False, "Failed to extract audio", 0
    else:
        mp3_path = file_path

    try:
        transcriber = ChunkedTranscriber(
            audio_path=str(mp3_path),
            output_srt=output_path,
            engine=engine,
            progress_callback=progress_callback,
            ffmpeg_path=ffmpeg_path,
            model_id=model_id,
        )
        return transcriber.transcribe()

    except Exception as e:
        return False, str(e), 0

    finally:
        # Clean up temporary MP3 if it was converted
        if needs_conversion and mp3_path.exists():
            try:
                mp3_path.unlink()
            except OSError:
                pass


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
