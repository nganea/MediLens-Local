from pathlib import Path
import os
import re
import shutil
import socket
import subprocess
from urllib.parse import urlparse

from .config import DEFAULT_MODEL_URL, DEFAULT_VISION_OCR_URL, MINICPM_V_MODEL_REF, TINY_AYA_MODEL_REF


def normalize_local_url(url: str, default_url: str) -> str:
    cleaned_url = (url or "").strip() or default_url
    if "://" not in cleaned_url:
        cleaned_url = f"http://{cleaned_url}"
    return cleaned_url


def parse_host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def is_port_reachable(url: str, timeout: float = 1.0) -> bool:
    try:
        host, port = parse_host_port(url)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _llama_cache_dirs() -> list:
    dirs = []
    env_cache = os.getenv("LLAMA_CACHE")
    if env_cache:
        dirs.append(Path(env_cache))
    if os.name == "nt":
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            dirs.append(Path(local_appdata) / "llama.cpp")
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        dirs.append(Path(xdg_cache) / "llama.cpp")
    home = Path.home()
    dirs.append(home / "Library" / "Caches" / "llama.cpp")
    dirs.append(home / ".cache" / "llama.cpp")
    return [d for d in dirs if d.is_dir()]


def model_is_downloaded(model_ref: str) -> bool:
    """True if the GGUF for model_ref is already cached locally (no download needed)."""
    ref = model_ref.split(":", 1)[0]
    user, _, repo = ref.partition("/")
    if not repo:
        user, repo = "", ref
    repo_core = re.sub(r"(?i)gguf", "", repo)
    candidates = {_normalize_token(repo_core)}
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", repo_core) if t]
    if tokens:
        candidates.add(_normalize_token("".join(tokens[:2])))
    candidates = {c for c in candidates if len(c) >= 4}

    for cache_dir in _llama_cache_dirs():
        try:
            for gguf in cache_dir.rglob("*.gguf"):
                name_norm = _normalize_token(gguf.name)
                if any(c in name_norm for c in candidates):
                    return True
        except OSError:
            continue

    hub_dirs = []
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        hub_dirs.append(Path(hf_home) / "hub")
    hub_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    repo_dir_name = f"models--{user}--{repo}" if user else f"models--{repo}"
    for hub in hub_dirs:
        repo_dir = hub / repo_dir_name
        if repo_dir.is_dir():
            try:
                if any(repo_dir.rglob("*.gguf")):
                    return True
            except OSError:
                continue
    return False


def start_llama_server(model_ref: str, url: str) -> str:
    if is_port_reachable(url):
        host, port = parse_host_port(url)
        return f"Already running on {host}:{port}."

    llama_server = shutil.which("llama-server")
    if not llama_server:
        return "Could not find llama-server on PATH. Open a new terminal after installing llama.cpp, or start llama-server manually."

    if not model_is_downloaded(model_ref):
        return (
            f"Model not downloaded, so it was not started. "
            f"Download it once in a terminal, then use this button again: "
            f"llama-server -hf {model_ref}"
        )

    _, port = parse_host_port(url)
    command = [llama_server, "-hf", model_ref, "--port", str(port)]
    kwargs = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    try:
        subprocess.Popen(command, **kwargs)
    except OSError as error:
        return f"Could not start llama-server: {error}"

    return f"Starting in the background on port {port}. Wait a minute for the model to load, then try again."


def start_local_model_servers(vision_ocr_url: str, model_url: str) -> str:
    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    model_url = normalize_local_url(model_url, DEFAULT_MODEL_URL)
    minicpm_status = start_llama_server(MINICPM_V_MODEL_REF, vision_ocr_url)
    tiny_aya_status = start_llama_server(TINY_AYA_MODEL_REF, model_url)
    return f"MiniCPM-V 4.6: {minicpm_status}\nTiny Aya: {tiny_aya_status}"


def check_local_model_servers(vision_ocr_url: str, model_url: str) -> str:
    vision_ocr_url = normalize_local_url(vision_ocr_url, DEFAULT_VISION_OCR_URL)
    model_url = normalize_local_url(model_url, DEFAULT_MODEL_URL)
    minicpm_status = "reachable" if is_port_reachable(vision_ocr_url) else "not reachable"
    tiny_aya_status = "reachable" if is_port_reachable(model_url) else "not reachable"
    return f"MiniCPM-V 4.6: {minicpm_status}\nTiny Aya: {tiny_aya_status}"

