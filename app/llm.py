"""Cliente LLM local (Ollama, API compatible con OpenAI)."""
import json
import urllib.request
from . import config

_cached_model_id: str | None = None


def _get(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{config.OLLAMA_URL}{endpoint}", timeout=10) as resp:
        return json.loads(resp.read())


def _post(endpoint: str, payload: dict, base: str = None) -> dict:
    req = urllib.request.Request(
        f"{base or config.OLLAMA_URL}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT) as resp:
        return json.loads(resp.read())


def get_status() -> dict:
    try:
        data = _get("/models")
        models = [m["id"] for m in data.get("data", [])]
        # el "activo" mostrado en la UI debe ser el que de verdad usa el chat
        # (config.OLLAMA_MODEL), no el primero que Ollama liste por azar.
        active = config.OLLAMA_MODEL if config.OLLAMA_MODEL in models else (models[0] if models else None)
        return {"ok": True, "models": models, "active": active}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


def _model_id() -> str:
    global _cached_model_id
    if config.OLLAMA_MODEL:
        return config.OLLAMA_MODEL
    if _cached_model_id:
        return _cached_model_id
    status = get_status()
    if not status["ok"] or not status["models"]:
        raise RuntimeError("El modelo local no esta disponible.")
    _cached_model_id = status["models"][0]
    return _cached_model_id


def chat(messages: list, temperature: float = 0.1, max_tokens: int = 450) -> str:
    # API nativa de Ollama (no la compatible con OpenAI): es la unica que
    # respeta keep_alive, asi el modelo no se descarga de memoria entre
    # preguntas y evitamos el recargo de 1-2 min por "cold start".
    #
    # temperature baja + seed fijo: es informacion financiera, no un chat
    # creativo — se prioriza que la MISMA pregunta con el MISMO contexto de
    # datos de siempre el mismo resultado (o muy cercano) en vez de variar
    # cada vez por el muestreo aleatorio del modelo. El Agente de
    # Consistencia (agents.py) todavia cachea la respuesta final por si el
    # backend no es perfectamente determinista con esto solo.
    payload = {
        "model": _model_id(),
        "messages": messages,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
    }
    result = _post("/api/chat", payload, base=config.OLLAMA_HOST)
    return result["message"]["content"].strip()
