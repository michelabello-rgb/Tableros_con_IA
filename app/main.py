import threading
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import config, llm, dataset, pbix_loader, agents, governance

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Copiloto Power BI")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    """Los estaticos (embed.js, app.js, style.css) cambian seguido durante
    desarrollo — sin esto, el navegador puede seguir usando una version
    vieja en cache tras un simple F5 y dar errores ya corregidos (como paso
    con el fix de MSAL initialize()). Es un proyecto local, no hay perdida
    real de rendimiento por no cachear."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.on_event("startup")
def _warm_up():
    """Precarga en memoria lo mas caro de la primera pregunta real: el
    modelo de Ollama (1-2 min de 'cold start') y las tablas .parquet mas
    grandes (leer un archivo de decenas de MB desde disco toma segundos).
    Todo en un hilo aparte para no bloquear el arranque del servidor."""
    def _ping():
        try:
            llm.chat([{"role": "user", "content": "hola"}], max_tokens=1)
        except Exception:
            pass
        if dataset.is_ready():
            for t in dataset.get_overview()[:3]:  # las 3 tablas mas grandes
                try:
                    dataset.load_table(t["table"])
                except Exception:
                    pass
    threading.Thread(target=_ping, daemon=True).start()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "embed_url": config.POWERBI_EMBED_URL,
        "original_url": config.POWERBI_ORIGINAL_URL,
        "azure_embed_enabled": config.AZURE_EMBED_ENABLED,
        "azure_client_id": config.AZURE_CLIENT_ID,
        "azure_tenant_id": config.AZURE_TENANT_ID,
        "powerbi_group_id": config.POWERBI_GROUP_ID,
        "powerbi_report_id": config.POWERBI_REPORT_ID,
    })


@app.get("/api/status")
def api_status():
    return {
        "llm": llm.get_status(),
        "data": {"ok": dataset.is_ready(), **(dataset.get_source_info() if dataset.is_ready() else {})},
        "azure_embed_enabled": config.AZURE_EMBED_ENABLED,
    }


@app.get("/api/overview")
def api_overview():
    if not dataset.is_ready():
        return JSONResponse({"error": "Datos no extraidos todavia. Usa el boton Actualizar datos."}, status_code=503)
    return {"source": dataset.get_source_info(), "tables": dataset.get_overview()}


@app.post("/api/refresh")
def api_refresh():
    """Vuelve a leer el .pbix de tablero/ y refresca el cache — el 'tiempo real'
    de este modelo: sin Premium/PPU no existe query en vivo a la API de Power
    BI, asi que la forma simple de reflejar cambios es re-extraer el archivo
    (reemplazalo en tablero/ y presiona este boton)."""
    try:
        manifest = pbix_loader.extract()
        dataset.clear_cache()
        governance.clear_cache()  # por si reglas_negocio.json cambio desde el ultimo arranque
        agents.clear_answer_cache()  # limpieza de memoria; la clave ya cambia sola igual
        return {"ok": True, "source": manifest.get("source_file"), "tables": len(manifest.get("tables", {}))}
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/chat")
async def api_chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Peticion invalida (JSON mal formado)."}, status_code=400)
    pregunta = (body.get("message") or "").strip()
    if not pregunta:
        raise HTTPException(400, "Mensaje vacio")

    try:
        # El pipeline (llamadas a Ollama incluidas) es codigo sincrono/bloqueante;
        # correrlo en un thread aparte evita congelar el servidor entero (y con
        # el, /api/status y cualquier otra pestana) mientras el modelo genera.
        result = await run_in_threadpool(agents.run_pipeline, pregunta)
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/report")
async def api_report():
    prompt = (
        "Genera un reporte ejecutivo corto (maximo 10 lineas) para directivos, "
        "listo para copiar y pegar, con lo mas relevante del dataset descrito "
        "arriba: titulo, 3-4 cifras clave con su lectura, 1-2 hallazgos y una "
        "recomendacion. Usa bullets cortos, tono cercano."
    )
    try:
        result = await run_in_threadpool(agents.run_report_pipeline, prompt)
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
