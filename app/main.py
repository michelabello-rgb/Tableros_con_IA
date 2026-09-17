import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import config, llm, dataset, pbix_loader, agents, governance, db

BASE_DIR = Path(__file__).resolve().parent

# docs_url/redoc_url/openapi_url en None: esto no es una API publica, es el
# backend de una sola pagina — el Swagger/OpenAPI por defecto de FastAPI
# quedaba alcanzable SIN login y exponia la forma completa de la API
# (rutas, parametros) a cualquiera que la encontrara.
app = FastAPI(title="Copiloto Power BI", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Firma la cookie de sesion del login (ver config.SESSION_SECRET) — sin
# esto, request.session no existe y las rutas de login no tendrian donde
# guardar quien entro.
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET, same_site="lax")


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Cabeceras basicas de seguridad — mitigan clickjacking (alguien
    embebiendo esta app en un iframe oculto de otro sitio para engañar a
    quien tenga sesion iniciada) y que el navegador adivine mal el tipo de
    un archivo servido."""
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


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


# ── Login ─────────────────────────────────────────────────────────────────
# Gate de acceso a TODA la app (pagina y API), no solo al reporte embebido
# de Power BI — ese ya filtraba por permisos reales de Power BI Service,
# pero el chat hablaba con los datos ya extraidos sin pedirle nada a nadie.
# Contra COE.User_Admin / COE.Acceso_Log (ver sql/ y app/db.py).

def _current_user(request: Request) -> dict | None:
    return request.session.get("user")


def require_login_api(request: Request) -> dict:
    """Para rutas de API (fetch/JSON): 401 con JSON en vez de una redirección
    HTML, que el frontend no sabria interpretar."""
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado — inicia sesión de nuevo.")
    return user


# Limite de intentos de login fallidos por username — sin esto, nada impide
# probar contraseñas sin parar contra una cuenta real (fuerza bruta). En
# memoria a proposito (un solo proceso, se reinicia solo con el servidor);
# la bitacora real y persistente de cada intento sigue siendo
# COE.Acceso_Log via db.log_access, esto es solo el freno.
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_INTENTOS = 5
_LOGIN_VENTANA_SEG = 15 * 60  # 15 minutos


def _login_bloqueado(key: str) -> bool:
    ahora = time.time()
    vivos = [t for t in _LOGIN_ATTEMPTS.get(key, []) if ahora - t < _LOGIN_VENTANA_SEG]
    _LOGIN_ATTEMPTS[key] = vivos
    return len(vivos) >= _LOGIN_MAX_INTENTOS


def _registrar_intento_fallido(key: str):
    _LOGIN_ATTEMPTS.setdefault(key, []).append(time.time())


def _limpiar_intentos(key: str):
    _LOGIN_ATTEMPTS.pop(key, None)


def require_admin(request: Request) -> dict:
    """Para el panel de administracion de usuarios — no basta con estar
    logueado, el Rol guardado en la sesion (viene de COE.User_Admin.Rol)
    debe ser 'admin'. 403, no 401: la persona SI esta autenticada, solo no
    tiene permiso para esto en particular."""
    user = require_login_api(request)
    if user.get("rol") != "admin":
        raise HTTPException(status_code=403, detail="Esta sección es solo para administradores.")
    return user


@app.get("/login")
def login_form(request: Request):
    if _current_user(request):
        return RedirectResponse("/", status_code=303)
    if not config.DB_CONFIGURED:
        return templates.TemplateResponse(request, "login.html", {
            "error": "El login no está configurado en este servidor (faltan DB_SERVER/DB_DATABASE/"
                     "DB_UID/DB_PWD en .env) — contacta al equipo técnico.",
        })
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    username = (form.get("username") or "").strip()
    password = form.get("password") or ""
    ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    login_key = username.lower()

    if not config.DB_CONFIGURED:
        return templates.TemplateResponse(request, "login.html", {
            "error": "El login no está configurado en este servidor — contacta al equipo técnico.",
        }, status_code=503)

    if login_key and _login_bloqueado(login_key):
        await run_in_threadpool(db.log_access, username, False, ip, user_agent, None)
        return templates.TemplateResponse(request, "login.html", {
            "error": "Demasiados intentos fallidos con este usuario — espera unos minutos antes de volver a intentar.",
        }, status_code=429)

    try:
        user = await run_in_threadpool(db.verify_login, username, password)
    except Exception as e:
        # El detalle real (server, driver, etc.) NUNCA se le muestra a quien
        # todavia no inicio sesion — antes esto filtraba infraestructura
        # interna (IP del servidor SQL, nombre de la base) a cualquiera que
        # llegara a /login, autenticado o no. El detalle si queda en la
        # consola del servidor, para quien lo administre.
        print(f"[login] error de conexion a la base de autenticacion: {e}")
        return templates.TemplateResponse(request, "login.html", {
            "error": "No se pudo validar el login en este momento — intenta de nuevo en unos minutos "
                     "o contacta al equipo técnico si sigue fallando.",
        }, status_code=502)

    await run_in_threadpool(
        db.log_access, username, user is not None, ip, user_agent,
        user["id"] if user else None,
    )

    if not user:
        if login_key:
            _registrar_intento_fallido(login_key)
        return templates.TemplateResponse(request, "login.html", {
            "error": "Usuario o contraseña incorrectos.",
        }, status_code=401)

    _limpiar_intentos(login_key)
    request.session["user"] = user
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ── Administracion de usuarios (solo rol admin) ─────────────────────────────

async def _render_admin(request: Request, user: dict, error: str | None, ok_msg: str | None):
    """Arma la respuesta del panel de administracion SIEMPRE de forma segura:
    incluso si db.list_users() falla (conexion caida, driver raro, lo que
    sea), esto se atrapa aqui y se muestra como mensaje en la pagina — nunca
    como un 500 en blanco. Un solo lugar para las 4 rutas de abajo, en vez
    de repetir el mismo try/except (y el mismo hueco que tenia antes, donde
    el refresco de la lista en los POST quedaba SIN atrapar) cuatro veces."""
    try:
        usuarios = await run_in_threadpool(db.list_users)
    except Exception as e:
        usuarios = []
        error = error or f"No se pudo leer la lista de usuarios: {e}"
    return templates.TemplateResponse(request, "admin.html", {
        "current_user": user, "usuarios": usuarios, "error": error, "ok_msg": ok_msg,
    })


@app.get("/admin/usuarios")
async def admin_usuarios(request: Request, user: dict = Depends(require_admin)):
    return await _render_admin(request, user, None, None)


@app.post("/admin/usuarios")
async def admin_crear_usuario(request: Request, user: dict = Depends(require_admin)):
    form = await request.form()
    try:
        await run_in_threadpool(
            db.create_user,
            (form.get("username") or "").strip(),
            (form.get("nombre") or "").strip(),
            (form.get("correo") or "").strip(),
            (form.get("rol") or "usuario").strip(),
            form.get("password") or "",
        )
        ok_msg, error = f"Usuario '{form.get('username')}' creado.", None
    except db.UserError as e:
        ok_msg, error = None, str(e)
    except Exception as e:
        ok_msg, error = None, f"No se pudo crear el usuario: {e}"
    return await _render_admin(request, user, error, ok_msg)


@app.post("/admin/usuarios/{user_id}/toggle")
async def admin_toggle_usuario(request: Request, user_id: int, user: dict = Depends(require_admin)):
    form = await request.form()
    activar = form.get("activo") == "1"
    try:
        await run_in_threadpool(db.set_user_active, user_id, activar)
        ok_msg, error = ("Usuario habilitado." if activar else "Usuario deshabilitado."), None
    except db.UserError as e:
        ok_msg, error = None, str(e)
    except Exception as e:
        ok_msg, error = None, f"No se pudo actualizar: {e}"
    return await _render_admin(request, user, error, ok_msg)


@app.post("/admin/usuarios/{user_id}/password")
async def admin_cambiar_password(request: Request, user_id: int, user: dict = Depends(require_admin)):
    form = await request.form()
    nueva = form.get("password") or ""
    try:
        await run_in_threadpool(db.set_user_password, user_id, nueva)
        ok_msg, error = "Contraseña actualizada.", None
    except db.UserError as e:
        ok_msg, error = None, str(e)
    except Exception as e:
        ok_msg, error = None, f"No se pudo actualizar la contraseña: {e}"
    return await _render_admin(request, user, error, ok_msg)


@app.get("/")
def index(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "index.html", {
        "embed_url": config.POWERBI_EMBED_URL,
        "original_url": config.POWERBI_ORIGINAL_URL,
        "azure_embed_enabled": config.AZURE_EMBED_ENABLED,
        "azure_client_id": config.AZURE_CLIENT_ID,
        "azure_tenant_id": config.AZURE_TENANT_ID,
        "powerbi_group_id": config.POWERBI_GROUP_ID,
        "powerbi_report_id": config.POWERBI_REPORT_ID,
        "current_user": user,
    })


@app.get("/api/status")
def api_status(user: dict = Depends(require_login_api)):
    return {
        "llm": llm.get_status(),
        "data": {"ok": dataset.is_ready(), **(dataset.get_source_info() if dataset.is_ready() else {})},
        "azure_embed_enabled": config.AZURE_EMBED_ENABLED,
    }


@app.get("/api/overview")
def api_overview(user: dict = Depends(require_login_api)):
    if not dataset.is_ready():
        return JSONResponse({"error": "Datos no extraidos todavia. Usa el boton Actualizar datos."}, status_code=503)
    return {"source": dataset.get_source_info(), "tables": dataset.get_overview()}


@app.post("/api/refresh")
def api_refresh(user: dict = Depends(require_admin)):
    """Vuelve a leer el .pbix de tablero/ y refresca el cache — el 'tiempo real'
    de este modelo: sin Premium/PPU no existe query en vivo a la API de Power
    BI, asi que la forma simple de reflejar cambios es re-extraer el archivo
    (reemplazalo en tablero/ y presiona este boton). Solo admin: afecta a
    TODOS los usuarios a la vez (reemplaza el dataset compartido), no deberia
    poder dispararlo cualquiera con sesion."""
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
async def api_chat(request: Request, user: dict = Depends(require_login_api)):
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
async def api_report(user: dict = Depends(require_login_api)):
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
