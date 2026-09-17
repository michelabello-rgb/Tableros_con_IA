import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"

# Sin valores por defecto reales a proposito (proyecto generico, publicable
# sin exponer el reporte de ninguna instalacion en particular) — configuralos
# en tu propio .env: POWERBI_REPORT_URL / POWERBI_GROUP_ID / POWERBI_REPORT_ID.
POWERBI_ORIGINAL_URL = os.environ.get("POWERBI_REPORT_URL", "")
POWERBI_GROUP_ID = os.environ.get("POWERBI_GROUP_ID", "")
POWERBI_REPORT_ID = os.environ.get("POWERBI_REPORT_ID", "")
POWERBI_EMBED_URL = (
    f"https://app.powerbi.com/reportEmbed?reportId={POWERBI_REPORT_ID}&groupId={POWERBI_GROUP_ID}"
    if POWERBI_GROUP_ID and POWERBI_REPORT_ID else ""
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")

OLLAMA_HOST = OLLAMA_URL[:-3] if OLLAMA_URL.endswith("/v1") else OLLAMA_URL
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")  
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))  
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m") 

# Embed real (interactivo, dentro de la pagina) via Azure AD + MSAL.js.
AZURE_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID", "")
AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
AZURE_EMBED_ENABLED = bool(AZURE_CLIENT_ID and AZURE_TENANT_ID)

# Login contra SQL Server (requiere las tablas COE.User_Admin y COE.Acceso_Log ya creadas en el servidor).
# DB_PWD nunca tiene default: sin ella, db.py se rehusa a conectar en vez de
# intentar con una contraseña vacia contra un servidor de produccion.
DB_SERVER = os.environ.get("DB_SERVER", "")
DB_DATABASE = os.environ.get("DB_DATABASE", "")
DB_UID = os.environ.get("DB_UID", "")
DB_PWD = os.environ.get("DB_PWD", "")
DB_CONFIGURED = bool(DB_SERVER and DB_DATABASE and DB_UID and DB_PWD)

# Firma las cookies de sesion del login — SIN default real a proposito: en
# produccion hay que poner una propia en .env (ej. `python -c "import
# secrets; print(secrets.token_hex(32))"`). Si no esta configurada, se usa
# una aleatoria generada en cada arranque (las sesiones no sobreviven un
# reinicio del servidor, pero al menos no queda una clave fija y predecible
# en el codigo fuente).
import secrets as _secrets
SESSION_SECRET = os.environ.get("SESSION_SECRET") or _secrets.token_hex(32)
