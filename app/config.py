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
