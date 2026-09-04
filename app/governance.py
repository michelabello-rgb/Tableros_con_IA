"""
Reglas de gobernanza de negocio sobre medidas DAX especificas: alias de
medidas eliminadas/duplicadas, medidas bloqueadas por falta de definicion,
avisos obligatorios sobre casi-duplicados que podrian dar resultados
distintos, y medidas cuyo significado de negocio esta pendiente de
confirmar.

Vive separado de agents.py y de data/pbix/ a proposito: son decisiones de
NEGOCIO (quien es el duplicado vigente, que medida no se puede calcular
todavia), no algo que se derive del modelo de datos ni del codigo — se
actualizan editando reglas_negocio.json, sin tocar Python, y sobreviven a
una re-extraccion del .pbix (que sí borra y recalcula todo lo de
data/pbix/). Ver GOBERNANZA.md para el detalle de cada regla vigente.
"""
import json
import re
from functools import lru_cache

from .config import BASE_DIR

RULES_PATH = BASE_DIR / "app" / "reglas_negocio.json"


@lru_cache(maxsize=1)
def _rules() -> dict:
    if not RULES_PATH.exists():
        return {}
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def clear_cache():
    _rules.cache_clear()


def resolve_alias(name: str) -> str:
    """Si `name` es una medida eliminada/duplicada (ej. Ingreso_Neto_Ejecutado),
    devuelve el nombre vigente que la reemplaza (Ingreso_Neto_General). Si no
    aplica ninguna regla, devuelve `name` tal cual."""
    return _rules().get("alias_medidas", {}).get(name, name)


def block_message(name: str) -> str | None:
    """Mensaje a devolver EN VEZ de calcular un valor, si la medida esta
    bloqueada (sin formula documentada, pendiente de definicion de negocio).
    None si la medida no esta bloqueada."""
    return _rules().get("medidas_bloqueadas", {}).get(name)


def disclaimers(name: str) -> list[str]:
    """Avisos que deben acompañar SIEMPRE a una respuesta que use `name` —
    casi-duplicados con resultado potencialmente distinto, o una regla de
    negocio sobre esta medida que aun no esta confirmada. Nunca se omiten
    en silencio: si la lista no esta vacia, agents.py debe mostrarlos."""
    notes = []
    dup = _rules().get("casi_duplicados", {}).get(name)
    if dup:
        notes.append(dup["aviso"])
    pending = _rules().get("medidas_con_regla_pendiente", {}).get(name)
    if pending:
        notes.append(pending)
    return notes


def match_synonym(pregunta_lower: str) -> str | None:
    """Si la pregunta menciona un sinonimo de negocio de un grupo de
    casi-duplicados (ej. "financiado" -> familia RECAUDO_CARTERA/
    RECAUDO_financiaciones) sin que el nombre de ninguna medida haga match
    textual directo, devuelve la medida por defecto de ese grupo — para que
    "recaudo financiado" dispare el mismo aviso obligatorio que "recaudo de
    cartera", tal como pide la regla de negocio, en vez de no responder
    nada por una simple diferencia de raiz de palabra."""
    for name, info in _rules().get("casi_duplicados", {}).items():
        for syn in info.get("sinonimos", []):
            if syn.lower() in pregunta_lower:
                return name
    return None


def default_measure(pregunta_lower: str) -> str | None:
    """Si la pregunta menciona una palabra generica sola (ej. "recaudo", sin
    "cartera"/"financiado"/etc.) que por si sola no alcanza a distinguir
    entre varias medidas parecidas, devuelve la medida por defecto que el
    negocio confirmo para ese caso (ej. "recaudo" -> RECAUDO_GRAL, que es
    SUM(Ventas_Recibos_contact[VALOR_RECAUDO])) — SOLO si ninguno de los
    calificativos que activarian una medida mas especifica esta presente."""
    for word, info in _rules().get("medida_por_defecto", {}).items():
        if not re.search(rf"\b{re.escape(word)}\b", pregunta_lower):
            continue
        if any(q in pregunta_lower for q in info.get("solo_si_no_hay", [])):
            continue
        return info["medida"]
    return None


def requires_period(name: str) -> bool:
    """True si esta medida no tiene sentido sin un año/periodo explicito en
    la pregunta (configurable en reglas_negocio.json por el equipo de
    negocio — vacio por defecto, no se asume nada sin que lo confirmen)."""
    return name in _rules().get("medidas_requieren_periodo", [])
