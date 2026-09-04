"""
Acceso genérico al dataset extraído del .pbix (data/pbix/), sin asumir el
esquema de ningún reporte en particular. Funciona igual con cualquier
tablero: arma un resumen de tablas/columnas para darle contexto al chat, en
vez de calcular KPIs de negocio hardcodeados a un reporte específico.
"""
import json
import re
from functools import lru_cache

import pandas as pd

from . import governance
from .config import BASE_DIR

CACHE_DIR = BASE_DIR / "data" / "pbix"

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9_%]{4,}")


def tokenize(text: str) -> list[str]:
    """Extrae palabras/identificadores de >=4 caracteres, separando por
    espacios y por puntuacion/corchetes — soporta sintaxis DAX tipo
    "Tabla[Columna]" (sin esto, ese token queda pegado y no matchea nada)."""
    return _TOKEN_RE.findall(text)


def is_ready() -> bool:
    return (CACHE_DIR / "profile.json").exists() and (CACHE_DIR / "manifest.json").exists()


def clear_cache():
    """Invalida los datos en memoria (llamar despues de una re-extraccion)."""
    for fn in (_manifest, _profile, _measures, _relationships, _schema, _usage, _used_measure_keys, _used_column_keys, _find_periodo_table):
        fn.cache_clear()


def data_version() -> float:
    """Fingerprint barato del snapshot de datos actual (mtime de
    manifest.json, que SIEMPRE se reescribe al final de cada extraccion) —
    usado por el Agente de Consistencia de agents.py para saber cuando
    invalidar respuestas cacheadas. No es un contador en memoria a
    proposito: sobrevive a un reinicio del servidor sin quedar desfasado."""
    try:
        return (CACHE_DIR / "manifest.json").stat().st_mtime
    except FileNotFoundError:
        return 0.0


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _profile() -> dict:
    return json.loads((CACHE_DIR / "profile.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _relationships() -> list:
    path = CACHE_DIR / "relationships.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def get_relationships(from_table: str = None) -> list:
    """Relaciones del modelo (ej. Ventas_Recibos_contact.PERIODO -> Dimension_
    tiempo_periodo.PERIODO). Si se pasa from_table, filtra solo esa tabla."""
    rels = _relationships()
    if from_table is None:
        return rels
    return [r for r in rels if r["FromTableName"] == from_table]


@lru_cache(maxsize=1)
def _find_periodo_table() -> str | None:
    """Encuentra la tabla de dimension de TIEMPO/PERIODO que ya trae la
    logica de periodos academicos resuelta (columnas "Periodo_Anterior" +
    "PERIODO") — se detecta por el PATRON de columna, no por un nombre de
    tabla fijo, para que esto siga funcionando igual con cualquier reporte
    que use esa misma convencion (o simplemente no haga nada si no la usa)."""
    for table, info in _manifest().get("tables", {}).items():
        cols_lower = {str(c).lower() for c in info["columns"]}
        if "periodo_anterior" in cols_lower and "periodo" in cols_lower:
            return table
    return None


def periodo_info(periodo: str) -> dict | None:
    """Metadata REAL de un periodo academico (año, tipo, si es relevante, y
    cual es el periodo anterior) — se lee directo de la tabla de tiempo
    curada por negocio, nunca se adivina con heuristicas de texto sobre el
    codigo del periodo."""
    table = _find_periodo_table()
    if not table:
        return None
    df = load_table(table)
    col_map = {c.lower(): c for c in df.columns}
    periodo_col = col_map.get("periodo")
    if not periodo_col:
        return None
    row = df[df[periodo_col].astype(str).str.upper() == periodo.upper()]
    if row.empty:
        return None
    r = row.iloc[0]

    def get(name):
        c = col_map.get(name)
        return str(r[c]) if c else None

    relevante_raw = get("periodo_relevante") or ""
    return {
        "tabla": table,
        "columna_periodo": periodo_col,
        "periodo": get("periodo"),
        "año": get("año") or get("ano"),
        "tipo": get("tipo"),
        "relevante": relevante_raw.strip().upper() == "SI",
        "anterior": get("periodo_anterior"),
    }


@lru_cache(maxsize=1)
def _usage() -> dict:
    """Que columnas/medidas estan REALMENTE puestas en algun visual del
    reporte (calculado una vez en la extraccion, ver pbix_loader._build_usage).
    Un modelo real suele acumular medidas y tablas viejas que ya nadie
    referencia en pantalla; esto le da a los agentes una senal objetiva
    ("de verdad se usa") para desempatar entre nombres parecidos."""
    path = CACHE_DIR / "usage.json"
    if not path.exists():
        return {"used_columns": [], "used_measures": []}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _used_measure_keys() -> frozenset:
    return frozenset((u["table"], u["name"]) for u in _usage().get("used_measures", []))


@lru_cache(maxsize=1)
def _used_column_keys() -> frozenset:
    return frozenset((u["table"], u["column"]) for u in _usage().get("used_columns", []))


def is_measure_used(table: str, name: str) -> bool:
    return (table, name) in _used_measure_keys()


def is_column_used(table: str, column: str) -> bool:
    return (table, column) in _used_column_keys()


@lru_cache(maxsize=1)
def _schema() -> dict:
    """Capa intermedia de esquema entidad-relacion (precalculada en la
    extraccion): rol de cada tabla (hechos/dimension/aislada) + con que
    otras tablas se conecta y por que columna. Ver pbix_loader._build_schema."""
    path = CACHE_DIR / "schema.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def describe_schema(max_fact_tables: int = 6, max_rel_per_table: int = 8) -> str:
    """Resumen legible del esquema entidad-relacion para el system prompt del
    chat: que tablas son de hechos, cuales de dimension, y como se conectan.
    Le da a los agentes el panorama completo del modelo (no solo tablas
    sueltas), asi entienden que una tabla de dimension NO es lo mismo que
    una de hechos (ej. una meta/proyeccion no es lo mismo que lo ejecutado)."""
    schema = _schema()
    if not schema:
        return ""
    facts = sorted(
        ((t, s) for t, s in schema.items() if s["role"] == "hechos"),
        key=lambda kv: -kv[1]["rows"],
    )[:max_fact_tables]
    dims = sorted(t for t, s in schema.items() if s["role"] == "dimension")

    lines = ["Esquema entidad-relacion del modelo:"]
    for t, s in facts:
        rels = s["relaciones"][:max_rel_per_table]
        rel_txt = "; ".join(f"\"{r['tabla_relacionada']}\" (via {r['columna']})" for r in rels)
        lines.append(f"- \"{t}\" — tabla de HECHOS ({_fmt_num(s['rows'])} filas). Se conecta con: {rel_txt or 'ninguna'}.")
    if dims:
        lines.append(f"- Tablas de DIMENSION (catalogos/lookup, NO son transacciones): {', '.join(dims)}.")
    lines.append(
        "Una tabla de dimension describe/clasifica (ej. catalogo de programas, "
        "de periodos de tiempo); una tabla de hechos registra transacciones "
        "reales. No confundas una meta/proyeccion (a menudo en su propia "
        "tabla de hechos separada) con lo realmente ejecutado."
    )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _measures() -> list:
    path = CACHE_DIR / "measures.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@lru_cache(maxsize=8)
def load_table(table: str) -> pd.DataFrame:
    info = _manifest()["tables"][table]
    return pd.read_parquet(CACHE_DIR / info["file"])


def _plural_variants(word: str) -> set:
    """Variantes simples de plural/singular en español (evita que 'ingresos'
    no matchee 'ingreso' o viceversa por una sola 's' de diferencia)."""
    w = word.lower()
    variants = {w}
    if w.endswith("s") and len(w) > 4:
        variants.add(w[:-1])
    else:
        variants.add(w + "s")
    return variants


def find_measure(query: str, limit: int = 5) -> list:
    """Busca medidas DAX cuyo nombre contenga `query` (tolera plural/singular).
    Las que de verdad estan en uso (usage.json) van primero — un modelo real
    acumula variantes/duplicados abandonados, y si hay mas candidatas que
    `limit` no queremos que una vieja le quite el cupo a la vigente.

    Los duplicados marcados como eliminados en reglas_negocio.json (ej.
    Ingreso_Neto_Ejecutado -> Ingreso_Neto_General) se sustituyen aqui por
    su medida vigente ANTES de buscar/puntuar — asi el nombre viejo nunca
    llega a agents.py ni al contexto del chat, sin importar por que palabra
    lo haya encontrado quien pregunta."""
    variants = _plural_variants(query)
    hits, seen = [], set()
    for m in _measures():
        name_lower = m["Name"].lower()
        if any(v in name_lower for v in variants):
            canonical = governance.resolve_alias(m["Name"])
            target = m if canonical == m["Name"] else next(
                (mm for mm in _measures() if mm["Name"] == canonical), m
            )
            key = target["Name"]
            if key not in seen:
                seen.add(key)
                hits.append(target)
    hits.sort(key=lambda m: not is_measure_used(m["TableName"], m["Name"]))
    return hits[:limit]


def find_column(query: str, limit: int = 5) -> list:
    """Busca columnas (en cualquier tabla) cuyo nombre contenga `query` (tolera
    plural/singular). Igual que find_measure, las columnas en uso real van
    primero para no perderlas por el limite cuando hay varias candidatas."""
    variants = _plural_variants(query)
    hits = []
    for table, prof in _profile().items():
        for col in prof["columns"]:
            name_lower = col["name"].lower()
            if any(v in name_lower for v in variants):
                hits.append({"table": table, **col})
    hits.sort(key=lambda c: not is_column_used(c["table"], c["name"]))
    return hits[:limit]


def get_overview() -> list:
    """Lista de tablas con su tamano, para mostrar en la UI (mas grandes primero)."""
    return [
        {"table": t, "rows": p["rows"], "columns": len(p["columns"])}
        for t, p in sorted(_profile().items(), key=lambda kv: -kv[1]["rows"])
    ]


def get_source_info() -> dict:
    m = _manifest()
    return {"source_file": m.get("source_file"), "tables": len(m.get("tables", {}))}


def top_values_map(table: str) -> dict:
    """{columna: [valores mas frecuentes]} segun el PERFIL ya calculado —
    permite detectar un filtro categorico comparando texto en memoria, SIN
    tocar el DataFrame en vivo (nunique()/unique() sobre una tabla de
    cientos de miles de filas es lento y se repetiria en cada pregunta)."""
    cols = _profile().get(table, {}).get("columns", [])
    return {c["name"]: [v["value"] for v in c["top_values"]] for c in cols if "top_values" in c}


def _fmt_num(n) -> str:
    return f"{n:,.0f}".replace(",", ".")


def build_context_text(max_tables: int = 3, max_cols_per_table: int = 6) -> str:
    """Resumen compacto de TODO el dataset (cualquier esquema) para el system prompt."""
    prof = _profile()
    if not prof:
        return ""
    tables = sorted(prof.items(), key=lambda kv: -kv[1]["rows"])[:max_tables]
    lines = [f"Dataset con {len(prof)} tablas. Resumen de las mas grandes (probablemente las de hechos):"]
    for table, p in tables:
        lines.append(f"\nTabla \"{table}\" — {_fmt_num(p['rows'])} filas:")
        for c in p["columns"][:max_cols_per_table]:
            if "numeric" in c:
                n = c["numeric"]
                lines.append(f"  - {c['name']} (numerico): total={_fmt_num(n['sum'])}, promedio={n['mean']:.1f}, min={_fmt_num(n['min'])}, max={_fmt_num(n['max'])}")
            elif "top_values" in c:
                tv = ", ".join(f"{v['value']} ({v['count']})" for v in c["top_values"][:6])
                lines.append(f"  - {c['name']} (categorico, {c['distinct']} valores): {tv}")
            elif "date_range" in c:
                lines.append(f"  - {c['name']} (fecha): de {c['date_range']['min']} a {c['date_range']['max']}")
    return "\n".join(lines)


def build_relevant_snippets(pregunta: str, limit: int = 6) -> list:
    """Medidas DAX y columnas cuyo nombre aparece mencionado en la pregunta (contexto puntual)."""
    words = [w.lower() for w in tokenize(pregunta)]
    found, seen = [], set()
    for w in words:
        for m in find_measure(w, limit=3):
            key = ("measure", m["Name"])
            if key in seen:
                continue
            seen.add(key)
            blocked = governance.block_message(m["Name"])
            if blocked:
                # Sin formula util que mostrar (esta pendiente de definicion
                # de negocio) — que el LLM vea el motivo, no una DAX vacia.
                found.append(f"Medida \"{m['Name']}\" ({m['TableName']}): BLOQUEADA — {blocked}")
                continue
            snippet = f"Medida \"{m['Name']}\" ({m['TableName']}): {m['Expression'].strip()[:250]}"
            for note in governance.disclaimers(m["Name"]):
                snippet += f" [AVISO OBLIGATORIO, incluir en la respuesta: {note}]"
            found.append(snippet)
        for c in find_column(w, limit=3):
            key = ("column", c["table"], c["name"])
            if key not in seen:
                seen.add(key)
                if "numeric" in c:
                    n = c["numeric"]
                    found.append(
                        f"Columna \"{c['name']}\" en tabla \"{c['table']}\" (numerico): "
                        f"total={_fmt_num(n['sum'])}, promedio={n['mean']:.1f}, min={_fmt_num(n['min'])}, max={_fmt_num(n['max'])}"
                    )
                else:
                    found.append(f"Columna \"{c['name']}\" en tabla \"{c['table']}\" ({c['dtype']}, {c['distinct']} valores distintos)")
    return found[:limit]
