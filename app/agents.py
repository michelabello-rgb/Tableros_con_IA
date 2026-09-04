"""
Orquestación de agentes para el chat del tablero.

Como se trata de información financiera, el diseño prioriza precisión sobre
velocidad: separar "redactar" de "verificar" reduce el riesgo de que una
cifra inventada llegue al usuario sin revisión. No todos los pasos usan el
LLM (más pasos con LLM = más lento y más superficies de alucinación posible
con modelos locales pequeños) — el enrutador y el verificador de cifras son
deterministas (Python puro), y solo se llama al modelo dos veces como máximo
por pregunta (redactor, y corrector solo si el verificador encuentra algo).

Pipeline:
  0. Agente de respuesta directa (Python, sin LLM) -> lookups simples e inequívocos
  1. Agente Enrutador   (heurística, sin LLM)  -> clasifica la intención
  2. Agente Investigador (Python, sin LLM)     -> arma el contexto relevante
  3. Agente Redactor     (LLM, 1 llamada)      -> borrador de respuesta
  4. Agente Verificador  (Python, sin LLM)     -> chequea cifras contra el contexto
  5. Agente Corrector    (LLM, solo si hace falta) -> corrige cifras no verificadas
  6. Agente Predictivo   (Python + numpy, sin LLM) -> proyecciones, anomalías, agrupamientos
  7. Agente Validador    (Python + numpy, sin LLM) -> chequea que las cifras del
                                                       motor directo sean matemáticamente
                                                       consistentes (recómputo independiente,
                                                       rango plausible, conservación, integridad)
  8. Agente de Consistencia (Python, sin LLM)      -> misma pregunta + mismos datos =
                                                       SIEMPRE la misma respuesta (cache por
                                                       pregunta+versión de datos), y baja la
                                                       temperatura/fija el seed del LLM para
                                                       que ni siquiera la primera vez varíe
                                                       de más
"""
import re
import unicodedata
import pandas as pd
from . import llm, dataset, ml, validator, governance


def _normalize_text(s: str) -> str:
    """Quita tildes/diacriticos y pasa a minusculas — regla de negocio: los
    nombres de programa/ciudad estan en mayusculas y sin tilde en la base,
    asi que la entrada del usuario ("Bogotá", "bogota", "BOGOTÁ") debe
    normalizarse igual antes de comparar, en vez de fallar silenciosamente
    (o disparar una aclaracion innecesaria) solo por una tilde de mas."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()

# ── 1. Agente Enrutador ──────────────────────────────────────────────────────

GREETING_RE = re.compile(
    r"^\s*(hola|buenos? d[ií]as|buenas( tardes| noches)?|hey|qu[eé] tal|gracias|listo|ok|vale)\W*$",
    re.IGNORECASE,
)
REPORT_KEYWORDS = ("reporte", "resumen ejecutivo", "informe corto")
COMPARISON_KEYWORDS = (" vs ", "compara", "diferencia entre", "más que", "menos que", "mejor que")
# Codigo de periodo academico: 2-4 digitos + 1-3 letras + 0-2 digitos
# (ej. "2025B", "24V05", "25ET1", "26ES6", "24BE3", "2025Q").
_PERIODO_CODE_RE = re.compile(r"\b\d{2,4}[A-Za-z]{1,3}\d{0,2}\b")


def classify_intent(pregunta: str) -> str:
    """Clasifica la pregunta sin usar el LLM (rápido, gratis, 100% determinista)."""
    p = pregunta.strip().lower()
    if GREETING_RE.match(p):
        return "saludo"
    if any(k in p for k in REPORT_KEYWORDS):
        return "reporte"
    if any(k in p for k in COMPARISON_KEYWORDS):
        return "comparacion"
    if any(k in p for k in (
        "qué significa", "que significa", "explica", "explíca", "cómo leo", "como leo", "qué es", "que es",
        "qué debería revisar", "que deberia revisar", "por dónde empiezo", "por donde empiezo",
        "cómo le saco provecho", "como le saco provecho", "cómo aprovecho", "como aprovecho",
        "recomiendas", "recomendación", "recomendacion", "guía rápida", "guia rapida", "soy nuevo",
    )):
        return "explicacion"
    return "general"


_AGG_PATTERNS = [
    (re.compile(r"\bsuma\w*", re.IGNORECASE), "sum"),
    (re.compile(r"\btotal\w*", re.IGNORECASE), "sum"),
    (re.compile(r"\bpromedi\w*", re.IGNORECASE), "mean"),
    (re.compile(r"\bmedia\b", re.IGNORECASE), "mean"),
    (re.compile(r"\bm[aá]xim\w*", re.IGNORECASE), "max"),
    (re.compile(r"\bmayor\b", re.IGNORECASE), "max"),
    (re.compile(r"\bm[ií]nim\w*", re.IGNORECASE), "min"),
    (re.compile(r"\bmenor\b", re.IGNORECASE), "min"),
]

_LOOKUP_STOPWORDS = {
    "dame", "cual", "cuales", "cuánto", "cuanto", "tabla", "columna", "dato", "cifra",
    "valor", "para", "ejecutado", "ejecutados", "ejecutada", "muestrame", "muéstrame",
    "quiero", "dime", "sobre",
}
_MEASURE_STOPWORDS = _LOOKUP_STOPWORDS - {"ejecutado", "ejecutados", "ejecutada"}

_NAME_PART_RE = re.compile(r"[_\s]+")


def _score_numeric_columns(candidate_words: list[str]) -> dict[tuple[str, str], int]:
    """Puntua columnas NUMERICAS candidatas por cuantas PARTES de su nombre
    aparecen en la pregunta (igual criterio que _match_named_measure con
    medidas), en vez de exigir que una sola palabra sea igual al nombre
    completo de la columna — si no, una columna corta como "ORDEN" gana por
    accidente contra "ORDEN_NETO" cuando preguntan por "orden neto" (las dos
    palabras juntas), porque "orden" sola ya calza exacto con la columna
    corta antes de considerar "neto". Usado tanto por el motor de respuesta
    directa (sumas/promedios) como por el predictivo (proyeccion/anomalia/
    agrupamiento) — mismo bug, mismo arreglo, un solo lugar."""
    if not candidate_words:
        return {}
    words_set = {w.lower() for w in candidate_words}
    scores, seen = {}, set()
    for w in candidate_words:
        for c in dataset.find_column(w, limit=8):
            if "numeric" not in c:
                continue
            key = (c["table"], c["name"])
            if key in seen:
                continue
            seen.add(key)
            parts = [p for p in _NAME_PART_RE.split(c["name"].lower()) if p]
            score = sum(1 for p in parts if p in words_set)
            if parts and score == len(parts):
                score += 10  # bonus: TODAS las partes del nombre estan en la pregunta
            if score:
                scores[key] = score
    return scores


def _fmt_direct(n: float) -> str:
    """Numero plano de miles (sin simbolo de moneda) — para conteos que NO
    son dinero: filas, tablas, estudiantes, medidas, etc."""
    return f"{n:,.0f}".replace(",", ".")


# Nombres de columna/medida que suenan a cifra MONETARIA (regla de negocio:
# formato pesos colombianos, "$X.XXX.XXX,XX" — nunca formato ingles, nunca
# sin simbolo). No es exhaustivo a proposito: ante la duda, mejor un numero
# plano de mas que un "$" de menos en algo que no era dinero.
_MONEY_NAME_RE = re.compile(
    r"ingreso|recaudo|\borden\b|orden_neto|valor|monto|cuota|cr[eé]dito|cartera|financiaci[oó]n|\bpago\b",
    re.IGNORECASE,
)


def _fmt_money(n: float) -> str:
    """Formato pesos colombianos: punto de miles, coma decimal, prefijo $
    (ej. "$35.230.896.108,07") — regla de negocio, nunca formato ingles."""
    s = f"{n:,.2f}"  # arranca en formato ingles: "35,230,896,108.07"
    s = s.replace(",", "|").replace(".", ",").replace("|", ".")
    return f"${s}"


def _fmt_smart(name: str, val: float) -> str:
    """Elige el formato de salida segun el NOMBRE de la columna/medida:
    porcentaje si es una razon, pesos colombianos si el nombre suena a cifra
    monetaria, o un numero plano de miles si es un conteo (estudiantes,
    matriculas, etc.) — la MISMA heuristica de razon que ya usaba
    _fmt_measure_value, mas el formato de moneda como regla de negocio."""
    looks_like_ratio = "%" in name or "cumplimiento" in name.lower() or abs(val) <= 5
    if looks_like_ratio:
        return f"{val:.1%}"
    if _MONEY_NAME_RE.search(name):
        return _fmt_money(val)
    return _fmt_direct(val)


_YEAR_COL_RE = re.compile(r"^a[nñ]o$|^year$", re.IGNORECASE)


def _year_filter_via_relationship(df: pd.DataFrame, table: str, year: str):
    """Filtra por año usando una relacion REAL del modelo (ej.
    Ventas_Recibos_contact.PERIODO -> Dimension_tiempo_periodo.PERIODO, que
    trae la columna "Año") en vez de adivinar por el formato del codigo —
    mas confiable, y funciona con cualquier reporte que tenga una dimension
    de tiempo con una columna literalmente llamada "Año"/"Year"."""
    for rel in dataset.get_relationships(table):
        to_table, to_col, from_col = rel["ToTableName"], rel["ToColumnName"], rel["FromColumnName"]
        if from_col not in df.columns:
            continue
        try:
            dim = dataset.load_table(to_table)
        except Exception:
            continue
        if to_col not in dim.columns:
            continue
        year_cols = [c for c in dim.columns if _YEAR_COL_RE.match(str(c))]
        if not year_cols:
            continue
        year_col = year_cols[0]
        mapping = dim.drop_duplicates(subset=[to_col]).set_index(to_col)[year_col].astype(str)
        mask = df[from_col].map(mapping) == year
        if mask.any():
            return mask, f'{to_table}."{year_col}" = "{year}" (vía {table}.{from_col})'
    return None, None

_ENTITY_TRIGGERS = {
    "programa": "PROGRAMA", "ciudad": "CIUDAD", "sede": "SEDE",
    "regional": "REGIONAL", "localidad": "LOCALIDAD", "departamento": "DEPARTAMENTO",
}
_CAPWORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{3,}")


def _check_unmatched_entity(pregunta: str, table: str, applied: list[str]) -> str | None:
    p_lower = pregunta.lower()
    trigger = next((k for k in _ENTITY_TRIGGERS if re.search(rf"\b{k}\w*", p_lower)), None)
    if not trigger:
        return None
    hint = _ENTITY_TRIGGERS[trigger]
    if any(hint in a.upper() for a in applied):
        return None  # ya se aplico un filtro de esta familia — no hay ambiguedad
    candidates = [
        w for w in _CAPWORD_RE.findall(pregunta)
        if w[0].isupper() and w.lower() not in _LOOKUP_STOPWORDS and not w.lower().startswith(trigger)
    ]
    if not candidates:
        return None  # el trigger esta pero no hay ningun nombre propio que intentar resolver
    top_map = dataset.top_values_map(table)
    known_col = next((c for c in top_map if hint in c.upper()), None)
    ejemplos = ", ".join(top_map[known_col][:5]) if known_col else None
    msg = (
        f"No encontré \"{candidates[0]}\" como un valor real de {trigger} en los datos — puede que "
        f"el nombre no coincida exactamente con como está escrito en el modelo (mayúsculas, acentos, "
        f"abreviatura). "
    )
    if ejemplos:
        msg += f"Algunos valores reales que sí existen: {ejemplos}. "
    msg += "¿Puedes confirmarme el nombre exacto?"
    return msg


def _apply_generic_filters(df: pd.DataFrame, pregunta: str, table: str) -> tuple[pd.DataFrame, list[str]]:
    """Filtra el DataFrame segun valores categoricos o un año mencionados en
    la pregunta — sin asumir nombres de columna, comparando contra los
    valores reales de la tabla ya cargada. Si no detecta nada, no filtra."""
    applied = []
    p_norm = _normalize_text(pregunta)

    # Filtro categorico: se compara contra los valores mas frecuentes YA
    # CACHEADOS en el perfil (texto en memoria, sin tocar el DataFrame) — el
    # DataFrame solo se toca una vez, para la columna que de verdad matcheo.
    # Exige limite de palabra (\b): un simple "in" haria que, por ejemplo,
    # "Ventas_Recibos_contact" matcheara por accidente el valor "CONTACT" de
    # otra columna, porque la palabra queda pegada dentro del nombre de tabla.
    # Se compara normalizado (sin tildes, minusculas) — regla de negocio: la
    # entrada del usuario ("Bogotá") debe reconocer el valor real de la base
    # ("BOGOTA") sin exigir que lo escriban identico.
    for col, values in dataset.top_values_map(table).items():
        if col not in df.columns:
            continue
        for val_str in values:
            val_norm = _normalize_text(val_str)
            if len(val_norm) >= 3 and re.search(rf"\b{re.escape(val_norm)}\b", p_norm):
                df = df[df[col].astype(str) == val_str]
                applied.append(f'{col} = "{val_str}"')
                break

    # Filtro de año: primero intenta via una relacion real del modelo (mas
    # confiable); si no hay una dimension de tiempo con columna "Año", cae a
    # una heuristica de prefijo sobre la propia tabla como respaldo.
    year_m = re.search(r"\b(20\d{2}|19\d{2})\b", pregunta)
    if year_m and len(df) and not any(year_m.group(1) in a for a in applied):
        year = year_m.group(1)
        mask, desc = _year_filter_via_relationship(df, table, year)
        if mask is not None:
            df = df[mask]
            applied.append(desc)
        else:
            year2 = year[2:]
            already = {a.split(" = ")[0].split(" empieza")[0] for a in applied}
            for col in df.columns:
                if col in already or (df[col].dtype != object and not pd.api.types.is_string_dtype(df[col])):
                    continue
                as_str = df[col].astype(str)
                cmask = as_str.str.startswith(year)
                if not cmask.any():
                    cmask = as_str.str.startswith(year2)
                if cmask.any() and cmask.sum() < len(df):
                    df = df[cmask]
                    applied.append(f'{col} empieza con "{year}"')
                    break

    return df, applied


_SIMPLE_AGG_RE = re.compile(
    r'^\s*(sum|average|min|max)\s*\(\s*[\'"]?([^\[\]\'"]+)[\'"]?\[\s*([^\]]+?)\s*\]\s*\)\s*$',
    re.IGNORECASE,
)
_DAX_AGG_TO_PANDAS = {"sum": "sum", "average": "mean", "min": "min", "max": "max"}


_CALC_SUM_FILTER_RE = re.compile(
    r'CALCULATE\s*\(\s*SUM\s*\(\s*[\'"]?([^\[\]\'"]+)[\'"]?\[\s*([^\]]+?)\s*\]\s*\)\s*,\s*'
    r'[\'"]?([^\[\]\'"]+)[\'"]?\[\s*([^\]]+?)\s*\]\s*=\s*"([^"]+)"',
    re.IGNORECASE | re.DOTALL,
)
_DIVIDE_RE = re.compile(r'^DIVIDE\s*\(\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]', re.IGNORECASE)
_VAR_PASSTHROUGH_RE = re.compile(r'^VAR\s+(\w+)\s*=\s*(.+?)\s*RETURN\s+\1\s*$', re.IGNORECASE | re.DOTALL)
# SUMX(VALUES(Tabla[Col]), CALCULATE(SUM(Tabla2[Col2]))) — patron muy comun en
# medidas tipo "meta"/"ejecucion" (suma por grupo y luego suma total); a nivel
# total (sin filtro por Tabla[Col]) equivale matematicamente a un SUM simple.
_SUMX_VALUES_SUM_RE = re.compile(
    r'SUMX\s*\(\s*VALUES\s*\([^)]+\)\s*,\s*CALCULATE\s*\(\s*\(?\s*SUM\s*\(\s*'
    r'[\'"]?([^\[\]\'"]+)[\'"]?\[\s*([^\]]+?)\s*\]\s*\)\s*\)?\s*\)\s*\)',
    re.IGNORECASE | re.DOTALL,
)
_COMMENT_LINE_RE = re.compile(r'//.*')


def _match_named_measure(pregunta: str) -> str | list[str] | None:
    """Encuentra el NOMBRE de una medida DAX mencionada en la pregunta (ej.
    "ingresos proyectados" -> "Ingreso_Proyectado"). Usa un puntaje (cuantas
    palabras de la pregunta aparecen en el nombre de cada medida candidata)
    en vez de exigir que TODAS coincidan — una palabra descriptiva suelta
    ("general", "total") que no esta literal en el nombre no debe descartar
    la medida correcta. Devuelve: un str si hay un ganador CLARO (unico
    maximo, y al menos 2 palabras si hubo mas de una medida candidata en
    juego); una LISTA de 2+ nombres si de verdad hay ambiguedad real entre
    medidas distintas (regla de gobernanza: nunca elegir en silencio, que el
    llamador pida aclaracion); None si no hay ningun candidato razonable.
    Los duplicados eliminados (reglas_negocio.json) ya llegan resueltos a su
    nombre vigente desde dataset.find_measure, asi que nunca aparecen aqui."""
    # Se excluyen los años: son una instruccion de filtro, no parte del
    # nombre de la medida (si no, "en 2026" descarta medidas legitimas que
    # no tienen "2026" en su nombre, como "Ingreso_Proyectado").
    words = [w for w in dataset.tokenize(pregunta)
             if w.lower() not in _MEASURE_STOPWORDS and not re.fullmatch(r"(19|20)\d{2}", w)]
    scores, all_candidates = {}, set()
    for w in words:
        names = {m["Name"] for m in dataset.find_measure(w, limit=15)}
        all_candidates |= names
        for n in names:
            scores[n] = scores.get(n, 0) + 1
    if not scores:
        return None
    max_score = max(scores.values())
    winners = [n for n, s in scores.items() if s == max_score]
    if len(winners) > 1:
        # Desempate 0: si la pregunta menciona un año EXACTO que aparece
        # literal en el nombre de una sola candidata (ej. "meta de
        # estudiantes 2025" -> "Meta_estudiantes_2025"), esa gana sobre la
        # version general — es una señal mas especifica que preferir la mas
        # corta, y evita resolver en silencio con un simple filtro de año
        # sobre la medida general cuando existe una medida CON SU PROPIA
        # logica de negocio para ese año exacto (pueden dar numeros
        # distintos, como Meta_estudiantes_2025 con su regla Tipo B/C).
        year_m = re.search(r"\b(19|20)\d{2}\b", pregunta)
        if year_m:
            year_specific = [n for n in winners if year_m.group(0) in n]
            if len(year_specific) == 1:
                return year_specific[0]
    if len(winners) > 1:
        # Desempate 1: preferir la que de verdad esta puesta en algun visual
        # del reporte (usage.json) sobre una medida vieja/abandonada con
        # nombre parecido — un modelo real acumula variantes que ya nadie
        # mira en pantalla, y el nombre solo no alcanza para distinguirlas.
        used = [n for n in winners if any(
            dataset.is_measure_used(m["TableName"], n)
            for m in dataset._measures() if m["Name"] == n
        )]
        if used and len(used) < len(winners):
            winners = used
    if len(winners) > 1:
        # Desempate 2: cuando varias medidas siguen empatadas (ej.
        # "Meta_estudiantes" vs "Meta_estudiantes_2024"/"_2025" — mismas
        # palabras, una es la version general y las otras variantes con año
        # fijo en el nombre), se prefiere la mas CORTA — suele ser la
        # version general, y el año que haya en la pregunta ya se aplica
        # como filtro por separado.
        winners.sort(key=len)
        if len(winners[0]) == len(winners[1]):
            # Empate real, ni por longitud se distingue: mas de una medida
            # podria responder esto (regla de gobernanza) — se devuelve la
            # lista para que try_direct_answer pida aclaracion en vez de
            # adivinar o pasarselo en silencio al LLM.
            return sorted(w for w in winners if len(w) == len(winners[0]))
        winners = winners[:1]
    # si hubo mas de una medida candidata en juego, exige al menos 2 palabras
    # de respaldo para evitar ganar solo por una palabra generica suelta.
    if len(all_candidates) > 1 and max_score < 2:
        return None
    return winners[0]


def _resolve_measure_value(name: str, pregunta: str, _depth: int = 0) -> tuple[float, list[str]] | None:
    """Calcula el valor real de una medida DAX para los patrones que se
    pueden recalcular con seguridad: SUM/AVERAGE/MIN/MAX simple, CALCULATE
    con un filtro de igualdad, y DIVIDE entre dos medidas (recursivo, hasta
    2 niveles). Cualquier otra forma de DAX (variables complejas, TOPN,
    CALCULATE con multiples condiciones, etc.) devuelve None a proposito —
    mejor no responder que reinterpretar mal una formula financiera."""
    if _depth > 2:
        return None
    m_def = next((m for m in dataset._measures() if m["Name"] == name), None)
    if not m_def:
        return None
    expr = _COMMENT_LINE_RE.sub("", m_def["Expression"]).strip()
    passthrough = _VAR_PASSTHROUGH_RE.match(expr)
    if passthrough:
        expr = passthrough.group(2).strip()

    ref = re.match(r'^\[([^\]]+)\]$', expr)
    if ref:
        return _resolve_measure_value(ref.group(1).strip(), pregunta, _depth + 1)

    div = _DIVIDE_RE.match(expr)
    if div:
        a = _resolve_measure_value(div.group(1).strip(), pregunta, _depth + 1)
        b = _resolve_measure_value(div.group(2).strip(), pregunta, _depth + 1)
        if a and b and b[0] != 0:
            return a[0] / b[0], a[1] + [x for x in b[1] if x not in a[1]]
        return None

    calc = _CALC_SUM_FILTER_RE.search(expr)
    if calc:
        agg_table, agg_col, filt_table, filt_col, filt_val = (g.strip() for g in calc.groups())
        try:
            df = dataset.load_table(agg_table)
        except Exception:
            return None
        if filt_col not in df.columns or agg_col not in df.columns:
            return None
        sub = df[df[filt_col].astype(str) == filt_val]
        serie = pd.to_numeric(sub[agg_col], errors="coerce").dropna()
        if serie.empty:
            return None
        return float(serie.sum()), [f'{filt_table}.{filt_col} = "{filt_val}"']

    sumx = _SUMX_VALUES_SUM_RE.search(expr)
    if sumx:
        table, col = sumx.group(1).strip(), sumx.group(2).strip()
        try:
            df = dataset.load_table(table)
        except Exception:
            return None
        if col not in df.columns:
            return None
        filtered, applied = _apply_generic_filters(df, pregunta, table)
        target = filtered if applied else df
        serie = pd.to_numeric(target[col], errors="coerce").dropna()
        if serie.empty:
            return None
        return float(serie.sum()), applied

    simple = _SIMPLE_AGG_RE.match(expr)
    if simple:
        dax_agg, table, col = (g.strip() for g in simple.groups())
        agg = _DAX_AGG_TO_PANDAS.get(dax_agg.lower())
        if not agg:
            return None
        try:
            df = dataset.load_table(table)
        except Exception:
            return None
        if col not in df.columns:
            return None
        filtered, applied = _apply_generic_filters(df, pregunta, table)
        target = filtered if applied else df
        serie = pd.to_numeric(target[col], errors="coerce").dropna()
        if serie.empty:
            return None
        return float(getattr(serie, agg)()), applied

    return None


def _fmt_measure_value(name: str, val: float, applied: list[str]) -> str:
    suffix = f" con {', '.join(applied)}" if applied else ""
    return f"**{name}**{suffix}: **{_fmt_smart(name, val)}**."


def _validation_suffix(col: str, agg: str, val: float, filtered: pd.DataFrame, full: pd.DataFrame) -> str:
    """Corre el Agente Validador; si algun chequeo matematico falla, lo dice
    en vez de entregar la cifra como si nada — transparencia ante todo."""
    result = validator.validate_aggregate(col, agg, val, filtered, full)
    if result["ok"]:
        return ""
    failed = ", ".join(c["nombre"] for c in result["checks"] if not c["ok"])
    return f" ⚠️ (no pasó validación: {failed} — revisa este dato con cautela)"


def _resolve_aggregate(table: str, col: str, agg: str, agg_label: str, pregunta: str, fallback_val: float) -> str:
    """Calcula el agregado, aplicando filtros si la pregunta los menciona.
    Sin filtros detectados, usa el valor precalculado del perfil
    (instantaneo — y sin necesidad de validar: el Validador solo tiene
    sentido para calculos NUEVOS, no para un valor que ya viene del mismo
    perfil que se re-verificaria contra si mismo). Con filtros, el calculo
    SI es nuevo y el Agente Validador lo revisa antes de devolverlo."""
    df = dataset.load_table(table)
    filtered, applied = _apply_generic_filters(df, pregunta, table)
    unmatched = _check_unmatched_entity(pregunta, table, applied)
    if unmatched:
        return unmatched
    if not applied:
        return f"**{col}** ({table}) — {agg_label}: **{_fmt_smart(col, fallback_val)}**."
    if filtered.empty:
        return f"No encontré filas en **{table}** con el filtro detectado ({', '.join(applied)}) — revisa si el valor está escrito igual que en los datos."
    serie = pd.to_numeric(filtered[col], errors="coerce").dropna()
    if serie.empty:
        return f"**{col}** ({table}) con {', '.join(applied)}: no hay valores numéricos en ese filtro."
    val = float(getattr(serie, agg)())
    warn = _validation_suffix(col, agg, val, filtered, df)
    return f"**{col}** ({table}) — {agg_label} con {', '.join(applied)}: **{_fmt_smart(col, val)}**.{warn}"


def try_period_comparison(pregunta: str) -> str | None:
    """Compara una columna numerica entre un periodo EXPLICITO mencionado en
    la pregunta y su periodo anterior REAL (columna Periodo_Anterior de la
    tabla de tiempo) — no adivina cual es "el periodo actual" si no se lo
    dicen (los distintos tipos de periodo — A/B/C/D, V01-V06, ET, ES... —
    no se pueden ordenar cronologicamente por texto de forma confiable, asi
    que mejor pedir el periodo explicito que arriesgarse a comparar mal)."""
    p_lower = pregunta.lower()
    if not (any(k in p_lower for k in COMPARISON_KEYWORDS)
            or re.search(r"periodo\s+anterior|con\s+el\s+anterior", pregunta, re.IGNORECASE)):
        return None
    if not re.search(r"periodo", pregunta, re.IGNORECASE):
        return None  # "compara X con Y" generico no es esto — es especificamente vs. periodo anterior

    explicito = None
    for m in _PERIODO_CODE_RE.finditer(pregunta):
        info = dataset.periodo_info(m.group(0))
        if info:
            explicito = info
            break
    if not explicito or not explicito["anterior"]:
        return None

    candidate_words = [w for w in dataset.tokenize(pregunta) if w.lower() not in _LOOKUP_STOPWORDS]
    scores = _score_numeric_columns(candidate_words)
    if not scores:
        return None
    max_score = max(scores.values())
    winners = [k for k, s in scores.items() if s == max_score]
    if len(winners) != 1:
        return None
    table, col = winners[0]

    # La columna de periodo en la tabla de HECHOS se encuentra via la
    # relacion real hacia la tabla de tiempo (no se asume el nombre) —
    # misma logica que _year_filter_via_relationship, pero para PERIODO.
    periodo_col = None
    tabla_tiempo = dataset._find_periodo_table()
    for rel in dataset.get_relationships(table):
        if rel["ToTableName"] == tabla_tiempo:
            periodo_col = rel["FromColumnName"]
            break
    if not periodo_col:
        return None

    df = dataset.load_table(table)
    if periodo_col not in df.columns:
        return None

    def valor_de(periodo_code):
        sub = df[df[periodo_col].astype(str).str.upper() == periodo_code.upper()]
        if sub.empty:
            return None
        serie = pd.to_numeric(sub[col], errors="coerce").dropna()
        return float(serie.sum()) if not serie.empty else None

    actual, anterior = explicito["periodo"], explicito["anterior"]
    v_actual, v_anterior = valor_de(actual), valor_de(anterior)
    if v_actual is None or v_anterior is None:
        faltante = actual if v_actual is None else anterior
        return (
            f"No encontré datos de **{col}** para **{faltante}** en **{table}** — no puedo comparar "
            f"**{actual}** con su periodo anterior (**{anterior}**) sin ese dato."
        )

    delta = v_actual - v_anterior
    pct = (delta / v_anterior * 100) if v_anterior else None
    signo = "subió" if delta > 0 else "bajó" if delta < 0 else "se mantuvo igual"
    pct_txt = f" ({abs(pct):.1f}%)" if pct is not None else ""
    return (
        f"**{col}** en **{actual}**: {_fmt_smart(col, v_actual)}. "
        f"En **{anterior}** (periodo anterior real, según {tabla_tiempo}): {_fmt_smart(col, v_anterior)}. "
        f"{signo}{pct_txt} respecto al periodo anterior."
    )


def try_direct_answer(pregunta: str) -> str | None:
    if not dataset.is_ready():
        return None
    p = pregunta.strip().lower()

    if re.search(r"cu[aá]nt[ao]s?\s+tablas", p):
        overview = dataset.get_overview()
        return f"El dataset tiene **{len(overview)} tablas**."

    if re.search(r"esquema|(como|cómo).*(relacionan|conectan)|entidad.?relaci[oó]n|modelo de datos", p):
        schema_txt = dataset.describe_schema()
        return schema_txt or "Todavía no hay un esquema calculado — usa \"Actualizar datos\"."

    if "tabla" in p and re.search(r"m[aá]s\s+(grande|filas|registros)", p):
        overview = dataset.get_overview()
        if overview:
            top = overview[0]
            return f"La tabla más grande es **\"{top['table']}\"**, con **{_fmt_direct(top['rows'])} filas**."

    if re.search(r"filas|registros", p):
        overview = dataset.get_overview()
        matches = [t for t in overview if t["table"].lower() in p]
        if len(matches) == 1:
            t = matches[0]
            return f"La tabla **\"{t['table']}\"** tiene **{_fmt_direct(t['rows'])} filas**."

    # Comparacion REAL entre un periodo y su anterior (ej. boton "Comparar
    # periodos") — se intenta ANTES del lookup simple de abajo porque una
    # pregunta como "compara X del periodo Y con el periodo anterior"
    # tambien contiene literal la frase "periodo anterior", y sin este
    # orden el lookup simple se la robaria devolviendo solo el codigo en
    # vez de la comparacion real que se esta pidiendo.
    comparison = try_period_comparison(pregunta)
    if comparison is not None:
        return comparison

    # Preguntas sobre METADATA de un periodo academico especifico (cual es
    # el anterior, si es relevante, que tipo es) — se leen directo de la
    # tabla de tiempo real (dataset.periodo_info), nunca se adivinan con
    # heuristicas de texto sobre el codigo del periodo.
    if re.search(r"periodo\s+anterior|anterior\s+a\b", p):
        for m in _PERIODO_CODE_RE.finditer(pregunta):
            info = dataset.periodo_info(m.group(0))
            if info and info["anterior"]:
                return f"El periodo anterior a **{info['periodo']}** es **{info['anterior']}**."
        return None

    if re.search(r"periodo\s+relevante", p):
        for m in _PERIODO_CODE_RE.finditer(pregunta):
            info = dataset.periodo_info(m.group(0))
            if info:
                return (
                    f"**{info['periodo']}** {'SÍ' if info['relevante'] else 'NO'} es un periodo relevante "
                    f"(tipo **{info['tipo']}**)."
                )
        return None

    if re.search(r"qu[eé]\s+tipo\s+de\s+periodo", p):
        for m in _PERIODO_CODE_RE.finditer(pregunta):
            info = dataset.periodo_info(m.group(0))
            if info:
                return f"**{info['periodo']}** es de tipo **{info['tipo']}** (año {info['año']})."
        return None

    # Si la pregunta trae un verbo de agregacion explicito ("suma",
    # "promedio"...) Y una columna cruda inequivoca lo respalda, esa gana
    # sobre cualquier medida con nombre parecido — ej. "suma de orden neto"
    # debe sumar la columna ORDEN_NETO, no la medida ORDEN_NETO_2024 (un
    # verbo de agregacion es señal de que se quiere calcular algo simple,
    # no consultar el valor puntual de una medida con logica propia).
    agg = agg_label = None
    for pattern, agg_key in _AGG_PATTERNS:
        m = pattern.search(p)
        if m:
            agg, agg_label = agg_key, m.group(0)
            break

    column_hit = None
    if agg:
        candidate_words = [
            w for w in dataset.tokenize(pregunta)
            if w.lower() not in _LOOKUP_STOPWORDS
            and not any(pat.match(w) for pat, _ in _AGG_PATTERNS)
        ]
        scores = _score_numeric_columns(candidate_words)
        if scores:
            max_score = max(scores.values())
            winners = [k for k, s in scores.items() if s == max_score]
            if len(winners) == 1:
                column_hit = winners[0]

    if column_hit:
        table, col = column_hit
        val = dataset._profile()[table]["columns"]
        val = next(c["numeric"][agg] for c in val if c["name"] == col)
        return _resolve_aggregate(table, col, agg, agg_label, pregunta, val)

    # Coincidencia con una medida DAX conocida por nombre (ej. "ingresos
    # proyectados" -> medida "Ingreso_Proyectado", o "cumplimiento del
    # presupuesto" -> una razon entre otras dos medidas) — se intenta antes
    # que la busqueda generica por columna porque es una señal mas especifica.
    measure_name = _match_named_measure(pregunta)
    if measure_name is None:
        # Respaldo: la pregunta no matcheo ninguna medida por nombre, pero
        # menciona un sinonimo de negocio de un grupo de casi-duplicados
        # (ej. "financiado" -> familia RECAUDO_CARTERA) — mismo tratamiento
        # que si hubiera matcheado por nombre, para no dejar sin respuesta
        # justo el caso que la regla de gobernanza pide cubrir.
        measure_name = governance.match_synonym(p)
    if measure_name is None:
        # Ultimo respaldo: palabra generica sola (ej. "recaudo", sin
        # "cartera"/"financiado"/etc. que la harian mas especifica) — el
        # match por nombre normal la rechaza a proposito por ser ambigua
        # entre varias medidas parecidas (RECAUDO_GRAL, RECAUDO_CARTERA,
        # RECAUDO_financiaciones, %Recaudo_sobre_Ingreso, Pendiente_de_
        # recaudo), pero el negocio confirmo cual es la que corresponde en
        # ese caso — no debe quedar sin responder.
        measure_name = governance.default_measure(p)
    if isinstance(measure_name, list):
        # Ambiguedad real entre 2+ medidas (regla de gobernanza): nunca
        # elegir en silencio — se pide la aclaracion puntual en vez de
        # ejecutar una consulta con la interpretacion mas probable.
        opciones = "\", \"".join(measure_name)
        return (
            f"Hay más de una medida que podría responder esto: \"{opciones}\". "
            f"¿A cuál te refieres exactamente?"
        )
    if measure_name:
        blocked = governance.block_message(measure_name)
        if blocked:
            return blocked
        if governance.requires_period(measure_name) and not re.search(r"\b(19|20)\d{2}\b", pregunta):
            # Regla de gobernanza: esta medida no tiene sentido de negocio
            # sin un periodo — no se debe adivinar un "total historico" como
            # si fuera la respuesta.
            return f"**{measure_name}** necesita que especifiques un año o periodo para tener sentido — ¿de cuál te refieres?"
        notes = governance.disclaimers(measure_name)
        resolved = _resolve_measure_value(measure_name, pregunta)
        if resolved is not None:
            val, applied = resolved
            text = _fmt_measure_value(measure_name, val, applied)
            for note in notes:
                text += f"\n\n⚠️ {note}"
            return text
        if notes:
            # No se pudo calcular esta medida (ej. su filtro no encontro
            # ninguna fila) Y tiene una regla de gobernanza asociada — decirlo
            # explicitamente en vez de caer en silencio a una respuesta
            # generica del LLM que no sepa nada de este caso.
            text = (
                f"No encontré ningún dato para **{measure_name}** con la fórmula tal como está "
                f"definida (puede que su filtro no matchee ninguna fila real)."
            )
            for note in notes:
                text += f"\n\n⚠️ {note}"
            return text

    # Si habia agg pero la columna no fue inequivoca (0 o 2+ empatadas) y la
    # medida tampoco resolvio nada: mejor que responda el LLM con contexto,
    # en vez de adivinar.
    return None


# ── 6. Agente Predictivo (modelos matematicos, sin LLM) ─────────────────────
# Para preguntas que NO son una suma/promedio historico sino que piden algo
# que requiere AJUSTAR un modelo: proyeccion (regresion lineal), anomalias
# (z-score) o agrupamientos (k-means). Reutiliza el mismo matching de
# columnas que el resto del motor directo — si hay ambiguedad, no adivina.

_FORECAST_RE = re.compile(r"proyecc|pron[oó]stic|tendencia|predi[cg]|pr[oó]ximo\s+periodo|futuro", re.IGNORECASE)
# Por debajo de este R2 la regresion lineal no explica casi nada de la
# variacion real (practicamente ruido) — se rehusa a proyectar un numero
# puntual en vez de mostrar una cifra que parece precisa pero no lo es.
_FORECAST_MIN_R2 = 0.15
_ANOMALY_RE = re.compile(r"anomal|at[ií]pic|inusual|\braro\b|fuera de lo normal", re.IGNORECASE)
_CLUSTER_RE = re.compile(r"agrupa|cluster|cl[uú]ster|segmenta|patrones|similares|parecidos", re.IGNORECASE)


def _match_numeric_column(pregunta: str) -> tuple[str, str] | None:
    """Encuentra una columna NUMERICA mencionada por nombre en la pregunta —
    sin exigir una palabra de agregacion (a diferencia de try_direct_answer,
    aqui la pregunta pide una proyeccion/anomalia/agrupamiento, no un total).
    Usa el mismo puntaje por partes del nombre que _score_numeric_columns."""
    candidate_words = [w for w in dataset.tokenize(pregunta) if w.lower() not in _LOOKUP_STOPWORDS]
    scores = _score_numeric_columns(candidate_words)
    if not scores:
        return None
    max_score = max(scores.values())
    winners = [k for k, s in scores.items() if s == max_score]
    return winners[0] if len(winners) == 1 else None


def _match_categorical_column(pregunta: str, table: str) -> str | None:
    """Encuentra una columna CATEGORICA de `table` mencionada por nombre en
    la pregunta (para agrupar en anomalias/clustering). "Categorica" aqui
    significa NO numerica y NO fecha — no exige que tenga "top_values"
    cacheados (eso solo se guarda para columnas con <=30 valores distintos;
    una columna como NOM_PROGRAMA con 68 valores sigue siendo perfectamente
    valida para agrupar, el groupby no depende de ese cache)."""
    candidate_words = [w for w in dataset.tokenize(pregunta) if w.lower() not in _LOOKUP_STOPWORDS]
    hits = set()
    for w in candidate_words:
        for c in dataset.find_column(w, limit=8):
            if c["table"] == table and "numeric" not in c and "date_range" not in c:
                hits.add(c["name"])
    return next(iter(hits)) if len(hits) == 1 else None


def try_predictive_answer(pregunta: str) -> str | None:
    if not dataset.is_ready():
        return None

    if _FORECAST_RE.search(pregunta):
        hit = _match_numeric_column(pregunta)
        if not hit:
            return None
        table, col = hit
        r = ml.forecast(table, col)
        if not r:
            return None
        if r["r2"] < _FORECAST_MIN_R2:
            # El ajuste lineal es practicamente ruido (R² asi de bajo = el
            # modelo no explica casi nada de la variacion real) — mostrar un
            # numero puntual igual seria mas enganoso que util, aunque venga
            # con la etiqueta "confianza baja": es facil leer solo la cifra
            # en negrita y saltarse la advertencia. Mejor decir que no hay
            # tendencia confiable, con el ultimo dato real como referencia.
            return (
                f"La serie de **{col}** ({table}) es demasiado irregular entre periodos "
                f"(R²={r['r2']:.2f} sobre {r['n_periodos']} periodos) como para proyectar un número "
                f"confiable con una tendencia lineal simple — preferí no darte una cifra que probablemente "
                f"esté muy lejos de la realidad. El último periodo real (**{r['ultimo_periodo']}**) cerró en "
                f"**{_fmt_smart(col, r['ultimo_valor'])}**."
            )
        confianza = "alta" if r["r2"] >= 0.7 else "media"
        return (
            f"**Proyección de {col}** ({table}, regresión lineal sobre {r['n_periodos']} periodos): "
            f"el último periodo real fue **{r['ultimo_periodo']}** con **{_fmt_smart(col, r['ultimo_valor'])}**; "
            f"la tendencia proyecta **{_fmt_smart(col, r['proyeccion'])}** para el siguiente periodo "
            f"(confianza {confianza}, R²={r['r2']:.2f})."
        )

    if _ANOMALY_RE.search(pregunta):
        hit = _match_numeric_column(pregunta)
        if not hit:
            return None
        table, col = hit
        group_col = _match_categorical_column(pregunta, table)
        if not group_col:
            return None
        outliers = ml.detect_anomalies(table, col, group_col)
        if outliers is None:
            return None
        if not outliers:
            return f"No encontré valores atípicos en **{col}** por **{group_col}** — todo dentro de lo esperado (±2 desviaciones estándar)."
        lines = "\n".join(
            f"- {o['grupo']}: {_fmt_smart(col, o['valor'])} "
            f"({'muy por encima' if o['tipo'] == 'alto' else 'muy por debajo'} del promedio, z={o['z_score']:.1f})"
            for o in outliers[:5]
        )
        return f"**Valores atípicos en {col} por {group_col}** (z-score, umbral 2σ):\n{lines}"

    if _CLUSTER_RE.search(pregunta):
        hit = _match_numeric_column(pregunta)
        if not hit:
            return None
        table, col = hit
        group_col = _match_categorical_column(pregunta, table)
        if not group_col:
            return None
        result = ml.cluster_groups(table, group_col, [col], k=3)
        if not result:
            return None
        lines = "\n".join(
            f"- Grupo {g}: {', '.join(sorted(members)[:8])}" + (" ..." if len(members) > 8 else "")
            for g, members in sorted(result["grupos"].items())
        )
        return f"**Agrupamiento de {group_col} por {col}** (k-means, k={result['k']}):\n{lines}"

    return None


# ── 2. Agente Investigador ──────────────────────────────────────────────────

def gather_context(pregunta: str) -> tuple[str, list[str]]:
    """Arma el contexto de datos: esquema entidad-relacion + resumen de
    tablas + snippets puntuales de la pregunta. El esquema va primero para
    que el modelo entienda la ESTRUCTURA (que es hechos, que es dimension,
    como se conectan) antes de ver las cifras sueltas — reduce que confunda
    una tabla de metas/proyecciones con una de ejecucion real."""
    if not dataset.is_ready():
        return "", []
    schema_txt = dataset.describe_schema()
    ctx = dataset.build_context_text()
    ctx = f"{schema_txt}\n\n{ctx}" if schema_txt else ctx
    snippets = dataset.build_relevant_snippets(pregunta)
    return ctx, snippets


# ── 3. Agente Redactor ───────────────────────────────────────────────────────

_RESOLUCION_CIFRA = """- Si el contexto de abajo trae el dato que piden, la respuesta SIEMPRE debe
  llegar a una resolución clara con la cifra real en negrita (valor o %) —
  nunca te quedes solo explicando el camino sin dar el número, y NUNCA
  escribas la palabra "cifra" o un marcador entre corchetes en vez del
  número real — si no tienes el número real, no lo simules, usa la regla de
  abajo para decir que no lo tienes. Usa el nombre EXACTO de la columna o
  medida tal como aparece en el contexto, nunca lo traduzcas, renombres ni
  le agregues calificativos que no estén ahí (ej. si la columna se llama
  "INGRESO NETO" de la tabla "Metas_2025", no le digas "Ejecutado" ni "Real"
  si el contexto no lo dice explícitamente — puede ser una meta o
  proyección, no lo asumas).
- Nada de rodeos tipo "para calcular esto necesitaríamos..." si el dato ya
  está disponible en el contexto — ve directo a la respuesta."""

_RESOLUCION_GUIA = """- Esta pregunta pide ORIENTACIÓN, no una cifra puntual — responde en
  prosa, con 2-3 sugerencias concretas basadas en las tablas/medidas reales
  del contexto (nómbralas). No inventes ni fuerces un número si no viene al
  caso; solo menciona una cifra si de verdad ayuda a ilustrar el punto y
  está respaldada por el contexto."""

ANALYST_PROMPT = """Eres el copiloto de un tablero de Power BI, embebido junto a este chat.
Le hablas a alguien de tu equipo: cercano, cálido y directo — nunca robótico
ni excesivamente formal. Respondes en español.

Formato obligatorio:
- Respuestas CORTAS: 2-4 líneas, salvo que te pidan explícitamente un reporte
  o una explicación a fondo.
{resolucion}
- NUNCA muestres fórmulas DAX, código, ni nombres técnicos internos
  (CALCULATE, SUMX, VAR, nombres de medidas con guion bajo, etc.) al
  usuario — eso no es una respuesta, es un problema sin resolver que le
  estás pasando a él. Tu trabajo es dar la cifra o decir en una frase que no
  la tienes, no explicar cómo se calcularía.
- Si no encuentras el dato exacto pero el contexto trae algo relacionado
  (una cifra parecida, de otra tabla o con otro nombre), ofrécela igual
  aclarando qué es exactamente — es más útil que un "no lo tengo" seco.
- Si una medida del contexto está marcada "BLOQUEADA", no des ningún valor
  para ella — responde solo con el motivo indicado. Si un fragmento trae
  "[AVISO OBLIGATORIO, incluir en la respuesta: ...]", ese texto debe
  aparecer sí o sí en tu respuesta si usas esa medida — es una regla de
  gobernanza de negocio, no una sugerencia.

{data_note}

Regla dura, sin excepciones (la información es financiera): NUNCA inventes
una cifra que no esté en el contexto de arriba. Si de verdad no hay nada
relacionado en el contexto, dilo en UNA frase simple ("no tengo esa cifra a
la mano") sin tecnicismos ni fórmulas — y ahí sí termina la respuesta, sin
alargarla. Las cifras de este tablero son pesos colombianos (COP) — para
cifras monetarias (ingresos, recaudo, órdenes, valores, cuotas, créditos,
cartera) usa SIEMPRE el formato colombiano exacto: símbolo "$", punto para
miles, coma para decimales (ej. "$35.230.896.108,07"), nunca el formato en
inglés ($35,230,896,108.07) ni sin símbolo. Para cifras que NO son dinero
(conteos de estudiantes, tablas, matrículas) no uses "$", solo el número con
puntos de miles."""

CORRECTOR_PROMPT = """Eres el mismo copiloto, revisando tu propia respuesta anterior antes de
enviarla. El verificador automático encontró cifras que NO aparecen en el
contexto de datos — probablemente las inventaste o las calculaste sin base.

{data_note}

Tu respuesta anterior fue:
\"\"\"{draft}\"\"\"

Las siguientes cifras no se pudieron confirmar en el contexto: {flagged}

Reescribe la respuesta completa: quita o corrige esas cifras (di que no
tienes el dato exacto en vez de aproximar), mantén el resto igual y el mismo
tono cercano. Responde solo con la respuesta final, sin explicar el cambio."""


def draft_answer(pregunta: str, ctx: str, snippets: list[str], intent: str = "general") -> str:
    resolucion = _RESOLUCION_GUIA if intent == "explicacion" else _RESOLUCION_CIFRA
    return llm.chat([
        {"role": "system", "content": ANALYST_PROMPT.format(
            data_note=_format_context(ctx, snippets), resolucion=resolucion,
        )},
        {"role": "user", "content": pregunta},
    ], max_tokens=280)


def _format_context(ctx: str, snippets: list[str]) -> str:
    text = ctx
    if snippets:
        text += "\n\nDetalle relacionado con la pregunta:\n" + "\n".join(f"- {s}" for s in snippets)
    if not text:
        text = (
            "Todavía no hay datos conectados (corre 'python -m app.pbix_loader' o usa "
            "el botón Actualizar datos): si preguntan cifras exactas, dilo con naturalidad "
            "en vez de inventar un número."
        )
    return text


# ── 4. Agente Verificador (determinista) ────────────────────────────────────

_NUMBER_RE = re.compile(r"\d[\d.,]{2,}\d|\d+")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def _normalize_numbers(text: str) -> set[float]:
    found = set()
    for raw in _NUMBER_RE.findall(text):
        if _YEAR_RE.match(raw):
            continue  # los años (2026, etc.) no cuentan como "cifra financiera"
        digits = re.sub(r"[.,]", "", raw)
        if digits.isdigit() and len(digits) >= 3:  # ignora numeros chicos (bullets, "3 hallazgos")
            found.add(float(digits))
    return found


def verify_numbers(draft: str, ctx: str, snippets: list[str]) -> list[float]:
    """Devuelve las cifras del borrador que NO aparecen (ni aproximadas) en el contexto."""
    context_numbers = _normalize_numbers(ctx + "\n" + "\n".join(snippets))
    draft_numbers = _normalize_numbers(draft)
    unverified = []
    for n in draft_numbers:
        if any(abs(n - c) <= max(1, c * 0.005) for c in context_numbers):
            continue
        unverified.append(n)
    return unverified


# ── 5. Agente Corrector (solo si hace falta) ────────────────────────────────

def correct_answer(draft: str, unverified: list[float], ctx: str, snippets: list[str]) -> str:
    flagged = ", ".join(f"{n:,.0f}".replace(",", ".") for n in unverified)
    return llm.chat([
        {"role": "system", "content": CORRECTOR_PROMPT.format(
            data_note=_format_context(ctx, snippets), draft=draft, flagged=flagged,
        )},
        {"role": "user", "content": "Corrige tu respuesta anterior."},
    ], temperature=0.1)


# ── Orquestador ──────────────────────────────────────────────────────────────

GREETING_REPLIES = [
    "¡Hola! Aquí estoy, listo para lo que necesites del tablero — pregúntame por una cifra, pídeme que te explique algo, o dime que te arme un reporte corto.",
]


# ── 8. Agente de Consistencia (determinista) ────────────────────────────────
# Cachea la respuesta final por (pregunta normalizada, version de los datos)
# — la MISMA pregunta, mientras los datos no cambien, devuelve SIEMPRE la
# misma respuesta exacta, sin volver a pasar por el LLM (mas rapido tambien:
# la segunda vez es instantaneo). Se complementa con temperature baja + seed
# fijo en llm.chat() — aquella reduce que el modelo varie, esto GARANTIZA
# que no varie para una pregunta ya respondida. El cache se invalida solo:
# la clave incluye dataset.data_version(), que cambia con cada "Actualizar
# datos" (nuevo manifest.json), asi nunca sirve una respuesta vieja de datos
# que ya no existen.
_ANSWER_CACHE: dict[tuple[str, float], dict] = {}
_ANSWER_CACHE_MAX = 500


def _cache_key(pregunta: str) -> tuple[str, float]:
    norm = re.sub(r"\s+", " ", pregunta.strip().lower())
    return norm, dataset.data_version()


def clear_answer_cache():
    """Vacia el cache de respuestas — no hace falta llamarlo despues de un
    /api/refresh (la clave ya cambia sola con dataset.data_version()), pero
    evita que queden dando vueltas en memoria respuestas de una version de
    datos que ya nadie va a volver a pedir."""
    _ANSWER_CACHE.clear()


def run_pipeline(pregunta: str) -> dict:
    """Corre el pipeline completo (con cache de consistencia). Devuelve
    {text, verified, intent, corrected, cached}."""
    key = _cache_key(pregunta)
    cached = _ANSWER_CACHE.get(key)
    if cached is not None:
        return {**cached, "cached": True}
    result = _run_pipeline(pregunta)
    if len(_ANSWER_CACHE) >= _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.clear()  # tope simple: a este tamaño no vale la pena un LRU real
    _ANSWER_CACHE[key] = result
    return {**result, "cached": False}


def _run_pipeline(pregunta: str) -> dict:
    """Corre el pipeline completo. Devuelve {text, verified, intent, corrected}."""
    intent = classify_intent(pregunta)

    if intent == "saludo":
        return {"text": GREETING_REPLIES[0], "verified": True, "intent": intent, "corrected": False}

    direct = try_direct_answer(pregunta)
    if direct is not None:
        return {"text": direct, "verified": True, "intent": "directo", "corrected": False}

    predictive = try_predictive_answer(pregunta)
    if predictive is not None:
        return {"text": predictive, "verified": True, "intent": "predictivo", "corrected": False}

    ctx, snippets = gather_context(pregunta)
    draft = draft_answer(pregunta, ctx, snippets, intent)

    if not ctx:
        # sin datos conectados: no hay nada que verificar contra
        return {"text": draft, "verified": True, "intent": intent, "corrected": False}

    unverified = verify_numbers(draft, ctx, snippets)
    if not unverified:
        return {"text": draft, "verified": True, "intent": intent, "corrected": False}

    corrected = correct_answer(draft, unverified, ctx, snippets)
    return {"text": corrected, "verified": True, "intent": intent, "corrected": True}


def run_report_pipeline(prompt: str) -> dict:
    """Igual que run_pipeline pero para el boton 'Reporte corto' (sin
    clasificar intencion) — tambien pasa por el Agente de Consistencia: el
    prompt es siempre el mismo texto fijo, asi que sin cache cada clic
    volveria a generar un reporte ligeramente distinto por el muestreo del
    LLM, aunque los datos no hayan cambiado en absoluto."""
    key = _cache_key("__reporte_corto__" + prompt)
    cached = _ANSWER_CACHE.get(key)
    if cached is not None:
        return {**cached, "cached": True}
    result = _run_report_pipeline(prompt)
    if len(_ANSWER_CACHE) >= _ANSWER_CACHE_MAX:
        _ANSWER_CACHE.clear()
    _ANSWER_CACHE[key] = result
    return {**result, "cached": False}


def _run_report_pipeline(prompt: str) -> dict:
    ctx, snippets = gather_context(prompt)
    draft = draft_answer(prompt, ctx, snippets)
    if not ctx:
        return {"text": draft, "verified": True, "corrected": False}
    unverified = verify_numbers(draft, ctx, snippets)
    if not unverified:
        return {"text": draft, "verified": True, "corrected": False}
    corrected = correct_answer(draft, unverified, ctx, snippets)
    return {"text": corrected, "verified": True, "corrected": True}
