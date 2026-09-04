"""
Extrae tablas, medidas DAX y un perfil estadistico de CUALQUIER .pbix que
pongas en tablero/ (no depende de nombres de columnas de un reporte
especifico). Guarda todo en data/pbix/ para que el resto de la app no tenga
que volver a parsear el archivo (puede pesar decenas de MB) en cada arranque.

Se corre una vez, o cada vez que quieras reflejar un .pbix actualizado:

    python -m app.pbix_loader

(la app tambien expone un boton "Actualizar datos" que dispara esto mismo
via POST /api/refresh, sin usar la terminal).
"""
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from pbixray import PBIXRay

from .config import BASE_DIR

TABLERO_DIR = BASE_DIR / "tablero"
CACHE_DIR = BASE_DIR / "data" / "pbix"

PII_COLUMN_PATTERN = re.compile(r"usuario|correo|email|vendedor|asesor", re.IGNORECASE)

MAX_TOP_VALUES_DISTINCT = 30  # columnas con <= N valores distintos se tratan como categoricas


def _find_pbix() -> Path:
    matches = sorted(TABLERO_DIR.glob("*.pbix"))
    if not matches:
        raise FileNotFoundError(f"No se encontro ningun .pbix en {TABLERO_DIR}")
    return matches[0]


def _profile_table(df: pd.DataFrame) -> dict:
    """Perfil estadistico generico de una tabla: no asume ningun esquema."""
    columns = []
    for col in df.columns:
        s = df[col]
        entry = {
            "name": str(col),
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "distinct": int(s.nunique(dropna=True)),
        }
        numeric_s, is_numeric = s, pd.api.types.is_numeric_dtype(s)
        if not is_numeric and entry["non_null"]:
            # pbixray a veces trae columnas numericas (ej. decimal.Decimal)
            # como dtype "object" — sin este intento se pierden silenciosamente
            # de las estadisticas y del contexto del chat.
            coerced = pd.to_numeric(s, errors="coerce")
            if coerced.notna().sum() >= entry["non_null"] * 0.95:
                numeric_s, is_numeric = coerced, True

        if is_numeric and entry["non_null"]:
            clean = numeric_s.dropna()
            entry["numeric"] = {
                "sum": float(clean.sum()),
                "mean": float(clean.mean()),
                "min": float(clean.min()),
                "max": float(clean.max()),
            }
        elif pd.api.types.is_datetime64_any_dtype(s) and entry["non_null"]:
            entry["date_range"] = {"min": str(s.min()), "max": str(s.max())}
        elif entry["distinct"] and entry["distinct"] <= MAX_TOP_VALUES_DISTINCT:
            vc = s.value_counts(dropna=True).head(10)
            entry["top_values"] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        columns.append(entry)
    return {"rows": len(df), "columns": columns}


def _build_schema(manifest: dict, relationships: list) -> dict:
    """Capa intermedia de esquema entidad-relacion: clasifica cada tabla como
    'hechos' (transacciones, muchas filas) o 'dimension' (catalogo/lookup,
    referenciada por otras) segun su rol en las relaciones del modelo, y arma
    el grafo de conexiones — asi los agentes entienden la ESTRUCTURA completa
    del modelo (que tabla se conecta con cual, y por que columna) en vez de
    ver cada tabla como una isla suelta. Se calcula una vez aqui, no en cada
    pregunta, para que el chat sea rapido y coherente entre preguntas."""
    out_degree, in_degree = {}, {}
    for r in relationships:
        out_degree[r["FromTableName"]] = out_degree.get(r["FromTableName"], 0) + 1
        in_degree[r["ToTableName"]] = in_degree.get(r["ToTableName"], 0) + 1

    schema = {}
    for name, info in manifest["tables"].items():
        out_d, in_d = out_degree.get(name, 0), in_degree.get(name, 0)
        if out_d > 0:
            role = "hechos"
        elif in_d > 0:
            role = "dimension"
        else:
            role = "aislada"
        relaciones = [
            {"columna": r["FromColumnName"], "tabla_relacionada": r["ToTableName"], "columna_relacionada": r["ToColumnName"]}
            for r in relationships if r["FromTableName"] == name
        ] + [
            {"columna": r["ToColumnName"], "tabla_relacionada": r["FromTableName"], "columna_relacionada": r["FromColumnName"]}
            for r in relationships if r["ToTableName"] == name
        ]
        schema[name] = {"rows": info["rows"], "role": role, "relaciones": relaciones}
    return schema


def _extract_field_refs(node, refs: set):
    """Recorre un JSON de visual/pagina/filtro buscando el patron
    {"Expression": {"SourceRef": {"Entity": X}}, "Property": Y} — asi es
    como Power BI referencia un campo (columna o medida) puesto en un
    visual, sin importar el tipo de visual o donde este anidado."""
    if isinstance(node, dict):
        expr = node.get("Expression")
        prop = node.get("Property")
        if isinstance(expr, dict) and isinstance(expr.get("SourceRef"), dict) and prop:
            entity = expr["SourceRef"].get("Entity")
            if entity:
                refs.add((entity, prop))
        for v in node.values():
            _extract_field_refs(v, refs)
    elif isinstance(node, list):
        for item in node:
            _extract_field_refs(item, refs)


def _build_usage(pbix_path: Path, manifest: dict, measures_records: list) -> dict:
    """Detecta que columnas/medidas estan REALMENTE puestas en algun visual
    del reporte (no solo presentes en el modelo de datos) — leyendo los
    JSON de definicion de cada visual/pagina/filtro dentro del .pbix.
    Un modelo suele acumular medidas y tablas viejas que ya nadie usa; esto
    le permite a los agentes preferir lo que SI esta en pantalla cuando hay
    ambiguedad entre varios nombres parecidos."""
    refs = set()
    try:
        with zipfile.ZipFile(pbix_path) as z:
            names = [n for n in z.namelist() if n.startswith("Report/definition/") and n.endswith(".json")]
            for n in names:
                try:
                    data = json.loads(z.read(n).decode("utf-8"))
                except Exception:
                    continue
                _extract_field_refs(data, refs)
    except Exception:
        return {"used_columns": [], "used_measures": []}

    table_names = set(manifest["tables"].keys())
    measure_keys = {(m["TableName"], m["Name"]) for m in measures_records}

    used_columns = [{"table": e, "column": p} for e, p in refs if e in table_names]
    used_measures = [{"table": e, "name": p} for e, p in refs if (e, p) in measure_keys]
    return {"used_columns": used_columns, "used_measures": used_measures}


def extract():
    pbix_path = _find_pbix()
    print(f"Leyendo modelo de {pbix_path.name} ...")
    model = PBIXRay(str(pbix_path))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"source_file": pbix_path.name, "tables": {}}
    profile = {}

    for table in model.tables:
        df = model.get_table(table)
        drop = [c for c in df.columns if PII_COLUMN_PATTERN.search(str(c))]
        if drop:
            df = df.drop(columns=drop)

        safe_name = re.sub(r"[^\w.-]", "_", table)
        out_path = CACHE_DIR / f"{safe_name}.parquet"
        df.to_parquet(out_path, index=False)
        manifest["tables"][table] = {"file": out_path.name, "rows": len(df), "columns": list(df.columns)}
        profile[table] = _profile_table(df)
        print(f"  {table}: {len(df):,} filas -> {out_path.name}")

    measures = model.dax_measures[["TableName", "Name", "Expression", "DisplayFolder"]].fillna("")
    measures_records = measures.to_dict(orient="records")
    (CACHE_DIR / "measures.json").write_text(
        json.dumps(measures_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {len(measures_records)} medidas DAX -> measures.json")

    # Relaciones del modelo (ej. Ventas_Recibos_contact.PERIODO -> Dimension_
    # tiempo_periodo.PERIODO) — permiten filtrar por dimensiones reales (año,
    # region, etc.) en vez de adivinar por el nombre/formato de una columna.
    rel_cols = ["FromTableName", "FromColumnName", "ToTableName", "ToColumnName"]
    try:
        rel_df = model.relationships[rel_cols]
        relationships = rel_df.to_dict(orient="records")
    except Exception:
        relationships = []
    (CACHE_DIR / "relationships.json").write_text(
        json.dumps(relationships, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {len(relationships)} relaciones -> relationships.json")

    # Que columnas/medidas estan REALMENTE en algun visual del reporte (no
    # solo presentes en el modelo) — para desempatar cuando hay nombres
    # parecidos y preferir lo que de verdad se usa sobre lo abandonado.
    usage = _build_usage(pbix_path, manifest, measures_records)
    (CACHE_DIR / "usage.json").write_text(
        json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {len(usage['used_measures'])} medidas y {len(usage['used_columns'])} columnas en uso real -> usage.json")

    schema = _build_schema(manifest, relationships)
    (CACHE_DIR / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_fact = sum(1 for s in schema.values() if s["role"] == "hechos")
    n_dim = sum(1 for s in schema.values() if s["role"] == "dimension")
    print(f"  esquema: {n_fact} tabla(s) de hechos, {n_dim} de dimension -> schema.json")

    (CACHE_DIR / "profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (CACHE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Listo.")
    return manifest


if __name__ == "__main__":
    extract()
