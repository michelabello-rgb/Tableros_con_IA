"""
Modelos matemáticos/estadísticos para preguntas que el motor de respuesta
directa NO puede resolver con una simple suma/promedio histórico, porque
piden algo que requiere AJUSTAR un modelo:

  - forecast()          regresión lineal  -> proyección al siguiente periodo
  - detect_anomalies()  z-score           -> valores atípicos por grupo
  - cluster_groups()    k-means (propio)  -> agrupa entidades similares

Genérico: no asume el esquema de un reporte específico. Reutiliza las
relaciones reales del modelo (igual que agents.py) para ordenar la serie de
tiempo cronológicamente en vez de adivinar por el formato de un código.
Solo numpy/pandas — sin dependencias nuevas.
"""
import re

import numpy as np
import pandas as pd

from . import dataset

_YEAR_COL_RE = re.compile(r"^a[nñ]o$|^year$", re.IGNORECASE)


def _time_ordered_series(table: str, value_col: str) -> pd.DataFrame | None:
    """[periodo, año, valor] agregado y ordenado cronológicamente, usando la
    relación real del modelo hacia una dimensión de tiempo con columna
    "Año" — misma lógica que el filtro de año de agents.py."""
    df = dataset.load_table(table)
    if value_col not in df.columns:
        return None
    for rel in dataset.get_relationships(table):
        to_table, to_col, from_col = rel["ToTableName"], rel["ToColumnName"], rel["FromColumnName"]
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
        tmp = df[[from_col, value_col]].copy()
        tmp["__anio"] = tmp[from_col].map(mapping)
        tmp = tmp.dropna(subset=["__anio"])
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        agg = tmp.groupby([from_col, "__anio"], as_index=False)[value_col].sum()
        agg = agg.sort_values(["__anio", from_col]).rename(columns={from_col: "periodo"})
        return agg[["periodo", "__anio", value_col]].reset_index(drop=True)
    return None


def forecast(table: str, value_col: str, n_periods: int = 1) -> dict | None:
    """Regresión lineal (numpy.polyfit) sobre la serie histórica por
    periodo. Devuelve la proyección + el R² (que tan bien se ajusta la
    tendencia) para poder reportar la confiabilidad, no solo el número."""
    serie = _time_ordered_series(table, value_col)
    if serie is None or len(serie) < 4:
        return None
    y = serie[value_col].to_numpy(dtype=float)
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    y_pred = trend(x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    future_y = trend(np.arange(len(y), len(y) + n_periods))
    return {
        "ultimo_periodo": str(serie["periodo"].iloc[-1]),
        "ultimo_valor": float(y[-1]),
        "proyeccion": float(future_y[-1]),
        "r2": float(max(0.0, min(1.0, r2))),
        "n_periodos": len(y),
    }


def detect_anomalies(table: str, value_col: str, group_col: str, z_threshold: float = 2.0) -> list | None:
    """Agrupa por group_col, suma value_col, y marca los grupos cuyo
    z-score (desviaciones estándar respecto al promedio del grupo) supera
    z_threshold — estadística simple, sin caja negra."""
    df = dataset.load_table(table)
    if value_col not in df.columns or group_col not in df.columns:
        return None
    tmp = df[[group_col, value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    agg = tmp.groupby(group_col)[value_col].sum().dropna()
    if len(agg) < 3:
        return []
    mean, std = agg.mean(), agg.std()
    if not std:
        return []
    z = (agg - mean) / std
    outliers = z[z.abs() >= z_threshold].sort_values(key=abs, ascending=False)
    return [
        {"grupo": str(k), "valor": float(agg[k]), "z_score": float(v), "tipo": "alto" if v > 0 else "bajo"}
        for k, v in outliers.items()
    ]


def _kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = 42):
    """K-means minimo (sin sklearn) — inicializacion aleatoria + Lloyd's."""
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.full(len(X), -1)
    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centers[c] = X[labels == c].mean(axis=0)
    return labels, centers


def cluster_groups(table: str, group_col: str, metric_cols: list, k: int = 3) -> dict | None:
    """K-means sobre metric_cols agregados por group_col — agrupa entidades
    similares (ej. programas, regionales) por desempeño combinado."""
    df = dataset.load_table(table)
    cols = [c for c in metric_cols if c in df.columns]
    if group_col not in df.columns or not cols:
        return None
    tmp = df[[group_col] + cols].copy()
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    agg = tmp.groupby(group_col)[cols].sum().dropna()
    if len(agg) < k + 1:
        return None
    X = agg.to_numpy(dtype=float)
    rng = X.max(axis=0) - X.min(axis=0)
    X_norm = (X - X.min(axis=0)) / np.where(rng == 0, 1, rng)  # normaliza 0-1 por metrica
    labels, _ = _kmeans(X_norm, k)
    grupos = {}
    for i, name in enumerate(agg.index):
        grupos.setdefault(int(labels[i]), []).append(str(name))
    return {"k": k, "grupos": grupos, "metricas": cols}
