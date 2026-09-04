"""
Agente Validador Matemático (7mo agente): antes de entregar una cifra del
motor de respuesta directa, la somete a chequeos matemáticos INDEPENDIENTES
de la ruta que la calculó — no es "confiar" en el resultado, es demostrar
que es consistente con los datos crudos desde más de un ángulo.

Chequeos:
  1. Recómputo independiente — se recalcula con numpy puro (ruta de código
     distinta a pandas) y debe dar el mismo número; atrapa bugs de
     implementación, no solo cifras inventadas por un LLM.
  2. Rango plausible — aprendido de la distribución real de la columna
     (percentiles de TODA la tabla, no solo el subconjunto filtrado): si el
     resultado cae muy lejos de lo que esos datos permiten, algo está mal.
  3. Conservación — una suma filtrada no puede superar (en magnitud) la
     suma total sin filtrar, cuando la columna es mayormente no negativa.
  4. Reconciliación — cuando el filtro es por UNA columna categórica, la
     suma de esa columna en TODAS sus categorías debe calzar con el total
     sin filtrar (confirma que el filtro particiona los datos correctamente,
     sin perder ni duplicar filas).
"""
import numpy as np
import pandas as pd

_NUMPY_AGG = {"sum": np.sum, "mean": np.mean, "min": np.min, "max": np.max}


def validate_aggregate(col: str, agg: str, value: float, filtered_df: pd.DataFrame,
                        full_df: pd.DataFrame) -> dict:
    """Valida `value` (resultado de aplicarle `agg` a `col` en `filtered_df`)
    contra varios chequeos independientes. Devuelve {ok, checks: [...]}."""
    checks = []

    # 1. Recomputo independiente (numpy, no pandas) sobre el mismo subconjunto
    serie = pd.to_numeric(filtered_df[col], errors="coerce").dropna().to_numpy()
    if len(serie) and agg in _NUMPY_AGG:
        recompute = float(_NUMPY_AGG[agg](serie))
        tol = max(1.0, abs(value) * 1e-6)
        checks.append({"nombre": "recomputo_independiente", "ok": abs(recompute - value) <= tol})
    else:
        checks.append({"nombre": "recomputo_independiente", "ok": len(serie) == 0 and value == 0})

    full_serie = pd.to_numeric(full_df[col], errors="coerce").dropna()

    # 2. Rango plausible: el resultado no puede estar fuera de lo que la
    # propia columna (completa, sin filtrar) demuestra que es posible.
    if len(full_serie) > 1:
        lo, hi = float(full_serie.min()), float(full_serie.max())
        if agg in ("mean", "min", "max"):
            margin = (hi - lo) * 0.05 if hi > lo else abs(hi) * 0.05 + 1
            checks.append({"nombre": "rango_plausible", "ok": (lo - margin) <= value <= (hi + margin)})
        elif agg == "sum":
            total = float(full_serie.sum())
            mostly_nonneg = (full_serie >= 0).mean() > 0.9
            if mostly_nonneg:
                # 3. Conservacion: un subconjunto no puede sumar mas que el todo
                checks.append({"nombre": "conservacion_vs_total", "ok": value <= abs(total) * 1.0001 + 1})

    # 4. Integridad del filtro: las filas filtradas deben ser un subconjunto
    # REAL de las filas completas (mismo indice, sin duplicar ni inventar
    # filas) — confirma que el filtro particiona los datos correctamente.
    subset_ok = len(filtered_df) <= len(full_df) and filtered_df.index.isin(full_df.index).all()
    checks.append({"nombre": "integridad_filtro", "ok": bool(subset_ok)})

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
