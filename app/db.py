"""
Acceso a la base de autenticacion (SQL Server, esquema COE — requiere las
tablas COE.User_Admin y COE.Acceso_Log ya creadas en el servidor). Separado
de dataset.py a proposito: dataset.py es el dataset
del tablero (parquet local, generico, sin credenciales); esto es login e
infraestructura, con credenciales reales de un servidor de la empresa.

Si DB_SERVER/DB_DATABASE/DB_UID/DB_PWD no estan configurados (.env), el
login queda deshabilitado y lo dice explicitamente en vez de fallar con un
error críptico de conexion — asi el resto de la app se puede seguir
probando en local sin acceso al servidor real.
"""
import bcrypt

from . import config

try:
    import pyodbc
except ImportError:
    pyodbc = None


def _driver() -> str:
    """Usa el driver ODBC de SQL Server mas nuevo instalado en el sistema."""
    candidatos = sorted((d for d in pyodbc.drivers() if "SQL Server" in d), reverse=True)
    if not candidatos:
        raise RuntimeError(
            "No hay ningun driver ODBC de SQL Server instalado en este servidor "
            "(msodbcsql17/18) — el login no puede conectarse a la base."
        )
    return candidatos[0]


def get_connection():
    """Conexion nueva a la base — no se reusa entre pedidos (pyodbc no es
    thread-safe si se comparte la misma conexion entre requests concurrentes;
    el login es poco frecuente, abrir una conexion por intento es barato)."""
    if pyodbc is None:
        raise RuntimeError("Falta el paquete pyodbc (pip install pyodbc).")
    if not config.DB_CONFIGURED:
        raise RuntimeError(
            "El login no esta configurado — faltan DB_SERVER/DB_DATABASE/DB_UID/DB_PWD en .env."
        )
    conn_str = (
        f"DRIVER={{{_driver()}}};"
        f"SERVER={config.DB_SERVER};"
        f"DATABASE={config.DB_DATABASE};"
        f"UID={config.DB_UID};"
        f"PWD={config.DB_PWD};"
        f"TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, timeout=8)


# Hash "de relleno" (de una contraseña que nadie usa) para comparar contra
# el cuando el username no existe — sin esto, un usuario inexistente vuelve
# MAS RAPIDO que uno real (que si llega a bcrypt.checkpw, deliberadamente
# lento), y esa diferencia de tiempo es suficiente para que alguien externo
# adivine que usernames existen probando uno por uno, sin ver ningun mensaje
# de error distinto.
_DUMMY_HASH = bcrypt.hashpw(b"relleno-sin-usar-nunca", bcrypt.gensalt()).decode("utf-8")


def verify_login(username: str, password: str) -> dict | None:
    """Busca el usuario ACTIVO por Username y compara la contraseña contra
    el hash bcrypt guardado — nunca compara texto plano. Devuelve el
    usuario (sin el hash) si es valido, None si no (usuario inexistente,
    inactivo, o contraseña incorrecta — a proposito no se distingue cual de
    los tres en la respuesta, para no darle pistas a quien intenta adivinar).
    Siempre corre un bcrypt.checkpw, exista o no el usuario (ver _DUMMY_HASH),
    para que el tiempo de respuesta tampoco delate si el username es real."""
    username = (username or "").strip()
    if not username or not password:
        return None
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT Id, Username, Nombre_Completo, Rol, Password_Hash "
            "FROM COE.User_Admin WHERE Username = ? AND Activo = 1",
            username,
        )
        row = cur.fetchone()
        if not row:
            bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH.encode("utf-8"))
            return None
        user_id, uname, nombre, rol, pwd_hash = row
        if not bcrypt.checkpw(password.encode("utf-8"), pwd_hash.encode("utf-8")):
            return None
        cur.execute("UPDATE COE.User_Admin SET Ultimo_Acceso = SYSDATETIME() WHERE Id = ?", user_id)
        conn.commit()
        return {"id": user_id, "username": uname, "nombre": nombre, "rol": rol}
    finally:
        conn.close()


# ── Administracion de usuarios (panel /admin/usuarios, solo rol admin) ──────

class UserError(Exception):
    """Error esperado (ej. username duplicado) — se muestra tal cual al
    admin en el panel, a diferencia de un error de conexion inesperado."""


def _fmt_fecha(v) -> str | None:
    """Algunos drivers ODBC (el generico 'SQL Server' de Windows, mas viejo
    que msodbcsql17/18) devuelven DATETIME2 como texto en vez de un objeto
    datetime real — formatear aqui, en Python, en vez de llamar .strftime()
    directo en la plantilla Jinja2, evita que eso reviente el panel entero
    con un 500 sin importar que driver este instalado en cada PC."""
    if v is None:
        return None
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d %H:%M")
    return str(v)[:16]  # ya viene como texto ("2026-09-16 10:30:00.0" -> recorta a minutos)


def list_users() -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT Id, Username, Nombre_Completo, Correo, Rol, Activo, Fecha_Creacion, Ultimo_Acceso "
            "FROM COE.User_Admin ORDER BY Nombre_Completo"
        )
        cols = [c[0] for c in cur.description]
        usuarios = [dict(zip(cols, row)) for row in cur.fetchall()]
        for u in usuarios:
            u["Ultimo_Acceso_Fmt"] = _fmt_fecha(u.get("Ultimo_Acceso"))
        return usuarios
    finally:
        conn.close()


def create_user(username: str, nombre: str, correo: str, rol: str, password: str):
    username = (username or "").strip()
    if not username or not nombre or not password:
        raise UserError("Username, nombre y contraseña son obligatorios.")
    if len(password) < 8:
        raise UserError("La contraseña debe tener al menos 8 caracteres.")
    pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM COE.User_Admin WHERE Username = ?", username)
        if cur.fetchone():
            raise UserError(f"Ya existe un usuario con el username '{username}'.")
        cur.execute(
            "INSERT INTO COE.User_Admin (Username, Nombre_Completo, Correo, Password_Hash, Rol) "
            "VALUES (?, ?, ?, ?, ?)",
            username, nombre, correo or None, pwd_hash, rol or "usuario",
        )
        conn.commit()
    finally:
        conn.close()


def set_user_active(user_id: int, activo: bool):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE COE.User_Admin SET Activo=?, Fecha_Actualizacion=SYSDATETIME() WHERE Id=?",
            1 if activo else 0, user_id,
        )
        if cur.rowcount == 0:
            raise UserError("Usuario no encontrado.")
        conn.commit()
    finally:
        conn.close()


def set_user_password(user_id: int, new_password: str):
    if not new_password or len(new_password) < 8:
        raise UserError("La contraseña debe tener al menos 8 caracteres.")
    pwd_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE COE.User_Admin SET Password_Hash=?, Fecha_Actualizacion=SYSDATETIME() WHERE Id=?",
            pwd_hash, user_id,
        )
        if cur.rowcount == 0:
            raise UserError("Usuario no encontrado.")
        conn.commit()
    finally:
        conn.close()


def log_access(username: str, exitoso: bool, ip: str = None, user_agent: str = None, usuario_id: int = None):
    """Escribe una fila en COE.Acceso_Log — se llama tanto en login exitoso
    como fallido (bitacora de seguridad, no solo de uso). Si la base no esta
    disponible, no revienta el flujo de login por esto: solo lo registra en
    consola — perder una fila de auditoria es preferible a que nadie pueda
    entrar al tablero porque el log fallo."""
    try:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO COE.Acceso_Log (Usuario_Id, Username_Intento, Exitoso, IP_Origen, User_Agent) "
                "VALUES (?, ?, ?, ?, ?)",
                usuario_id, username, 1 if exitoso else 0, ip, user_agent,
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[db.log_access] no se pudo escribir la bitacora de acceso: {e}")
