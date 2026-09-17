const chatMsgs = document.getElementById('chat-msgs');
const input = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

/** Si la sesion expiro (401) mientras se usaba el chat, manda a /login en
 * vez de mostrar un error críptico. Devuelve true si redirigio (para que
 * el que llama corte el flujo ahi mismo). */
function redirectIfLoggedOut(res) {
    if (res.status === 401) { window.location.href = '/login'; return true; }
    return false;
}

input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

function addMsg(text, cls, footer) {
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    d.textContent = text;
    if (cls === 'msg-bot') {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'msg-copy-btn';
        copyBtn.textContent = '⧉';
        copyBtn.title = 'Copiar respuesta';
        copyBtn.onclick = () => copyMsg(copyBtn, text);
        d.appendChild(copyBtn);
    }
    if (footer) {
        const f = document.createElement('div');
        f.className = 'msg-footer';
        f.textContent = footer;
        d.appendChild(f);
    }
    chatMsgs.appendChild(d);
    scrollChatToBottom();
    return d;
}

function copyMsg(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        btn.textContent = '✓';
        setTimeout(() => { btn.classList.remove('copied'); btn.textContent = '⧉'; }, 1500);
    }).catch(() => {});
}

function clearChat() {
    if (!confirm('¿Limpiar toda la conversación?')) return;
    chatMsgs.innerHTML = '';
    addMsg(
        'Conversación limpia. Sigo aquí con los mismos datos conectados — pregúntame cuando quieras.',
        'msg-bot'
    );
}

// ── Scroll: baja siempre al fondo salvo que el usuario haya subido a leer
// algo — ahi solo se muestra el boton flotante en vez de forzar el scroll. ──
const scrollBtn = document.getElementById('scroll-bottom-btn');
function isNearBottom() {
    return chatMsgs.scrollHeight - chatMsgs.scrollTop - chatMsgs.clientHeight < 80;
}
function scrollChatToBottom() {
    chatMsgs.scrollTop = chatMsgs.scrollHeight;
    scrollBtn.classList.remove('visible');
}
chatMsgs.addEventListener('scroll', () => {
    scrollBtn.classList.toggle('visible', !isNearBottom());
});

// ── Ajustar el ancho del panel (arrastrar) y colapsar/expandir el chat ──
const layoutEl = document.getElementById('layout');
const chatPaneEl = document.getElementById('chat-pane');
const resizerEl = document.getElementById('resizer');
const reopenBtn = document.getElementById('chat-reopen-btn');
const CHAT_W_KEY = 'pbi_copiloto_chat_w';
const CHAT_COLLAPSED_KEY = 'pbi_copiloto_chat_collapsed';
const CHAT_W_MIN = 300, CHAT_W_MAX = 640;

function setChatWidth(px, persist) {
    const clamped = Math.min(CHAT_W_MAX, Math.max(CHAT_W_MIN, px));
    document.documentElement.style.setProperty('--chat-w', clamped + 'px');
    if (persist) { try { localStorage.setItem(CHAT_W_KEY, String(clamped)); } catch {} }
}

(function restoreChatWidth() {
    try {
        const saved = parseInt(localStorage.getItem(CHAT_W_KEY), 10);
        if (saved) setChatWidth(saved, false);
    } catch {}
})();

resizerEl.addEventListener('mousedown', e => {
    e.preventDefault();
    if (chatPaneEl.classList.contains('collapsed')) return;
    chatPaneEl.classList.add('no-transition');
    resizerEl.classList.add('dragging');
    const onMove = ev => setChatWidth(window.innerWidth - ev.clientX, false);
    const onUp = ev => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        chatPaneEl.classList.remove('no-transition');
        resizerEl.classList.remove('dragging');
        setChatWidth(window.innerWidth - ev.clientX, true);
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
});

function toggleChatCollapse(forceState) {
    const collapse = forceState !== undefined ? forceState : !chatPaneEl.classList.contains('collapsed');
    chatPaneEl.classList.toggle('collapsed', collapse);
    layoutEl.classList.toggle('chat-collapsed', collapse);
    reopenBtn.hidden = !collapse;
    try { localStorage.setItem(CHAT_COLLAPSED_KEY, collapse ? '1' : '0'); } catch {}
}

(function restoreChatCollapsed() {
    try {
        if (localStorage.getItem(CHAT_COLLAPSED_KEY) === '1') toggleChatCollapse(true);
    } catch {}
})();

// ── Pantalla completa para el panel del reporte ──
function toggleFullscreen() {
    const pane = document.getElementById('report-pane');
    if (!document.fullscreenElement) {
        pane.requestFullscreen?.().catch(() => {});
    } else {
        document.exitFullscreen?.();
    }
}

function sendQuick(text) { input.value = text; sendChat(); }

function botFooter(d, secs) {
    const parts = [`⏱ ${fmtSecs(secs)}`];
    if (d.cached) parts.push('♻️ misma respuesta que ya te di antes (datos sin cambios)');
    if (d.corrected) parts.push('🔎 revisé y ajusté una cifra que no coincidía con los datos');
    return parts.join(' · ');
}

// ── Timer con estimado (aprende de tus tiempos reales, por tipo de pregunta) ──

const BASE_ETA = { chat: 55, reporte: 45 };
const LS_KEY = 'pbi_copiloto_timings';

function getTimings() { try { return JSON.parse(localStorage.getItem(LS_KEY) || '{}'); } catch { return {}; } }
function saveTimings(t) { localStorage.setItem(LS_KEY, JSON.stringify(t)); }
function getEta(tipo) {
    const t = getTimings();
    if (t[tipo] && t[tipo].count > 0) return Math.round(t[tipo].avg * 0.7 + (BASE_ETA[tipo] || 45) * 0.3);
    return BASE_ETA[tipo] || 45;
}
function recordTiming(tipo, seconds) {
    const t = getTimings();
    if (!t[tipo]) t[tipo] = { avg: seconds, count: 1 };
    else {
        const n = Math.min(t[tipo].count + 1, 10); // ventana de las ultimas 10 muestras
        t[tipo].avg = ((t[tipo].avg * (n - 1)) + seconds) / n;
        t[tipo].count = n;
    }
    saveTimings(t);
}
function fmtSecs(s) {
    s = Math.floor(s);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

/** Crea una burbuja de "pensando" con cronómetro + barra de progreso estimada. */
function createThinkTimer(tipo) {
    const eta = getEta(tipo);
    const el = document.createElement('div');
    el.className = 'msg msg-think';
    el.innerHTML = `
        <div class="think-row">
            <span class="think-dots" aria-hidden="true"><span></span><span></span><span></span></span>
            <span class="think-status" id="tk-status">Buscando en los datos y redactando</span>
            <span class="think-elapsed" id="tk-elapsed">0s</span>
        </div>
        <div class="think-bar"><div class="think-fill" id="tk-fill"></div></div>
        <div class="think-eta" id="tk-eta">~${fmtSecs(eta)} (según tu histórico local)</div>`;
    chatMsgs.appendChild(el);
    chatMsgs.scrollTop = chatMsgs.scrollHeight;

    const elapsedEl = el.querySelector('#tk-elapsed');
    const fillEl = el.querySelector('#tk-fill');
    const etaEl = el.querySelector('#tk-eta');
    const statusEl = el.querySelector('#tk-status');

    const start = Date.now();
    const iv = setInterval(() => {
        const elapsed = (Date.now() - start) / 1000;
        elapsedEl.textContent = fmtSecs(elapsed);
        const pct = Math.min((elapsed / eta) * 100, 100);
        fillEl.style.width = pct + '%';
        if (elapsed > eta) {
            fillEl.classList.add('overtime');
            etaEl.textContent = `+${fmtSecs(elapsed - eta)} sobre lo habitual`;
            statusEl.textContent = 'Modelo local pensando (puede tardar si estaba inactivo)…';
        } else {
            etaEl.textContent = `~${fmtSecs(eta - elapsed)} restantes`;
            statusEl.textContent = elapsed < 3 ? 'Enviando a Ollama…' : elapsed < 10 ? 'Redactando…' : 'Verificando cifras contra los datos…';
        }
        chatMsgs.scrollTop = chatMsgs.scrollHeight;
    }, 500);

    return {
        el,
        stop(secs) { clearInterval(iv); recordTiming(tipo, secs); },
        cancel() { clearInterval(iv); }, // no registra tiempo (ej. saludos, que no pasan por el LLM)
    };
}

async function runReport() {
    addMsg('Generar reporte corto', 'msg-user');
    const timer = createThinkTimer('reporte');
    sendBtn.disabled = true;
    const t0 = Date.now();
    try {
        const res = await fetch('/api/report', { method: 'POST' });
        if (redirectIfLoggedOut(res)) return;
        const d = await res.json();
        const secs = (Date.now() - t0) / 1000;
        d.cached ? timer.cancel() : timer.stop(secs);
        timer.el.remove();
        d.ok ? addMsg(d.text, 'msg-bot', botFooter(d, secs)) : addMsg('Error: ' + d.error, 'msg-err');
    } catch (e) {
        timer.stop((Date.now() - t0) / 1000);
        timer.el.remove();
        addMsg('Error de conexión: ' + e.message, 'msg-err');
    }
    sendBtn.disabled = false;
}

async function sendChat() {
    const q = input.value.trim();
    if (!q) return;
    input.value = '';
    sendBtn.disabled = true;
    addMsg(q, 'msg-user');
    const timer = createThinkTimer('chat');
    const t0 = Date.now();
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: q }),
        });
        if (redirectIfLoggedOut(res)) return;
        const d = await res.json();
        const secs = (Date.now() - t0) / 1000;
        // Saludos, respuestas directas (lookup sin LLM) y respuestas
        // cacheadas (Agente de Consistencia) son instantaneos: no cuentan
        // para el promedio aprendido, o la estimacion para preguntas que si
        // pasan por el modelo quedaria artificialmente baja.
        (d.intent === 'saludo' || d.intent === 'directo' || d.cached) ? timer.cancel() : timer.stop(secs);
        timer.el.remove();
        d.ok ? addMsg(d.text, 'msg-bot', botFooter(d, secs)) : addMsg('Error: ' + d.error, 'msg-err');
    } catch (e) {
        timer.stop((Date.now() - t0) / 1000);
        timer.el.remove();
        addMsg('Error de conexión: ' + e.message, 'msg-err');
    }
    sendBtn.disabled = false;
}

async function refreshStatus() {
    const pillLlm = document.getElementById('pill-llm');
    const pillData = document.getElementById('pill-data');
    const dataNote = document.getElementById('data-note');
    const brandSub = document.getElementById('brand-sub');
    try {
        const res = await fetch('/api/status');
        if (redirectIfLoggedOut(res)) return;
        const d = await res.json();
        pillLlm.textContent = d.llm.ok ? `Asistente activo · ${d.llm.active || ''}` : 'Asistente no disponible';
        pillLlm.className = 'pill ' + (d.llm.ok ? 'ok' : 'err');

        if (d.data.ok) {
            pillData.textContent = `Datos: ${d.data.tables} tablas conectadas`;
            pillData.className = 'pill ok';
            dataNote.textContent = 'Respondiendo con los datos reales del tablero';
            if (d.data.source_file) brandSub.textContent = d.data.source_file;
        } else {
            pillData.textContent = 'Sin datos conectados';
            pillData.className = 'pill err';
            dataNote.textContent = 'Usa "Actualizar datos" arriba para conectar el .pbix';
        }
    } catch (e) {
        pillLlm.textContent = 'Sin respuesta del servidor';
        pillLlm.className = 'pill err';
    }
}

async function refreshData() {
    const btn = document.getElementById('refresh-btn');
    btn.classList.add('loading');
    btn.textContent = '🔄 Leyendo .pbix…';
    try {
        const res = await fetch('/api/refresh', { method: 'POST' });
        if (redirectIfLoggedOut(res)) return;
        const d = await res.json();
        if (!d.ok) throw new Error(d.error || 'Error desconocido');
        await refreshStatus();
    } catch (e) {
        alert('No se pudo actualizar: ' + e.message);
    }
    btn.classList.remove('loading');
    btn.textContent = '🔄 Actualizar datos';
}

refreshStatus();
