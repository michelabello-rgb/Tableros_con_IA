// Embed real e interactivo del reporte, via Azure AD (MSAL) + powerbi-client.
// Solo se carga si AZURE_CLIENT_ID / AZURE_TENANT_ID estan configurados (ver .env).
const PBI_SCOPES = ["https://analysis.windows.net/powerbi/api/Report.Read.All"];

let _msalApp = null;
let _msalReady = null;
async function getMsalApp() {
    if (typeof msal === 'undefined') {
        throw new Error('No se pudo cargar la libreria de Microsoft (MSAL) desde el CDN. Revisa tu conexion o si un bloqueador/proxy esta filtrando alcdn.msauth.net.');
    }
    if (typeof window['powerbi-client'] === 'undefined') {
        throw new Error('No se pudo cargar la libreria powerbi-client desde el CDN. Revisa tu conexion.');
    }
    if (!_msalApp) {
        _msalApp = new msal.PublicClientApplication({
            auth: {
                clientId: window.EMBED_CONFIG.clientId,
                authority: `https://login.microsoftonline.com/${window.EMBED_CONFIG.tenantId}`,
                redirectUri: window.location.origin,
            },
        });
        // MSAL v3 ya no se auto-inicializa: hay que esperar initialize() antes
        // de llamar a cualquier otro metodo (login, tokens, cuentas...), o
        // tira "uninitialized_public_client_application".
        _msalReady = _msalApp.initialize();
    }
    await _msalReady;
    return _msalApp;
}

async function getPbiToken() {
    const msalApp = await getMsalApp();
    let account = msalApp.getAllAccounts()[0];
    if (!account) {
        const res = await msalApp.loginPopup({ scopes: PBI_SCOPES });
        account = res.account;
    }
    try {
        const res = await msalApp.acquireTokenSilent({ scopes: PBI_SCOPES, account });
        return res.accessToken;
    } catch (e) {
        const res = await msalApp.acquireTokenPopup({ scopes: PBI_SCOPES, account });
        return res.accessToken;
    }
}

async function connectReport() {
    const errEl = document.getElementById('embed-error');
    errEl.textContent = '';
    try {
        const token = await getPbiToken();
        const { groupId, reportId } = window.EMBED_CONFIG;

        const res = await fetch(`https://api.powerbi.com/v1.0/myorg/groups/${groupId}/reports/${reportId}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error(`Power BI API respondio ${res.status}. Verifica que tu cuenta tenga acceso al reporte y que el permiso Report.Read.All tenga consentimiento.`);
        const reportMeta = await res.json();

        const models = window['powerbi-client'].models;
        const service = new window['powerbi-client'].service.Service(
            window['powerbi-client'].factories.hpmFactory,
            window['powerbi-client'].factories.wpmpFactory,
            window['powerbi-client'].factories.routerFactory,
        );
        const container = document.getElementById('report-container');
        service.embed(container, {
            type: 'report',
            tokenType: models.TokenType.Aad,
            accessToken: token,
            embedUrl: reportMeta.embedUrl,
            id: reportId,
            permissions: models.Permissions.Read,
            settings: { panes: { filters: { visible: true } }, background: models.BackgroundType.Transparent },
        });

        document.getElementById('embed-gate').style.display = 'none';
        container.style.display = 'block';
    } catch (e) {
        errEl.textContent = 'Error al conectar: ' + e.message;
    }
}
