import json

import streamlit as st
import streamlit.components.v1 as components


def _escape_html_attr(texto: str) -> str:
    return (
        texto.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )


def botao_copiar_js(texto_para_copiar):
    texto_js = json.dumps(texto_para_copiar)
    html_code = f"""
    <div style="display:flex; justify-content:center; margin-bottom:12px;">
        <button id="copy-btn" onclick="copiarTexto()" style="width:100%; height:50px; background:transparent; color:inherit; border:1px solid currentColor; border-radius:8px; font-weight:bold; font-size:16px; cursor:pointer;">📋 COPIAR</button>
        <script>
            const btn = document.getElementById('copy-btn');

            function getParentThemeColor() {{
                try {{
                    const parentDoc = window.parent.document;
                    const appRoot = parentDoc.querySelector('.stApp') || parentDoc.body || parentDoc.documentElement;
                    return window.parent.getComputedStyle(appRoot).color || '#31333f';
                }} catch (e) {{
                    return '#31333f';
                }}
            }}

            function applyThemeToCopyButton() {{
                const color = getParentThemeColor();
                document.body.style.background = 'transparent';
                document.body.style.color = color;
                document.documentElement.style.background = 'transparent';
                btn.style.color = color;
                btn.style.borderColor = color;
                btn.style.background = 'transparent';
            }}

            function copiarTexto() {{
                const texto = {texto_js};
                const el = document.createElement('textarea');
                el.value = texto;
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                const originalText = btn.innerText;
                btn.innerText = '✅ COPIADO!';
                setTimeout(() => {{ btn.innerText = originalText; }}, 2000);
            }}

            applyThemeToCopyButton();

            try {{
                const parentDoc = window.parent.document;
                const observer = new MutationObserver(() => applyThemeToCopyButton());
                observer.observe(parentDoc.documentElement, {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }});
                const appRoot = parentDoc.querySelector('.stApp');
                if (appRoot) {{
                    observer.observe(appRoot, {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }});
                }}
            }} catch (e) {{
                // no-op
            }}
        </script>
    </div>
    """
    components.html(html_code, height=70)


def botao_compartilhar_js(texto_para_compartilhar):
    texto_js = json.dumps(texto_para_compartilhar)
    titulo_attr = _escape_html_attr('Compartilhar resultado')
    html_code = f"""
    <div style="display:flex; justify-content:center; margin-bottom:20px;">
        <button id="share-btn" type="button" aria-label="{titulo_attr}" onclick="compartilharTexto()" style="width:100%; height:50px; background:transparent; color:inherit; border:1px solid currentColor; border-radius:8px; font-weight:bold; font-size:16px; cursor:pointer;">📤 COMPARTILHAR</button>
        <script>
            const btn = document.getElementById('share-btn');

            function getParentThemeColor() {{
                try {{
                    const parentDoc = window.parent.document;
                    const appRoot = parentDoc.querySelector('.stApp') || parentDoc.body || parentDoc.documentElement;
                    return window.parent.getComputedStyle(appRoot).color || '#31333f';
                }} catch (e) {{
                    return '#31333f';
                }}
            }}

            function applyThemeToShareButton() {{
                const color = getParentThemeColor();
                document.body.style.background = 'transparent';
                document.body.style.color = color;
                document.documentElement.style.background = 'transparent';
                btn.style.color = color;
                btn.style.borderColor = color;
                btn.style.background = 'transparent';
            }}

            async function compartilharTexto() {{
                const texto = {texto_js};
                const originalText = btn.innerText;
                try {{
                    if (navigator.share) {{
                        btn.innerText = '📤 ABRINDO COMPARTILHAMENTO...';
                        await navigator.share({{ text: texto }});
                        btn.innerText = '✅ COMPARTILHADO';
                        setTimeout(() => {{ btn.innerText = originalText; }}, 2200);
                        return;
                    }}

                    const el = document.createElement('textarea');
                    el.value = texto;
                    document.body.appendChild(el);
                    el.select();
                    document.execCommand('copy');
                    document.body.removeChild(el);
                    btn.innerText = '✅ COPIADO';
                    setTimeout(() => {{ btn.innerText = originalText; }}, 2400);
                }} catch (e) {{
                    if (e && (e.name === 'AbortError' || e.name === 'NotAllowedError')) {{
                        btn.innerText = originalText;
                        return;
                    }}
                    btn.innerText = '⚠️ NÃO FOI POSSÍVEL';
                    setTimeout(() => {{ btn.innerText = originalText; }}, 2200);
                }}
            }}

            applyThemeToShareButton();

            try {{
                const parentDoc = window.parent.document;
                const observer = new MutationObserver(() => applyThemeToShareButton());
                observer.observe(parentDoc.documentElement, {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }});
                const appRoot = parentDoc.querySelector('.stApp');
                if (appRoot) {{
                    observer.observe(appRoot, {{ attributes: true, attributeFilter: ['class', 'style', 'data-theme'] }});
                }}
            }} catch (e) {{
                // no-op
            }}
        </script>
    </div>
    """
    components.html(html_code, height=70)


def botao_instalar_app():
    st.html(
        """
        <div id="install-app-container" style="margin:0.2rem 0 0.8rem 0; text-align:center;">
            <button id="install-app-btn" type="button"
                style="width:auto; min-width:230px; height:42px; padding:0 16px; background:transparent; color:inherit; border:1px solid currentColor; border-radius:999px; font-weight:600; font-size:14px; cursor:pointer; box-shadow:none;">
                📲 INSTALAR APLICATIVO
            </button>
            <div id="install-app-msg"
                style="margin-top:8px; min-height:20px; font-size:13px; color:inherit; line-height:1.45;">
            </div>
        </div>

        <script>
        (() => {
            const btn = document.getElementById("install-app-btn");
            const msg = document.getElementById("install-app-msg");

            let deferredPrompt = null;
            const ua = window.navigator.userAgent || "";
            const isIOS = /iphone|ipad|ipod/i.test(ua);
            const isAndroid = /android/i.test(ua);
            const isMobile = /android|iphone|ipad|ipod|mobile/i.test(ua);
            const isEdge = /edg/i.test(ua);
            const isChrome = /chrome|chromium|crios/i.test(ua) && !isEdge;
            const isSafari = /safari/i.test(ua) && !isChrome && !isEdge;
            const isStandalone =
                window.matchMedia("(display-mode: standalone)").matches ||
                window.navigator.standalone === true;

            function mensagemFallback() {
                if (isIOS || isSafari) {
                    return "No iPhone/iPad: abra no Safari, toque em Compartilhar e depois em Adicionar à Tela de Início.";
                }

                if (isMobile) {
                    if (isChrome || isEdge || isAndroid) {
                        return "No celular: toque nos três pontinhos do navegador e depois em Adicionar à tela inicial ou Instalar aplicativo.";
                    }
                    return "No celular: abra o menu do navegador e procure por Adicionar à tela inicial ou Instalar aplicativo.";
                }

                if (isChrome || isEdge) {
                    return "No computador: clique nos três pontinhos do navegador, depois em Transmitir, salvar e compartilhar e depois em Instalar Sorteador Pelada PRO. Em alguns casos pode aparecer como Instalar Streamlit.";
                }

                return "Neste navegador: abra o menu e procure por Instalar aplicativo ou Adicionar à tela inicial.";
            }

            if (isStandalone) {
                btn.style.display = "none";
                msg.innerHTML = "✅ O app já está instalado neste dispositivo.";
                return;
            }

            msg.innerHTML = "";

            window.addEventListener("beforeinstallprompt", (e) => {
                e.preventDefault();
                deferredPrompt = e;
                msg.innerHTML = "";
            });

            btn.addEventListener("click", async () => {
                if (!deferredPrompt) {
                    const texto = mensagemFallback();
                    msg.innerText = texto;
                    window.alert(texto);
                    return;
                }

                deferredPrompt.prompt();
                const choice = await deferredPrompt.userChoice;

                if (choice.outcome === "accepted") {
                    msg.innerHTML = "✅ Instalação iniciada.";
                    btn.style.display = "none";
                } else {
                    const texto = mensagemFallback();
                    msg.innerText = texto;
                    window.alert(texto);
                }

                deferredPrompt = null;
            });

            window.addEventListener("appinstalled", () => {
                btn.style.display = "none";
                msg.innerHTML = "✅ App instalado com sucesso.";
            });
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
