import json

import streamlit as st
import streamlit.components.v1 as components


def botao_copiar_js(texto_para_copiar):
    texto_js = json.dumps(texto_para_copiar)
    html_code = f"""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <button onclick="copiarTexto()" style="width: 100%; height: 50px; background-color: #25D366; color: white; border: none; border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">📋 COPIAR PARA WHATSAPP</button>
        <script>
            function copiarTexto() {{
                const texto = {texto_js};
                const el = document.createElement('textarea'); el.value = texto; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
                const btn = document.querySelector('button'); const originalText = btn.innerText; btn.innerText = '✅ COPIADO!'; btn.style.backgroundColor = '#128C7E';
                setTimeout(() => {{ btn.innerText = originalText; btn.style.backgroundColor = '#25D366'; }}, 2000);
            }}
        </script>
    </div>
    """
    components.html(html_code, height=70)


def botao_instalar_app():
    st.html(
        """
        <div id="install-app-container" style="margin: 0.2rem 0 0.8rem 0; text-align: center;">
            <button id="install-app-btn" type="button"
                style="
                    width: auto;
                    min-width: 230px;
                    height: 42px;
                    padding: 0 16px;
                    background-color: rgba(15, 118, 110, 0.16);
                    color: #d1fae5;
                    border: 1px solid #2dd4bf;
                    border-radius: 999px;
                    font-weight: 600;
                    font-size: 14px;
                    cursor: pointer;
                    box-shadow: none;
                "
            >
                📲 INSTALAR APLICATIVO
            </button>
            <div id="install-app-msg"
                style="
                    margin-top: 8px;
                    min-height: 20px;
                    font-size: 13px;
                    color: #94a3b8;
                    line-height: 1.45;
                "
            >
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
