"""Camada centralizada de estilos do app em base neutra.

Nesta etapa, removemos cores customizadas para deixar o Streamlit governar
o tema. Permanecem apenas regras de layout, espaçamento, tamanho e borda
baseadas em herança/currentColor.
"""

import streamlit as st

APP_BASE_CSS = """
.stButton>button {
    width: 100%;
    height: 3.5em;
    font-weight: bold;
    border-radius: 8px;
}

.stTextArea textarea {
    font-size: 16px;
}

.block-container {
    padding-top: 1.15rem;
    padding-bottom: 3rem;
}

.stAlert {
    font-weight: bold;
}

.section-title {
    margin-top: 1.2rem;
    margin-bottom: 0.45rem;
    font-size: 1.08rem;
    font-weight: 700;
}

.section-subtitle {
    margin-top: -0.10rem;
    margin-bottom: 0.85rem;
    font-size: 0.93rem;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 0.5rem 0 1rem 0;
}

.summary-card {
    background: transparent;
    border: 1px solid currentColor;
    border-radius: 14px;
    padding: 12px 14px;
}

.summary-label {
    font-size: 0.76rem;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.summary-value {
    font-size: 1.2rem;
    font-weight: 800;
}

.theme-panel {
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0.35rem 0 0.8rem 0;
    background: transparent;
    border: 1px solid currentColor;
}

.theme-panel--summary {
    margin-bottom: 0.75rem;
    padding: 10px 14px;
}

.theme-panel__title {
    font-weight: 700;
    margin-bottom: 8px;
}

.theme-panel--summary .theme-panel__title {
    font-size: 0.98rem;
    margin-bottom: 6px;
}

.theme-panel__line {
    margin-bottom: 4px;
}

.theme-panel--summary .theme-panel__line {
    margin-bottom: 3px;
}

.theme-panel__line:last-child {
    margin-bottom: 0;
}

.theme-panel__strong {
    font-weight: 700;
}

.theme-panel__label {
    font-weight: 600;
}

.team-card {
    background: transparent;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid currentColor;
}

.team-card__header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    border-bottom: 1px solid currentColor;
    padding-bottom: 10px;
    gap: 10px;
    align-items: center;
}

.team-card__title {
    margin: 0;
}

.team-card__odd {
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 700;
    white-space: nowrap;
    border: 1px solid currentColor;
    background: transparent;
}

.team-card__stats {
    padding: 8px;
    border-radius: 8px;
    display: flex;
    justify-content: space-around;
    margin-bottom: 10px;
    gap: 8px;
    flex-wrap: wrap;
    border: 1px solid currentColor;
    background: transparent;
}

.team-card__player-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid currentColor;
    gap: 10px;
}

.team-card__player-row:last-child {
    border-bottom: none;
}

.team-card__player-main {
    display: flex;
    align-items: center;
    gap: 8px;
}

.team-card__player-name {
    font-weight: 500;
}

.team-card__player-pos {
    padding: 2px 6px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid currentColor;
    background: transparent;
}

.team-card__metrics {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

h1 {
    margin-top: 0.1rem !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.05 !important;
}

#install-app-container {
    margin: 0.15rem 0 0.12rem 0 !important;
}

#install-app-container a,
#install-app-container button,
#install-app-container [role="button"],
#install-app-container .stButton > button {
    background: transparent !important;
    color: inherit !important;
    border: 1px solid currentColor !important;
    box-shadow: none !important;
}

#install-app-container a:hover,
#install-app-container button:hover,
#install-app-container [role="button"]:hover,
#install-app-container .stButton > button:hover {
    background: transparent !important;
    color: inherit !important;
    border-color: currentColor !important;
    box-shadow: none !important;
    transform: none !important;
}

#install-app-container a *,
#install-app-container button *,
#install-app-container [role="button"] *,
#install-app-container .stButton > button * {
    color: currentColor !important;
    fill: currentColor !important;
    stroke: currentColor !important;
}

@media (max-width: 900px) {
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .session-status-panel__grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""

ACTION_BUTTON_CSS = """
[class*="st-key-action-primary-"] div.stButton > button,
[class*="st-key-action-primary-"] div[data-testid="stFormSubmitButton"] > button,
[class*="st-key-action-secondary-"] div.stButton > button,
[class*="st-key-action-secondary-"] div[data-testid="stFormSubmitButton"] > button,
[class*="st-key-action-danger-"] div.stButton > button,
[class*="st-key-action-danger-"] div[data-testid="stFormSubmitButton"] > button {
    background: transparent !important;
    color: inherit !important;
    border: 1px solid currentColor !important;
    border-radius: 14px !important;
    min-height: 3.15rem !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

[class*="st-key-action-primary-"] div.stButton > button:hover,
[class*="st-key-action-primary-"] div[data-testid="stFormSubmitButton"] > button:hover,
[class*="st-key-action-secondary-"] div.stButton > button:hover,
[class*="st-key-action-secondary-"] div[data-testid="stFormSubmitButton"] > button:hover,
[class*="st-key-action-danger-"] div.stButton > button:hover,
[class*="st-key-action-danger-"] div[data-testid="stFormSubmitButton"] > button:hover {
    background: transparent !important;
    color: inherit !important;
    border-color: currentColor !important;
}

[class*="st-key-action-"] div.stButton > button:disabled,
[class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled {
    background: transparent !important;
    color: inherit !important;
    border: 1px solid currentColor !important;
    opacity: 0.65 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

.session-status-panel {
    margin: 0.3rem 0 0.85rem 0;
    padding: 0.8rem 0.95rem;
    border: 1px solid currentColor;
    border-radius: 16px;
    background: transparent;
}

.session-status-panel__eyebrow {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.78;
    margin-bottom: 0.45rem;
}

.session-status-panel__grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 0.7rem;
}

.session-status-panel__item {
    border: 1px solid currentColor;
    border-radius: 12px;
    padding: 0.7rem 0.8rem;
    min-width: 0;
}

.session-status-panel__label {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.74;
    margin-bottom: 0.22rem;
}

.session-status-panel__value {
    font-size: 0.94rem;
    font-weight: 700;
    line-height: 1.35;
    word-break: break-word;
}

.session-status-panel__next {
    border-top: 1px solid currentColor;
    padding-top: 0.55rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    align-items: baseline;
}

.session-status-panel__next-label {
    font-size: 0.84rem;
    font-weight: 600;
    opacity: 0.8;
}

.session-status-panel__next-value {
    font-size: 0.96rem;
    font-weight: 700;
}

.action-hint {
    margin-top: 0.35rem;
    margin-bottom: 0.6rem;
    font-size: 0.92rem;
}

.inline-status-note {
    margin: 0.15rem 0 0.55rem 0;
    padding: 0.5rem 0.75rem;
    border: 1px solid currentColor;
    border-radius: 12px;
    background: transparent;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    align-items: baseline;
}

.inline-status-note--info {
    border-color: rgba(255,255,255,0.18);
}

.inline-status-note--success {
    border-color: rgba(46, 204, 113, 0.35);
    background: rgba(46, 204, 113, 0.06);
}

.inline-status-note--warning {
    border-color: rgba(241, 196, 15, 0.35);
    background: rgba(241, 196, 15, 0.06);
}

.inline-status-note__title {
    font-weight: 700;
}

.inline-status-note__desc {
    opacity: 0.88;
    line-height: 1.35;
    font-size: 0.93rem;
}

.step-cta-panel {
    margin: 0.45rem 0 0.85rem 0;
    padding: 0.9rem 1rem;
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
}

.step-cta-panel--info {
    border-color: rgba(255,255,255,0.14);
}

.step-cta-panel--success {
    border-color: rgba(46, 204, 113, 0.35);
    background: rgba(46, 204, 113, 0.08);
}

.step-cta-panel--warning {
    border-color: rgba(241, 196, 15, 0.35);
    background: rgba(241, 196, 15, 0.08);
}

.step-cta-panel__eyebrow {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 0.25rem;
}

.step-cta-panel__title {
    font-size: 1.02rem;
    font-weight: 700;
    margin-bottom: 0.18rem;
}

.step-cta-panel__desc {
    font-size: 0.93rem;
    opacity: 0.9;
    line-height: 1.45;
}
"""


def apply_app_styles():
    st.markdown(
        f"""<style>{APP_BASE_CSS}
{ACTION_BUTTON_CSS}</style>""",
        unsafe_allow_html=True,
    )
