"""Camada centralizada de estilos do app.

Este módulo concentra os blocos CSS do Streamlit para facilitar auditoria
e futuras melhorias de tema sem espalhar estilo pelo app principal.
"""

import streamlit as st

APP_BASE_CSS = """
.stButton>button {
    width: 100%;
    height: 3.5em;
    font-weight: bold;
    background: var(--button-default-bg);
    color: var(--button-default-text);
    border-radius: 8px;
    border: 1px solid var(--button-default-border);
}
.stButton>button:hover {
    background: var(--button-default-bg-hover);
    color: var(--button-default-text);
    border-color: var(--button-default-border-hover);
}
.stTextArea textarea { font-size: 16px; }
.block-container { padding-top: 1.15rem; padding-bottom: 3rem; }
.stAlert { font-weight: bold; }

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
    background: var(--summary-card-bg);
    border: 1px solid var(--summary-card-border);
    border-top: 3px solid var(--summary-card-accent);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: var(--summary-card-shadow);
}

.summary-label {
    font-size: 0.76rem;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--summary-label-text);
}

.summary-value {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--summary-value-text);
}

:root {
    --button-default-bg: #e2e8f0;
    --button-default-bg-hover: #cbd5e1;
    --button-default-border: #94a3b8;
    --button-default-border-hover: #64748b;
    --button-default-text: #0f172a;

    --install-button-bg: rgba(241, 245, 249, 0.98);
    --install-button-bg-hover: rgba(226, 232, 240, 0.98);
    --install-button-border: rgba(148, 163, 184, 0.78);
    --install-button-border-hover: rgba(100, 116, 139, 0.82);
    --install-button-text: #334155;

    --custom-panel-bg: linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.98) 100%);
    --custom-panel-border: #cbd5e1;
    --custom-panel-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    --custom-panel-text: #0f172a;
    --custom-panel-muted: #475569;

    --summary-card-bg: linear-gradient(180deg, rgba(248, 250, 252, 0.99) 0%, rgba(241, 245, 249, 0.99) 100%);
    --summary-card-border: #d7dee7;
    --summary-card-accent: #16a34a;
    --summary-card-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    --summary-label-text: #475569;
    --summary-value-text: #0f172a;

    --team-card-bg: #ffffff;
    --team-card-border: #dbe2ea;
    --team-card-shadow: 0 2px 5px rgba(15, 23, 42, 0.08);
    --team-card-divider: #334155;
    --team-card-badge-bg: #facc15;
    --team-stats-bg: #f8fafc;
    --team-player-divider: #e5e7eb;
    --team-player-pos-bg: #e5e7eb;
}

.theme-panel {
    border-radius: 12px;
    padding: 12px 14px;
    margin: 0.35rem 0 0.8rem 0;
    background: var(--custom-panel-bg);
    border: 1px solid var(--custom-panel-border);
    box-shadow: var(--custom-panel-shadow);
    color: var(--custom-panel-text);
}

.theme-panel--summary {
    margin-bottom: 0.75rem;
    padding: 10px 14px;
}

.theme-panel__title {
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--custom-panel-text);
}

.theme-panel--summary .theme-panel__title {
    font-size: 0.98rem;
    margin-bottom: 6px;
}

.theme-panel__line {
    margin-bottom: 4px;
    color: var(--custom-panel-text);
}

.theme-panel--summary .theme-panel__line {
    margin-bottom: 3px;
}

.theme-panel__line:last-child {
    margin-bottom: 0;
}

.theme-panel__strong {
    font-weight: 700;
    color: var(--custom-panel-text);
}

.theme-panel__label {
    font-weight: 600;
    color: var(--custom-panel-muted);
}


html[data-theme="light"] .theme-panel,
body[data-theme="light"] .theme-panel,
[data-theme="light"] .theme-panel {
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.98) 100%) !important;
    border-color: #cbd5e1 !important;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05) !important;
    color: #0f172a !important;
}

html[data-theme="light"] .theme-panel__title,
html[data-theme="light"] .theme-panel__line,
html[data-theme="light"] .theme-panel__strong,
body[data-theme="light"] .theme-panel__title,
body[data-theme="light"] .theme-panel__line,
body[data-theme="light"] .theme-panel__strong,
[data-theme="light"] .theme-panel__title,
[data-theme="light"] .theme-panel__line,
[data-theme="light"] .theme-panel__strong {
    color: #0f172a !important;
}

html[data-theme="light"] .theme-panel__label,
body[data-theme="light"] .theme-panel__label,
[data-theme="light"] .theme-panel__label {
    color: #475569 !important;
}

html[data-theme="dark"] .theme-panel,
body[data-theme="dark"] .theme-panel,
[data-theme="dark"] .theme-panel {
    background: rgba(15, 23, 42, 0.55) !important;
    border-color: #334155 !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22) !important;
    color: #f8fafc !important;
}

html[data-theme="dark"] .theme-panel__title,
html[data-theme="dark"] .theme-panel__line,
html[data-theme="dark"] .theme-panel__strong,
body[data-theme="dark"] .theme-panel__title,
body[data-theme="dark"] .theme-panel__line,
body[data-theme="dark"] .theme-panel__strong,
[data-theme="dark"] .theme-panel__title,
[data-theme="dark"] .theme-panel__line,
[data-theme="dark"] .theme-panel__strong {
    color: #f8fafc !important;
}

html[data-theme="dark"] .theme-panel__label,
body[data-theme="dark"] .theme-panel__label,
[data-theme="dark"] .theme-panel__label {
    color: #cbd5e1 !important;
}

html[data-theme="light"] .summary-card,
body[data-theme="light"] .summary-card,
[data-theme="light"] .summary-card {
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.99) 0%, rgba(241, 245, 249, 0.99) 100%) !important;
    border-color: #d7dee7 !important;
    border-top-color: #16a34a !important;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06) !important;
}

html[data-theme="light"] .summary-label,
body[data-theme="light"] .summary-label,
[data-theme="light"] .summary-label {
    color: #475569 !important;
}

html[data-theme="light"] .summary-value,
body[data-theme="light"] .summary-value,
[data-theme="light"] .summary-value {
    color: #0f172a !important;
}

html[data-theme="dark"] .summary-card,
body[data-theme="dark"] .summary-card,
[data-theme="dark"] .summary-card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 24, 39, 0.92) 100%) !important;
    border-color: #253247 !important;
    border-top-color: #22c55e !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.16) !important;
}

html[data-theme="dark"] .summary-label,
body[data-theme="dark"] .summary-label,
[data-theme="dark"] .summary-label {
    color: #93c5fd !important;
}

html[data-theme="dark"] .summary-value,
body[data-theme="dark"] .summary-value,
[data-theme="dark"] .summary-value {
    color: #f8fafc !important;
}

.team-card {
    background: var(--team-card-bg);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid var(--team-card-border);
    box-shadow: var(--team-card-shadow);
}

.team-card__header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    border-bottom: 2px solid var(--team-card-divider);
    padding-bottom: 10px;
    gap: 10px;
    align-items: center;
}

.team-card__title {
    margin: 0;
}

.team-card__odd {
    background: var(--team-card-badge-bg);
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 700;
    white-space: nowrap;
}

.team-card__stats {
    background: var(--team-stats-bg);
    padding: 8px;
    border-radius: 8px;
    display: flex;
    justify-content: space-around;
    margin-bottom: 10px;
    gap: 8px;
    flex-wrap: wrap;
}

.team-card__player-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--team-player-divider);
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
    background: var(--team-player-pos-bg);
    padding: 2px 6px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
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

html[data-theme="dark"] {
    --button-default-bg: #1e293b;
    --button-default-bg-hover: #334155;
    --button-default-border: #475569;
    --button-default-border-hover: #64748b;
    --button-default-text: #e2e8f0;

    --install-button-bg: rgba(15, 23, 42, 0.28);
    --install-button-bg-hover: rgba(15, 23, 42, 0.42);
    --install-button-border: rgba(45, 212, 191, 0.55);
    --install-button-border-hover: rgba(45, 212, 191, 0.75);
    --install-button-text: #dbe7ef;

    --custom-panel-bg: rgba(15, 23, 42, 0.55);
    --custom-panel-border: #334155;
    --custom-panel-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    --custom-panel-text: #f8fafc;
    --custom-panel-muted: #cbd5e1;

    --summary-card-bg: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 24, 39, 0.92) 100%);
    --summary-card-border: #253247;
    --summary-card-accent: #22c55e;
    --summary-card-shadow: 0 6px 18px rgba(0,0,0,0.16);
    --summary-label-text: #93c5fd;
    --summary-value-text: #f8fafc;

    --team-card-bg: #1e1e1e;
    --team-card-border: #333333;
    --team-card-shadow: none;
    --team-card-divider: #555555;
    --team-card-badge-bg: #facc15;
    --team-stats-bg: #2a2a2a;
    --team-player-divider: #333333;
    --team-player-pos-bg: #333333;
}

@media (prefers-color-scheme: dark) {
    html:not([data-theme="light"]) {
        --button-default-bg: #1e293b;
        --button-default-bg-hover: #334155;
        --button-default-border: #475569;
        --button-default-border-hover: #64748b;
        --button-default-text: #e2e8f0;

        --custom-panel-bg: rgba(15, 23, 42, 0.55);
        --custom-panel-border: #334155;
        --custom-panel-text: #f8fafc;
        --custom-panel-muted: #cbd5e1;

        --team-card-bg: #1e1e1e;
        --team-card-border: #333333;
        --team-card-shadow: none;
        --team-card-divider: #555555;
        --team-card-badge-bg: #facc15;
        --team-stats-bg: #2a2a2a;
        --team-player-divider: #333333;
        --team-player-pos-bg: #333333;
    }
}

#install-app-container a,
#install-app-container button,
#install-app-container [role="button"],
#install-app-container .stButton > button {
    background: var(--install-button-bg) !important;
    color: var(--install-button-text) !important;
    border: 1px solid var(--install-button-border) !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

#install-app-container a:hover,
#install-app-container button:hover,
#install-app-container [role="button"]:hover,
#install-app-container .stButton > button:hover {
    background: var(--install-button-bg-hover) !important;
    color: var(--install-button-text) !important;
    border-color: var(--install-button-border-hover) !important;
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


html[data-theme="light"] #install-app-container a,
html[data-theme="light"] #install-app-container button,
html[data-theme="light"] #install-app-container [role="button"],
html[data-theme="light"] #install-app-container .stButton > button,
body[data-theme="light"] #install-app-container a,
body[data-theme="light"] #install-app-container button,
body[data-theme="light"] #install-app-container [role="button"],
body[data-theme="light"] #install-app-container .stButton > button,
[data-theme="light"] #install-app-container a,
[data-theme="light"] #install-app-container button,
[data-theme="light"] #install-app-container [role="button"],
[data-theme="light"] #install-app-container .stButton > button {
    background: rgba(241, 245, 249, 0.98) !important;
    color: #334155 !important;
    border-color: rgba(148, 163, 184, 0.78) !important;
}

html[data-theme="dark"] #install-app-container a,
html[data-theme="dark"] #install-app-container button,
html[data-theme="dark"] #install-app-container [role="button"],
html[data-theme="dark"] #install-app-container .stButton > button,
body[data-theme="dark"] #install-app-container a,
body[data-theme="dark"] #install-app-container button,
body[data-theme="dark"] #install-app-container [role="button"],
body[data-theme="dark"] #install-app-container .stButton > button,
[data-theme="dark"] #install-app-container a,
[data-theme="dark"] #install-app-container button,
[data-theme="dark"] #install-app-container [role="button"],
[data-theme="dark"] #install-app-container .stButton > button {
    background: rgba(15, 23, 42, 0.28) !important;
    color: #dbe7ef !important;
    border-color: rgba(45, 212, 191, 0.55) !important;
}

@media (max-width: 900px) {
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""

ACTION_BUTTON_CSS = """
:root {
    --action-primary-bg: #0f766e;
    --action-primary-bg-hover: #115e59;
    --action-primary-border: #0f766e;
    --action-primary-text: #f8fafc;

    --action-secondary-bg: #ffffff;
    --action-secondary-bg-hover: #f8fafc;
    --action-secondary-border: #cbd5e1;
    --action-secondary-border-hover: #94a3b8;
    --action-secondary-text: #0f172a;

    --action-danger-bg: #dc2626;
    --action-danger-bg-hover: #b91c1c;
    --action-danger-border: #dc2626;
    --action-danger-text: #ffffff;

    --action-disabled-bg: #f1f5f9;
    --action-disabled-border: #cbd5e1;
    --action-disabled-text: #94a3b8;

    --action-radius: 14px;
    --action-height: 3.15rem;
    --action-font-weight: 700;
}

html[data-theme="dark"] {
    --action-primary-bg: #14b8a6;
    --action-primary-bg-hover: #0f9f94;
    --action-primary-border: #2dd4bf;
    --action-primary-text: #f8fafc;

    --action-secondary-bg: #1e293b;
    --action-secondary-bg-hover: #334155;
    --action-secondary-border: #475569;
    --action-secondary-border-hover: #64748b;
    --action-secondary-text: #e2e8f0;

    --action-danger-bg: #ef4444;
    --action-danger-bg-hover: #dc2626;
    --action-danger-border: #f87171;
    --action-danger-text: #ffffff;

    --action-disabled-bg: #111827;
    --action-disabled-border: #374151;
    --action-disabled-text: #6b7280;
}

@media (prefers-color-scheme: dark) {
    html:not([data-theme="light"]) {
        --action-primary-bg: #14b8a6;
        --action-primary-bg-hover: #0f9f94;
        --action-primary-border: #2dd4bf;
        --action-primary-text: #f8fafc;

        --action-secondary-bg: #1e293b;
        --action-secondary-bg-hover: #334155;
        --action-secondary-border: #475569;
        --action-secondary-border-hover: #64748b;
        --action-secondary-text: #e2e8f0;

        --action-danger-bg: #ef4444;
        --action-danger-bg-hover: #dc2626;
        --action-danger-border: #f87171;
        --action-danger-text: #ffffff;
    }
}

[class*="st-key-action-primary-"] div.stButton > button,
[class*="st-key-action-primary-"] div[data-testid="stFormSubmitButton"] > button {
    background: var(--action-primary-bg) !important;
    color: var(--action-primary-text) !important;
    border: 1px solid var(--action-primary-border) !important;
    border-radius: var(--action-radius) !important;
    min-height: var(--action-height) !important;
    font-weight: var(--action-font-weight) !important;
    box-shadow: 0 6px 16px rgba(20, 184, 166, 0.18) !important;
}

[class*="st-key-action-primary-"] div.stButton > button:hover,
[class*="st-key-action-primary-"] div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--action-primary-bg-hover) !important;
    color: var(--action-primary-text) !important;
    border-color: var(--action-primary-border) !important;
}

[class*="st-key-action-secondary-"] div.stButton > button,
[class*="st-key-action-secondary-"] div[data-testid="stFormSubmitButton"] > button {
    background: var(--action-secondary-bg) !important;
    color: var(--action-secondary-text) !important;
    border: 1px solid var(--action-secondary-border) !important;
    border-radius: var(--action-radius) !important;
    min-height: var(--action-height) !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

[class*="st-key-action-secondary-"] div.stButton > button:hover,
[class*="st-key-action-secondary-"] div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--action-secondary-bg-hover) !important;
    color: var(--action-secondary-text) !important;
    border-color: var(--action-secondary-border-hover) !important;
}

[class*="st-key-action-danger-"] div.stButton > button,
[class*="st-key-action-danger-"] div[data-testid="stFormSubmitButton"] > button {
    background: var(--action-danger-bg) !important;
    color: var(--action-danger-text) !important;
    border: 1px solid var(--action-danger-border) !important;
    border-radius: var(--action-radius) !important;
    min-height: var(--action-height) !important;
    font-weight: var(--action-font-weight) !important;
}

[class*="st-key-action-danger-"] div.stButton > button:hover,
[class*="st-key-action-danger-"] div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--action-danger-bg-hover) !important;
    color: var(--action-danger-text) !important;
    border-color: var(--action-danger-border) !important;
}

[class*="st-key-action-"] div.stButton > button:disabled,
[class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled {
    background: var(--action-disabled-bg) !important;
    color: var(--action-disabled-text) !important;
    border: 1px solid var(--action-disabled-border) !important;
    opacity: 1 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}


html[data-theme="light"] [class*="st-key-action-"] div.stButton > button:disabled,
html[data-theme="light"] [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled,
body[data-theme="light"] [class*="st-key-action-"] div.stButton > button:disabled,
body[data-theme="light"] [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled,
[data-theme="light"] [class*="st-key-action-"] div.stButton > button:disabled,
[data-theme="light"] [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled {
    background: #f1f5f9 !important;
    color: #94a3b8 !important;
    border-color: #cbd5e1 !important;
}

html[data-theme="dark"] [class*="st-key-action-"] div.stButton > button:disabled,
html[data-theme="dark"] [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled,
body[data-theme="dark"] [class*="st-key-action-"] div.stButton > button:disabled,
body[data-theme="dark"] [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled,
[data-theme="dark"] [class*="st-key-action-"] div.stButton > button:disabled,
[data-theme="dark"] [class*="st-key-action-"] div[data-testid="stFormSubmitButton"] > button:disabled {
    background: #111827 !important;
    color: #6b7280 !important;
    border-color: #374151 !important;
}

.action-hint {
    margin-top: 0.35rem;
    margin-bottom: 0.6rem;
    font-size: 0.92rem;
}
"""


def apply_app_styles():
    st.markdown(
        f"""<style>{APP_BASE_CSS}
{ACTION_BUTTON_CSS}</style>""",
        unsafe_allow_html=True,
    )
