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

    --install-button-bg: rgba(245, 247, 251, 0.98);
    --install-button-bg-hover: rgba(236, 241, 247, 0.98);
    --install-button-border: rgba(148, 163, 184, 0.62);
    --install-button-border-hover: rgba(100, 116, 139, 0.72);
    --install-button-text: #334155;

    --custom-panel-bg: linear-gradient(180deg, rgba(242, 246, 251, 0.98) 0%, rgba(235, 241, 248, 0.98) 100%);
    --custom-panel-border: #bcc8d8;
    --custom-panel-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
    --custom-panel-text: #14213d;
    --custom-panel-muted: #52607a;

    --summary-card-bg: linear-gradient(180deg, rgba(247, 249, 252, 0.99) 0%, rgba(239, 243, 248, 0.99) 100%);
    --summary-card-border: #cfd8e3;
    --summary-card-accent: #3fb27f;
    --summary-card-shadow: 0 10px 20px rgba(15, 23, 42, 0.04);
    --summary-label-text: #52607a;
    --summary-value-text: #162033;

    --team-card-bg: linear-gradient(180deg, rgba(249, 251, 254, 0.99) 0%, rgba(242, 246, 250, 0.99) 100%);
    --team-card-border: #cfd8e3;
    --team-card-shadow: 0 10px 20px rgba(15, 23, 42, 0.045);
    --team-card-divider: #dde5ee;
    --team-card-badge-bg: #facc15;
    --team-card-badge-text: #3f2f00;
    --team-card-text: #172033;
    --team-card-muted-text: #52607a;
    --team-card-title-text: #24324a;
    --team-stats-bg: #f1f5f9;
    --team-stats-text: #4a5872;
    --team-player-divider: #e4eaf2;
    --team-player-pos-bg: #e9eef5;
    --team-player-pos-text: #51617d;
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
[data-theme="light"] .theme-panel,
.stApp[data-theme="light"] .theme-panel {
    background: linear-gradient(180deg, rgba(238, 244, 251, 0.96) 0%, rgba(230, 238, 248, 0.96) 100%) !important;
    border-color: #b9c8dc !important;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045) !important;
    color: #14213d !important;
}

html[data-theme="light"] .theme-panel__title,
html[data-theme="light"] .theme-panel__line,
html[data-theme="light"] .theme-panel__strong,
body[data-theme="light"] .theme-panel__title,
body[data-theme="light"] .theme-panel__line,
body[data-theme="light"] .theme-panel__strong,
[data-theme="light"] .theme-panel__title,
[data-theme="light"] .theme-panel__line,
[data-theme="light"] .theme-panel__strong,
.stApp[data-theme="light"] .theme-panel__title,
.stApp[data-theme="light"] .theme-panel__line,
.stApp[data-theme="light"] .theme-panel__strong {
    color: #14213d !important;
}

html[data-theme="light"] .theme-panel__label,
body[data-theme="light"] .theme-panel__label,
[data-theme="light"] .theme-panel__label,
.stApp[data-theme="light"] .theme-panel__label {
    color: #52607a !important;
}

html[data-theme="dark"] .theme-panel,
body[data-theme="dark"] .theme-panel,
[data-theme="dark"] .theme-panel,
.stApp[data-theme="dark"] .theme-panel {
    background: linear-gradient(180deg, rgba(8, 18, 40, 0.82) 0%, rgba(9, 17, 34, 0.78) 100%) !important;
    border-color: #29456b !important;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18) !important;
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
[data-theme="dark"] .theme-panel__strong,
.stApp[data-theme="dark"] .theme-panel__title,
.stApp[data-theme="dark"] .theme-panel__line,
.stApp[data-theme="dark"] .theme-panel__strong {
    color: #f8fafc !important;
}

html[data-theme="dark"] .theme-panel__label,
body[data-theme="dark"] .theme-panel__label,
[data-theme="dark"] .theme-panel__label,
.stApp[data-theme="dark"] .theme-panel__label {
    color: #c9d6ea !important;
}

html[data-theme="light"] .summary-card,
body[data-theme="light"] .summary-card,
[data-theme="light"] .summary-card,
.stApp[data-theme="light"] .summary-card {
    background: linear-gradient(180deg, rgba(245, 248, 252, 0.98) 0%, rgba(236, 241, 247, 0.98) 100%) !important;
    border-color: #c9d5e3 !important;
    border-top-color: #3fb27f !important;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05) !important;
}

html[data-theme="light"] .summary-label,
body[data-theme="light"] .summary-label,
[data-theme="light"] .summary-label,
.stApp[data-theme="light"] .summary-label {
    color: #52607a !important;
}

html[data-theme="light"] .summary-value,
body[data-theme="light"] .summary-value,
[data-theme="light"] .summary-value,
.stApp[data-theme="light"] .summary-value {
    color: #162033 !important;
}

html[data-theme="dark"] .summary-card,
body[data-theme="dark"] .summary-card,
[data-theme="dark"] .summary-card,
.stApp[data-theme="dark"] .summary-card {
    background: linear-gradient(180deg, rgba(12, 21, 39, 0.94) 0%, rgba(18, 28, 48, 0.9) 100%) !important;
    border-color: #29456b !important;
    border-top-color: #34b67a !important;
    box-shadow: 0 8px 22px rgba(0,0,0,0.14) !important;
}

html[data-theme="dark"] .summary-label,
body[data-theme="dark"] .summary-label,
[data-theme="dark"] .summary-label,
.stApp[data-theme="dark"] .summary-label {
    color: #a8c8f2 !important;
}

html[data-theme="dark"] .summary-value,
body[data-theme="dark"] .summary-value,
[data-theme="dark"] .summary-value,
.stApp[data-theme="dark"] .summary-value {
    color: #eef4ff !important;
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
    color: var(--team-card-title-text);
}

.team-card__odd {
    background: var(--team-card-badge-bg);
    color: var(--team-card-badge-text);
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 700;
    white-space: nowrap;
}

.team-card__stats {
    background: var(--team-stats-bg);
    color: var(--team-stats-text);
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
    color: var(--team-card-text);
}

.team-card__player-pos {
    background: var(--team-player-pos-bg);
    color: var(--team-player-pos-text);
    padding: 2px 6px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
}

.team-card__metrics {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    color: var(--team-card-muted-text);
}


html[data-theme="light"] .team-card,
body[data-theme="light"] .team-card,
[data-theme="light"] .team-card,
.stApp[data-theme="light"] .team-card {
    background: linear-gradient(180deg, rgba(249, 251, 254, 0.99) 0%, rgba(242, 246, 250, 0.99) 100%) !important;
    border-color: #cfd8e3 !important;
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.045) !important;
}

html[data-theme="light"] .team-card__title,
html[data-theme="light"] .team-card__player-name,
body[data-theme="light"] .team-card__title,
body[data-theme="light"] .team-card__player-name,
[data-theme="light"] .team-card__title,
[data-theme="light"] .team-card__player-name,
.stApp[data-theme="light"] .team-card__title,
.stApp[data-theme="light"] .team-card__player-name {
    color: #20283a !important;
}

html[data-theme="light"] .team-card__stats,
html[data-theme="light"] .team-card__metrics,
body[data-theme="light"] .team-card__stats,
body[data-theme="light"] .team-card__metrics,
[data-theme="light"] .team-card__stats,
[data-theme="light"] .team-card__metrics,
.stApp[data-theme="light"] .team-card__stats,
.stApp[data-theme="light"] .team-card__metrics {
    background: #f1f5f9 !important;
    color: #4a5872 !important;
}

html[data-theme="light"] .team-card__player-pos,
body[data-theme="light"] .team-card__player-pos,
[data-theme="light"] .team-card__player-pos,
.stApp[data-theme="light"] .team-card__player-pos {
    background: #e5ebf2 !important;
    color: #42506a !important;
}

html[data-theme="light"] .team-card__header,
body[data-theme="light"] .team-card__header,
[data-theme="light"] .team-card__header,
.stApp[data-theme="light"] .team-card__header {
    border-bottom-color: #dde5ee !important;
}

html[data-theme="light"] .team-card__player-row,
body[data-theme="light"] .team-card__player-row,
[data-theme="light"] .team-card__player-row,
.stApp[data-theme="light"] .team-card__player-row {
    border-bottom-color: #e4eaf2 !important;
}

html[data-theme="light"] .team-card__odd,
body[data-theme="light"] .team-card__odd,
[data-theme="light"] .team-card__odd,
.stApp[data-theme="light"] .team-card__odd {
    background: #f7d21d !important;
    color: #4a3700 !important;
}

html[data-theme="light"] .theme-panel--summary,
body[data-theme="light"] .theme-panel--summary,
[data-theme="light"] .theme-panel--summary,
.stApp[data-theme="light"] .theme-panel--summary {
    background: linear-gradient(180deg, rgba(242, 246, 251, 0.99) 0%, rgba(235, 241, 248, 0.99) 100%) !important;
    border-color: #bcc8d8 !important;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035) !important;
}

html[data-theme="light"] .theme-panel--summary .theme-panel__title,
html[data-theme="light"] .theme-panel--summary .theme-panel__line,
html[data-theme="light"] .theme-panel--summary .theme-panel__strong,
body[data-theme="light"] .theme-panel--summary .theme-panel__title,
body[data-theme="light"] .theme-panel--summary .theme-panel__line,
body[data-theme="light"] .theme-panel--summary .theme-panel__strong,
[data-theme="light"] .theme-panel--summary .theme-panel__title,
[data-theme="light"] .theme-panel--summary .theme-panel__line,
[data-theme="light"] .theme-panel--summary .theme-panel__strong,
.stApp[data-theme="light"] .theme-panel--summary .theme-panel__title,
.stApp[data-theme="light"] .theme-panel--summary .theme-panel__line,
.stApp[data-theme="light"] .theme-panel--summary .theme-panel__strong {
    color: #23324d !important;
}

html[data-theme="light"] .theme-panel--summary .theme-panel__label,
body[data-theme="light"] .theme-panel--summary .theme-panel__label,
[data-theme="light"] .theme-panel--summary .theme-panel__label,
.stApp[data-theme="light"] .theme-panel--summary .theme-panel__label {
    color: #64748b !important;
}

html[data-theme="dark"] .team-card,
body[data-theme="dark"] .team-card,
[data-theme="dark"] .team-card,
.stApp[data-theme="dark"] .team-card {
    background: linear-gradient(180deg, rgba(18, 24, 39, 0.98) 0%, rgba(12, 18, 31, 0.98) 100%) !important;
    border-color: #33445e !important;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16) !important;
}

html[data-theme="dark"] .team-card__title,
html[data-theme="dark"] .team-card__player-name,
body[data-theme="dark"] .team-card__title,
body[data-theme="dark"] .team-card__player-name,
[data-theme="dark"] .team-card__title,
[data-theme="dark"] .team-card__player-name,
.stApp[data-theme="dark"] .team-card__title,
.stApp[data-theme="dark"] .team-card__player-name {
    color: #f8fbff !important;
}

html[data-theme="dark"] .team-card__stats,
html[data-theme="dark"] .team-card__metrics,
body[data-theme="dark"] .team-card__stats,
body[data-theme="dark"] .team-card__metrics,
[data-theme="dark"] .team-card__stats,
[data-theme="dark"] .team-card__metrics,
.stApp[data-theme="dark"] .team-card__stats,
.stApp[data-theme="dark"] .team-card__metrics {
    color: #d7e4f7 !important;
}

html[data-theme="dark"] .team-card__player-pos,
body[data-theme="dark"] .team-card__player-pos,
[data-theme="dark"] .team-card__player-pos,
.stApp[data-theme="dark"] .team-card__player-pos {
    background: rgba(148, 163, 184, 0.12) !important;
    color: #d9e5f7 !important;
}

html[data-theme="dark"] .theme-panel--summary,
body[data-theme="dark"] .theme-panel--summary,
[data-theme="dark"] .theme-panel--summary,
.stApp[data-theme="dark"] .theme-panel--summary {
    background: linear-gradient(180deg, rgba(10, 21, 44, 0.88) 0%, rgba(7, 17, 37, 0.84) 100%) !important;
    border-color: #2b4569 !important;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16) !important;
}

html[data-theme="dark"] .theme-panel--summary .theme-panel__title,
html[data-theme="dark"] .theme-panel--summary .theme-panel__line,
html[data-theme="dark"] .theme-panel--summary .theme-panel__strong,
body[data-theme="dark"] .theme-panel--summary .theme-panel__title,
body[data-theme="dark"] .theme-panel--summary .theme-panel__line,
body[data-theme="dark"] .theme-panel--summary .theme-panel__strong,
[data-theme="dark"] .theme-panel--summary .theme-panel__title,
[data-theme="dark"] .theme-panel--summary .theme-panel__line,
[data-theme="dark"] .theme-panel--summary .theme-panel__strong,
.stApp[data-theme="dark"] .theme-panel--summary .theme-panel__title,
.stApp[data-theme="dark"] .theme-panel--summary .theme-panel__line,
.stApp[data-theme="dark"] .theme-panel--summary .theme-panel__strong {
    color: #f8fafc !important;
}

html[data-theme="dark"] .theme-panel--summary .theme-panel__label,
body[data-theme="dark"] .theme-panel--summary .theme-panel__label,
[data-theme="dark"] .theme-panel--summary .theme-panel__label,
.stApp[data-theme="dark"] .theme-panel--summary .theme-panel__label {
    color: #c9d6ea !important;
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

    --install-button-bg: rgba(241, 245, 249, 0.96);
    --install-button-bg-hover: rgba(226, 232, 240, 0.96);
    --install-button-border: rgba(148, 163, 184, 0.32);
    --install-button-border-hover: rgba(148, 163, 184, 0.48);
    --install-button-text: #475569;

    --custom-panel-bg: linear-gradient(180deg, rgba(8, 18, 40, 0.82) 0%, rgba(9, 17, 34, 0.78) 100%);
    --custom-panel-border: #29456b;
    --custom-panel-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
    --custom-panel-text: #f8fafc;
    --custom-panel-muted: #c9d6ea;

    --summary-card-bg: linear-gradient(180deg, rgba(12, 21, 39, 0.94) 0%, rgba(18, 28, 48, 0.9) 100%);
    --summary-card-border: #29456b;
    --summary-card-accent: #34b67a;
    --summary-card-shadow: 0 8px 22px rgba(0,0,0,0.14);
    --summary-label-text: #a8c8f2;
    --summary-value-text: #eef4ff;

    --team-card-bg: linear-gradient(180deg, rgba(18, 24, 39, 0.98) 0%, rgba(12, 18, 31, 0.98) 100%);
    --team-card-border: #33445e;
    --team-card-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
    --team-card-divider: #314159;
    --team-card-badge-bg: #facc15;
    --team-card-badge-text: #3f2f00;
    --team-card-text: #eef4ff;
    --team-card-muted-text: #b8c7dc;
    --team-card-title-text: #f8fbff;
    --team-stats-bg: rgba(255, 255, 255, 0.04);
    --team-stats-text: #d7e4f7;
    --team-player-divider: rgba(148, 163, 184, 0.18);
    --team-player-pos-bg: rgba(148, 163, 184, 0.12);
    --team-player-pos-text: #d9e5f7;
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

        --team-card-bg: linear-gradient(180deg, rgba(18, 24, 39, 0.98) 0%, rgba(12, 18, 31, 0.98) 100%);
        --team-card-border: #33445e;
        --team-card-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
        --team-card-divider: #314159;
        --team-card-badge-bg: #facc15;
        --team-card-badge-text: #3f2f00;
        --team-card-text: #eef4ff;
        --team-card-muted-text: #b8c7dc;
        --team-card-title-text: #f8fbff;
        --team-stats-bg: rgba(255, 255, 255, 0.04);
        --team-stats-text: #d7e4f7;
        --team-player-divider: rgba(148, 163, 184, 0.18);
        --team-player-pos-bg: rgba(148, 163, 184, 0.12);
        --team-player-pos-text: #d9e5f7;
    }
}

#install-app-container a,
#install-app-container button,
#install-app-container [role="button"],
#install-app-container .stButton > button {
    background: var(--install-button-bg) !important;
    color: var(--install-button-text) !important;
    border: 1px solid var(--install-button-border) !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
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
    background: rgba(245, 247, 251, 0.98) !important;
    color: #334155 !important;
    border-color: rgba(148, 163, 184, 0.62) !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
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
    background: rgba(241, 245, 249, 0.96) !important;
    color: #475569 !important;
    border-color: rgba(148, 163, 184, 0.32) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

@media (max-width: 900px) {
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""

ACTION_BUTTON_CSS = """
:root {
    --action-primary-bg: #2db7aa;
    --action-primary-bg-hover: #24a89d;
    --action-primary-border: #2aa99d;
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
    --action-primary-bg: #2db7aa;
    --action-primary-bg-hover: #24a89d;
    --action-primary-border: #31c4b5;
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
        --action-primary-bg: #2db7aa;
        --action-primary-bg-hover: #24a89d;
        --action-primary-border: #31c4b5;
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
    box-shadow: 0 4px 12px rgba(45, 183, 170, 0.16) !important;
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
    background: #eef3f8 !important;
    color: #8fa0b8 !important;
    border-color: #c7d3e0 !important;
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
