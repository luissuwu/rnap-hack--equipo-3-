import html
import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="ChatSeguro AI — Detector Humano vs. IA", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.stApp { background: radial-gradient(circle at top right, #0F2342 0%, #080F1E 55%, #060B16 100%); color: #F8FAFC; }
[data-testid="stSidebar"] { background-color: #0B1426; border-right: 1px solid #1E3254; }
header[data-testid="stHeader"] { background: transparent; }
.hero-banner { background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,182,212,0.10) 100%); border: 1px solid rgba(16,185,129,0.35); border-radius: 16px; padding: 1.3rem 1.7rem; margin-bottom: 1.4rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
.hero-left { display: flex; align-items: center; gap: 1rem; }
.hero-title { font-size: 1.8rem; font-weight: 800; margin: 0; background: linear-gradient(90deg, #FFFFFF 0%, #A7F3D0 60%, #67E8F9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2; }
.hero-subtitle { color: #94A3B8; font-size: 0.95rem; margin: 0.2rem 0 0 0; }
.status-pill { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.45); color: #34D399; padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600; display: flex; align-items: center; gap: 0.45rem; }
.status-dot { width: 8px; height: 8px; background-color: #10B981; border-radius: 50%; display: inline-block; }
.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div { background-color: #0E1A2F !important; border: 1px solid #233B63 !important; border-radius: 12px !important; color: #F1F5F9 !important; }
.stTextArea textarea:focus { border-color: #10B981 !important; box-shadow: 0 0 0 1px #10B981 !important; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #10B981 0%, #06B6D4 100%) !important; color: #04101E !important; font-weight: 700 !important; font-size: 1rem !important; border: none !important; border-radius: 12px !important; padding: 0.65rem 1.2rem !important; box-shadow: 0 4px 18px rgba(16,185,129,0.25) !important; }
.verdict-human { background: linear-gradient(135deg, rgba(16,185,129,0.18) 0%, rgba(6,95,70,0.25) 100%); border: 1px solid #10B981; border-radius: 14px; padding: 1.15rem 1.35rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem; }
.verdict-bot { background: linear-gradient(135deg, rgba(244,63,94,0.18) 0%, rgba(136,19,55,0.25) 100%); border: 1px solid #F43F5E; border-radius: 14px; padding: 1.15rem 1.35rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 1rem; }
.verdict-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; margin-bottom: 0.2rem; }
.verdict-title { font-size: 1.38rem; font-weight: 800; margin: 0; color: #FFFFFF; }
.verdict-conf { font-size: 0.92rem; margin-top: 0.2rem; color: #E2E8F0; }
.alert-amber { background: rgba(245,158,11,0.14); border: 1px solid rgba(245,158,11,0.5); border-radius: 12px; padding: 0.85rem 1rem; margin-bottom: 1rem; display: flex; align-items: flex-start; gap: 0.75rem; color: #FDE68A; font-size: 0.88rem; line-height: 1.45; }
.alert-info { background: rgba(6,182,212,0.12); border: 1px solid rgba(6,182,212,0.4); border-radius: 12px; padding: 0.85rem 0.95rem; display: flex; align-items: flex-start; gap: 0.7rem; color: #CFFAFE; font-size: 0.83rem; line-height: 1.45; }
.prob-box { background-color: #0E1A2F; border: 1px solid #1E3254; border-radius: 12px; padding: 0.85rem 1rem; text-align: center; }
.prob-title { font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.2rem; }
.prob-human { font-size: 1.6rem; font-weight: 800; color: #34D399; }
.prob-bot { font-size: 1.6rem; font-weight: 800; color: #FB7185; }
.token-row { margin-bottom: 0.65rem; }
.token-header { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 0.25rem; color: #E2E8F0; }
.token-word { font-weight: 700; color: #67E8F9; font-family: monospace; font-size: 0.95rem; }
.token-bar-bg { width: 100%; height: 8px; background-color: #162642; border-radius: 999px; overflow: hidden; }
.token-bar-fill { height: 100%; background: linear-gradient(90deg, #10B981 0%, #06B6D4 100%); border-radius: 999px; }
.sidebar-card { background-color: #111E36; border: 1px solid #1E3254; border-radius: 12px; padding: 0.8rem; text-align: center; }
.sidebar-num { font-size: 1.4rem; font-weight: 800; color: #34D399; }
.sidebar-lbl { font-size: 0.78rem; color: #94A3B8; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

MODEL_PATH = "Soobb06/beto-chatseguro"

@st.cache_resource
def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
    model.eval()
    return tokenizer, model

tokenizer, model = cargar_modelo()

LOGO_SVG = '<svg width="44" height="44" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="shieldGrad" x1="6" y1="4" x2="42" y2="44" gradientUnits="userSpaceOnUse"><stop offset="0%" stop-color="#10B981"/><stop offset="100%" stop-color="#06B6D4"/></linearGradient></defs><path d="M24 4L8 10.5V22.5C8 33.1 14.8 42.8 24 46C33.2 42.8 40 33.1 40 22.5V10.5L24 4Z" fill="url(#shieldGrad)" fill-opacity="0.18" stroke="url(#shieldGrad)" stroke-width="2.5"/><circle cx="24" cy="21" r="4.5" fill="#34D399"/><path d="M16 31C16 27.5 19.5 25.5 24 25.5C28.5 25.5 32 27.5 32 31" stroke="#67E8F9" stroke-width="2.5" stroke-linecap="round"/><circle cx="34" cy="15" r="2" fill="#67E8F9"/><circle cx="14" cy="15" r="2" fill="#34D399"/></svg>'
ICON_HUMAN = '<svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><polyline points="16 11 18 13 22 9"/></svg>'
ICON_BOT = '<svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#FB7185" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8.01" y2="16"/><line x1="16" y1="16" x2="16.01" y2="16"/></svg>'
ICON_WARN = '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
ICON_INFO = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22D3EE" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0; margin-top:2px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'

with st.sidebar:
    st.markdown(f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">{LOGO_SVG}<div><div style="font-weight:800; font-size:1.15rem; color:#FFFFFF;">ChatSeguro AI</div><div style="font-size:0.78rem; color:#67E8F9;">v1.0 • Spanish NLP Security</div></div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Rendimiento Validado**")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown('<div class="sidebar-card"><div class="sidebar-num">93.14%</div><div class="sidebar-lbl">F1 Oficial</div></div>', unsafe_allow_html=True)
    with c_m2:
        st.markdown('<div class="sidebar-card"><div class="sidebar-num">90.99%</div><div class="sidebar-lbl">Robustez</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="background:#0E1A2F; border:1px solid #1E3254; border-radius:12px; padding:0.9rem; font-size:0.84rem; color:#CBD5E1; line-height:1.55;"><div style="color:#67E8F9; font-weight:700; margin-bottom:4px;">Especificaciones Técnicas</div>• <b>Motor:</b> BETO (<code>bert-base-spanish-wwm-cased</code>)<br>• <b>Dataset:</b> AuTexTification 2023 (32,062 textos)<br>• <b>Explicabilidad:</b> Autoatención <code>[CLS]</code> de última capa</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="alert-info">{ICON_INFO}<div><b>Human-in-the-Loop:</b> Sistema preventivo de apoyo para agentes; nunca bloquea clientes automáticamente.</div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="hero-banner"><div class="hero-left">{LOGO_SVG}<div><h1 class="hero-title">ChatSeguro AI — Detector Humano vs. Máquina</h1><p class="hero-subtitle">Triaje inteligente en tiempo real para soporte al cliente en español</p></div></div><div class="status-pill"><span class="status-dot"></span>En Vivo • Soobb06/beto-chatseguro</div></div>', unsafe_allow_html=True)

ejemplos = {
    "Escribir mensaje personalizado...": "",
    "Caso Humano 1 — Reclamo de paquete": "hola buenas tardes oye no me ha llegado mi paquete y ya paso la fecha q decia la app",
    "Caso Bot 1 — Respuesta estructurada": "Lamento mucho los inconvenientes que estás experimentando con la entrega de tu pedido. Con gusto revisaré el estado de tu envío de inmediato.",
    "Caso Humano 2 — Problema con cobro": "me cobraron dos veces la suscripcion este mes ayudaaa",
    "Caso Bot 2 — Instrucciones formales": "Para cancelar tu suscripción, dirígete a Configuración > Suscripciones y selecciona la opción 'Administrar plan'. Estoy aquí para ayudarte.",
    "Caso Humano 3 — Duda de factura": "oye disculpa donde puedo descargar mi factura del pedido 4829? urge porfa"
}

col_izq, col_der = st.columns([1.1, 1], gap="large")

with col_izq:
    st.markdown("#### 1. Mensaje Entrante del Chat")
    seleccion = st.selectbox("Cargar escenario de prueba:", list(ejemplos.keys()))
    mensaje = st.text_area("Contenido del mensaje:", value=ejemplos[seleccion], height=175, placeholder="Escribe o pega aquí el mensaje del cliente para analizar...")
    analizar = st.button("Analizar Autenticidad del Mensaje", type="primary", use_container_width=True)

with col_der:
    st.markdown("#### 2. Diagnóstico y Explicabilidad")
    if analizar:
        if not mensaje or not mensaje.strip():
            st.markdown(f'<div class="alert-amber">{ICON_WARN}<div>Por favor ingresa un mensaje en el cuadro de texto antes de analizar.</div></div>', unsafe_allow_html=True)
        else:
            palabras_usuario = mensaje.strip().split()
            inputs = tokenizer(mensaje, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

            prob_humano = float(probs[0]) * 100
            prob_bot = float(probs[1]) * 100
            clase_pred = int(np.argmax(probs))
            confianza = max(prob_humano, prob_bot)

            if clase_pred == 1:
                st.markdown(f'<div class="verdict-bot">{ICON_BOT}<div><div class="verdict-label" style="color:#FDA4AF;">Alerta de Automatización</div><div class="verdict-title">GENERADO POR MÁQUINA (Bot)</div><div class="verdict-conf">Nivel de confianza del modelo: <b>{confianza:.2f}%</b></div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-human">{ICON_HUMAN}<div><div class="verdict-label" style="color:#6EE7B7;">Cliente Verificado</div><div class="verdict-title">ESCRITO POR HUMANO</div><div class="verdict-conf">Nivel de confianza del modelo: <b>{confianza:.2f}%</b></div></div></div>', unsafe_allow_html=True)

            if len(palabras_usuario) < 4 or confianza < 75.0:
                st.markdown(f'<div class="alert-amber">{ICON_WARN}<div><b>Baja entropía / Confianza moderada:</b> El mensaje es muy corto o ambiguo. Se recomienda verificación por un agente humano.</div></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="prob-box"><div class="prob-title">Probabilidad Humano</div><div class="prob-human">{prob_humano:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="prob-box"><div class="prob-title">Probabilidad Máquina</div><div class="prob-bot">{prob_bot:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            ultima_atencion = outputs.attentions[-1][0]
            pesos_cls = ultima_atencion[:, 0, :].mean(dim=0).numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            palabras_pesos = []
            for tok, peso in zip(tokens, pesos_cls):
                if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                if tok.startswith("##") and palabras_pesos:
                    palabras_pesos[-1][0] += tok[2:]
                    palabras_pesos[-1][1] += float(peso)
                else:
                    palabras_pesos.append([tok, float(peso)])

            palabras_pesos.sort(key=lambda x: x[1], reverse=True)
            top_5 = palabras_pesos[:5]
            total_top = sum(p[1] for p in top_5) or 1.0

            st.markdown("**Palabras con mayor peso (Atención BETO):**")
            barras_html = ""
            for palabra, peso in top_5:
                pct = (peso / total_top) * 100
                palabra_segura = html.escape(palabra)
                barras_html += f'<div class="token-row"><div class="token-header"><span class="token-word">\'{palabra_segura}\'</span><span>{pct:.1f}%</span></div><div class="token-bar-bg"><div class="token-bar-fill" style="width: {min(pct, 100):.1f}%;"></div></div></div>'
            st.markdown(barras_html, unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#0E1A2F; border:1px dashed #2A4570; border-radius:14px; padding:2.2rem 1.5rem; text-align:center; color:#94A3B8; margin-top:0.5rem;"><div style="margin-bottom:0.6rem;">{LOGO_SVG}</div><div style="font-weight:700; color:#E2E8F0; margin-bottom:0.3rem;">Listo para analizar</div><div style="font-size:0.9rem;">Selecciona un escenario de prueba o escribe un mensaje a la izquierda y presiona <b>Analizar Autenticidad del Mensaje</b>.</div></div>', unsafe_allow_html=True)

# FIN DEL ARCHIVO
