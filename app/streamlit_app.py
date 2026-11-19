import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="ChatSeguro — Detector IA",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "Soobb06/beto-chatseguro"

@st.cache_resource
def cargar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        output_attentions=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = cargar_modelo()

# --- BARRA LATERAL CON MÉTRICAS DEL PROYECTO ---
with st.sidebar:
    st.header("🛡️ ChatSeguro AI")
    st.caption("Track 1 — Human or Machine?")
    st.divider()
    st.subheader("Rendimiento del Modelo")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("F1 Oficial", "93.14%")
    col_m2.metric("Robustez", "90.99%")
    st.markdown("""
    **Arquitectura:** BETO (`bert-base-spanish-wwm-cased`)  
    **Dataset:** AuTexTification 2023 (32,062 textos)  
    **Explicabilidad:** Pesos de atención `[CLS]` de la última capa oculta.
    """)
    st.divider()
    st.info("💡 **Uso responsable:** Herramienta de apoyo para agentes de soporte (*Human-in-the-Loop*).")

# --- INTERFAZ PRINCIPAL ---
st.title("🛡️ ChatSeguro — Detector de Mensajes Humano vs. IA")
st.markdown("Analiza mensajes de atención al cliente en español para detectar respuestas generadas por modelos de lenguaje.")

ejemplos = {
    "✍️ Escribir mensaje personalizado...": "",
    "🧑 Humano 1 (Reclamo de paquete)": "hola buenas tardes oye no me ha llegado mi paquete y ya paso la fecha q decia la app",
    "🤖 Bot 1 (Respuesta estructurada)": "Lamento mucho los inconvenientes que estás experimentando con la entrega de tu pedido. Con gusto revisaré el estado de tu envío de inmediato.",
    "🧑 Humano 2 (Problema con cobro)": "me cobraron dos veces la suscripcion este mes ayudaaa",
    "🤖 Bot 2 (Instrucciones formales)": "Para cancelar tu suscripción, dirígete a Configuración > Suscripciones y selecciona la opción 'Administrar plan'. Estoy aquí para ayudarte.",
    "🧑 Humano 3 (Duda de factura)": "oye disculpa donde puedo descargar mi factura del pedido 4829? urge porfa"
}

col_izq, col_der = st.columns([1.15, 1], gap="large")

with col_izq:
    st.subheader("1. Entrada del mensaje")
    seleccion = st.selectbox("Cargar ejemplo de prueba:", list(ejemplos.keys()))
    mensaje = st.text_area(
        "Mensaje del cliente:",
        value=ejemplos[seleccion],
        height=160,
        placeholder="Escribe o pega aquí el mensaje del chat para analizar..."
    )
    analizar = st.button("🔍 Analizar mensaje", type="primary", use_container_width=True)

with col_der:
    st.subheader("2. Diagnóstico y Explicabilidad")
    if analizar:
        if not mensaje or not mensaje.strip():
            st.warning("⚠️ Por favor ingresa un mensaje antes de analizar.")
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
                st.error(f"### 🤖 GENERADO POR MÁQUINA (Bot)\n**Confianza del modelo:** {confianza:.2f}%")
            else:
                st.success(f"### 🧑 ESCRITO POR HUMANO\n**Confianza del modelo:** {confianza:.2f}%")

            # Alerta preventiva para textos muy cortos o duda del modelo
            if len(palabras_usuario) < 4 or confianza < 75.0:
                st.warning(
                    "⚠️ **Baja entropía / Confianza moderada:** El mensaje es muy corto o ambiguo. "
                    "Se recomienda verificación por un agente humano."
                )

            # Desglose de probabilidades
            c1, c2 = st.columns(2)
            c1.metric("Probabilidad Humano", f"{prob_humano:.1f}%")
            c2.metric("Probabilidad Máquina", f"{prob_bot:.1f}%")

            st.divider()

            # Explicabilidad mediante atención
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
            for palabra, peso in top_5:
                pct = (peso / total_top) * 100
                st.progress(min(int(pct), 100), text=f"'{palabra}' — {pct:.1f}%")
    else:
        st.info("👈 Selecciona un ejemplo o escribe un mensaje a la izquierda y presiona **Analizar mensaje**.")
