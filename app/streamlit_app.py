import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="ChatSeguro — Detector Human or Machine", page_icon="🛡️")

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

st.title("🛡️ ChatSeguro — Detector Human or Machine (v1.0)")
st.write("Clasificador basado en **BETO ajustado** para detectar mensajes generados por IA en soporte al cliente.")

ejemplos = {
    "Escribir mensaje personalizado...": "",
    "Ejemplo Humano 1": "hola buenas tardes oye no me ha llegado mi paquete y ya paso la fecha q decia la app",
    "Ejemplo IA 1": "Lamento mucho los inconvenientes que estás experimentando con la entrega de tu pedido. Con gusto revisaré el estado de tu envío de inmediato.",
    "Ejemplo Humano 2": "me cobraron dos veces la suscripcion este mes ayudaaa",
    "Ejemplo IA 2": "Para cancelar tu suscripción, dirígete a Configuración > Suscripciones y selecciona la opción 'Administrar plan'. Estoy aquí para ayudarte."
}

seleccion = st.selectbox("Probar con un ejemplo rápido:", list(ejemplos.keys()))
mensaje = st.text_area(
    "Mensaje del cliente (ChatSeguro):",
    value=ejemplos[seleccion],
    height=130,
    placeholder="Escribe o pega aquí el mensaje del chat..."
)

if st.button("Analizar mensaje", type="primary"):
    if not mensaje or not mensaje.strip():
        st.warning("Ingresa un mensaje para analizar.")
    else:
        inputs = tokenizer(mensaje, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

        clase_pred = int(np.argmax(probs))
        confianza = float(probs[clase_pred]) * 100

        col1, col2 = st.columns(2)
        if clase_pred == 1:
            col1.error("🤖 GENERADO POR MÁQUINA (Bot)")
        else:
            col1.success("🧑 ESCRITO POR HUMANO")
        col2.metric("Nivel de Confianza", f"{confianza:.2f}%")

        # Explicabilidad: pesos de atención de la última capa respecto a [CLS]
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

        st.subheader("Palabras / señales con mayor peso (Atención BETO)")
        for palabra, peso in top_5:
            st.write(f"- **'{palabra}'** — peso relativo: `{(peso / total_top) * 100:.1f}%`")
