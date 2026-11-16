import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "dccuchile/bert-base-spanish-wwm-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    output_attentions=True
).to(device)
model.eval()

def analizar_mensaje(mensaje):
    if not mensaje or not mensaje.strip():
        return "Ingresa un mensaje", "0.00%", "Sin datos"

    inputs = tokenizer(mensaje, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    clase_pred = int(np.argmax(probs))
    confianza = float(probs[clase_pred]) * 100
    veredicto = "GENERADO POR MÁQUINA (Bot)" if clase_pred == 1 else "ESCRITO POR HUMANO"

    ultima_atencion = outputs.attentions[-1][0]
    pesos_cls = ultima_atencion[:, 0, :].mean(dim=0).cpu().numpy()
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
    explicacion = "\n".join([f"- {p}: {(w/total_top)*100:.1f}%" for p, w in top_5])

    return veredicto, f"{confianza:.2f}%", explicacion

demo = gr.Interface(
    fn=analizar_mensaje,
    inputs=gr.Textbox(lines=5, label="Mensaje del cliente (ChatSeguro)"),
    outputs=[
        gr.Textbox(label="Veredicto"),
        gr.Textbox(label="Confianza"),
        gr.Textbox(label="Palabras con mayor peso (Atención)", lines=5)
    ],
    title="ChatSeguro - Human or Machine Detector v1"
)

if __name__ == "__main__":
    demo.launch()
