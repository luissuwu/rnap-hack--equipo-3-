# ChatSeguro — Human or Machine? Detector (Track 1)

AI-generated text detection system built for **ChatSeguro** customer support to distinguish human-written messages from LLM-generated responses in Spanish.

## Live Demo & Model Weights
* **Live Web App (Streamlit Cloud):** [https://ifvdyypej4mw9ykfindrkr.streamlit.app/]
* **Fine-Tuned Model on Hugging Face Hub:** [`Soobb06/beto-chatseguro`](https://huggingface.co/Soobb06/beto-chatseguro)

## Dataset & Methodology
* **Primary Dataset:** *AuTexTification 2023* (Spanish detection subtask, 32,062 texts), strictly split prior to training into **Train (60% — 19,236)**, **Validation (20% — 6,413)**, and **Test (20% — 6,413)**.
* **Robustness Benchmark:** Additional evaluation on **100 modern customer-support texts** (50 colloquial human messages + 50 modern chatbot-style responses).

## Results Summary

| Model | Evaluation Set | F1 Score (Primary) | Accuracy |
| :--- | :--- | :---: | :---: |
| Baseline (TF-IDF + Logistic Regression) | Official Test (6,413) | 0.7874 | 0.7875 |
| Pretrained Zero-Shot (mDeBERTa-v3) | Test Sample (100) | 0.5123 | 0.5400 |
| **Fine-Tuned BETO (`bert-base-spanish-wwm-cased`)** | **Official Test (6,413)** | **0.9314** | **0.9314** |
| **Fine-Tuned BETO (`bert-base-spanish-wwm-cased`)** | **Modern Robustness (100)** | **0.9099** | **0.9100** |

## Features & Explainability
* Binary classification verdict (**Human** vs. **Machine**) with calibrated confidence score (`0% - 100%`).
* Token-level interpretability extracted from BETO's final self-attention layer (`[CLS]` attention weights), highlighting the top 5 most influential words.

## Local Execution (Fallback)
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# Or using the Gradio backup interface:
python app/app.py


## Team (`rnap-hack--equipo-3-`)
* **Barcenas Sánchez Alondra (2251431)** — Dataset management and `README.md` documentation
* **Campos Rivera Luis Alberto (2252149)** — Fine-tuning pipeline, Hugging Face Hub upload, and Streamlit deployment
* **Cavazos López Noah Demian (2251426)** — 100-text robustness validation and `PROGRESS.md`
* **Garcia Hernández César Angel (2251075)** — Hyperparameter analysis and Section 6 conclusions
* **Morin Perales Emiliano (2251525)** — Error case inspection and Gradio backup verification
* **Tamez García Ángela Regina (2251246)** — Live demo preparation and repository quality check
