## CP4 2026-11-12
**What we have done:** We built and connected the v1 Gradio interface displaying the classification verdict, confidence percentage (0-100%), and top weighted words extracted from BETO's attention layer. We also uploaded `app/app.py` and `requirements.txt` to the repository.
**What we haven't done yet:** Deploy the final public link on Hugging Face Spaces and complete Section 6 of the technical report.
**Next step:** Test the interface with hidden prompts, publish the final deployment, and prepare the live demo in English for Phase 3.
**Who did what:** Luis (Gradio interface v1 and model integration), Alondra (Repository structure and `requirements.txt`), Noah (Error analysis and documentation).

## CP3 2026-11-12
**What we have done:** We fine-tuned the Spanish Transformer model BETO (`dccuchile/bert-base-spanish-wwm-cased`) on the training set and evaluated it on the full test set using F1 Score and Accuracy. We completed Sections 4 and 5 in `notebooks/entrenamiento.ipynb` including comparison charts and 5 error analysis cases.
**What we haven't done yet:** Connect the fine-tuned model to the v1 user interface and upload `requirements.txt`.
**Next step:** Launch the Gradio interface v1 and record CP4.
**Who did what:** Luis (BETO fine-tuning pipeline and metrics), Alondra (Hyperparameter documentation and test evaluation), Noah (Error case inspection).

## CP2 2026-11-05
**What we have done:** We created the repository and set up the project structure. We loaded the AuTexTification 2023 dataset to classify customer service messages for ChatSeguro as human or machine. We properly split the data and are currently running the baseline model (TF-IDF + Logistic Regression) and a pre-trained zero-shot classifier. The main metric evaluated is F1.
**What we haven't done yet:** We haven't fine-tuned a Transformer model yet, and we haven't built the interactive Gradio interface.
**Next step:** Document the baseline metrics in the notebook and start building the fine-tuned Transformer solution.
**Who did what:** Luis (Repository setup and code integration), Alondra (Dataset management and split), Noah (Documentation).
