FROM local-ia/base-llm:latest

LABEL description="Tuto 05 - RAG Local avec preuve d'origine"

# Installation de scikit-learn pour la recherche sémantique TF-IDF / Cosinus
RUN pip install --no-cache-dir scikit-learn

# Copie des documents de la base de connaissances et des scripts
COPY --chown=appuser:appuser docs /home/appuser/docs
COPY --chown=appuser:appuser rag_app.py /home/appuser/rag_app.py

USER appuser
CMD ["python3", "rag_app.py"]