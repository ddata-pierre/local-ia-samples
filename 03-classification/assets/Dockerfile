FROM local-ia/base-llm:latest

LABEL description="Tuto 03 - Classification"

# Plus de téléchargement de modèle ici ! Le volume s'en charge.
COPY --chown=appuser:appuser tickets.json /home/appuser/tickets.json
COPY --chown=appuser:appuser classify.py /home/appuser/classify.py

USER appuser
CMD ["python3", "classify.py"]