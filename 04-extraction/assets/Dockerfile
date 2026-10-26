FROM local-ia/base-llm:latest

LABEL maintainer="ddata.fr"
LABEL description="Tuto 04 - Extraction d'entités et création de fiches profils"

# Copie du script et du jeu de données
COPY --chown=appuser:appuser prospects.csv /home/appuser/prospects.csv
COPY --chown=appuser:appuser extract.py /home/appuser/extract.py

USER appuser
CMD ["python3", "extract.py"]