# Image de base officielle Ollama (basée sur Debian slim)
FROM docker.io/ollama/ollama:latest

# Auteur / métadonnées
LABEL maintainer="ddata.fr"
LABEL description="Ollama avec modèles tiny pour N100"

# Variable d'environnement : modèles à pull au build (cache layers)
ENV OLLAMA_MODELS="smollm2:135m qwen2.5:0.5b tinyllama:1.1b"

# Script d'initialisation : pull les modèles pendant le build
# Utilise un script shell explicite pour éviter les problèmes de continuation de ligne
RUN set -eux; \
    for m in $OLLAMA_MODELS; do \
        ollama serve & \
        sleep 3 && \
        ollama pull "$m" && \
        pkill ollama; \
    done

# Expose le port API REST
EXPOSE 11434

# Ajout ping pour démonstration hors réseau
RUN apt-get update \
    && apt-get install -y --no-install-recommends iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Point d'entrée : lance le serveur Ollama (ENTRYPOINT du base = /bin/ollama)
CMD ["serve"]