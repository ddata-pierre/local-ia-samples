FROM local-ia/base-llm:latest

LABEL description="Tuto 06 - Serveur et Client MCP local"

# Installation du SDK officiel MCP d'Anthropic
RUN pip install --no-cache-dir "mcp<2"

# Copie des scripts serveur et client
COPY --chown=appuser:appuser mcp_server.py /home/appuser/mcp_server.py
COPY --chown=appuser:appuser mcp_client.py /home/appuser/mcp_client.py

USER appuser
CMD ["python3", "mcp_client.py"]