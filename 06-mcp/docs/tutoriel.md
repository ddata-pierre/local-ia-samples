# 06-mcp-server — Créer et consommer un serveur MCP local

> **Objectif** : Développer un mini-serveur MCP (Model Context Protocol) sur mesure offrant des outils d'entreprise (consultation de stock & météo d'un site), puis connecter un LLM local qui utilise ce serveur via un protocole standardisé.
> **Durée** : ~25 min.
> **Prérequis** : Image de base `local-ia/base-llm:latest` et le volume `shared-models` configurés précédemment.

---

## 1. Comprendre l'Architecture MCP
Le **Model Context Protocol (MCP)** est le standard ouvert promu par Anthropic qui permet de connecter de façon sécurisée des LLM à des outils externes (bases de données, API, outils locaux). Dans ce tutoriel, nous créons un **serveur MCP personnalisé en Python** et nous le connectons à notre LLM local (`Qwen2.5-3B-Instruct`) via un **client MCP**.

Le **Model Context Protocol (MCP)** fonctionne sur un schéma Client-Serveur :

* **Serveur MCP** : Expose des ressources (fichiers, bases de données) ou des outils exécutables (API, calculs). La communication se fait via FastMCP sur le canal d'entrée/sortie standard (`stdio`) ou via HTTP/SSE.
* **Client MCP + LLM** : Interroge le serveur pour connaître les outils disponibles, transmet l'intention de l'utilisateur au LLM, et exécute les outils retournés par le LLM via le protocole MCP.

```
[Utilisateur] ──(Question)──> [Client MCP / LLM Engine] ──(Outils dispo)──> [Serveur MCP]
                                    │                                           │
                                    ├───(Appel d'outil : check_stock)───────────┤
                                    │                                           │
                                    └───────(Résultat JSON)─────────────────────┘

```

---

## 2. Structure du projet

Créez un dossier `06-mcp-server/assets` avec les fichiers suivants :

```
06-mcp-server/assets/
├── Dockerfile
├── mcp_server.py      # Le mini serveur MCP (FastMCP)
└── mcp_client.py      # Le client LLM qui interroge le serveur MCP

```

---

## 3. Développement du Serveur MCP — `assets/mcp_server.py`

Nous utilisons le SDK officiel `mcp` et sa surcouche simplifiée `FastMCP` pour exposer deux outils :

1. `verifier_stock(article_id: str)` : Vérifie le stock d'un produit dans la base.
2. `obtenir_meteo_site(ville: str)` : Donne les conditions météo d'un site d'entreprise.

```python
#!/usr/bin/env python3
# mcp_server.py — Mini Serveur MCP avec FastMCP

from mcp.server.fastmcp import FastMCP

# Initialisation du serveur MCP
mcp = FastMCP("DData-Enterprise-Tools")

# Base de données fictive pour la démonstration
STOCKS = {
    "SERVEUR-N100": {"stock": 14, "emplacement": "Entrepôt Marseille"},
    "ECRAN-4K": {"stock": 0, "emplacement": "Rupture de stock"},
    "CABLE-ETH-10M": {"stock": 150, "emplacement": "Entrepôt Aix"}
}

METEO_SITES = {
    "marseille": "Soleil, 22°C, Vent 15 km/h",
    "aix": "Nuageux, 19°C, Vent faible",
    "paris": "Pluie, 12°C, Vent 30 km/h"
}

@mcp.tool()
def verifier_stock(article_id: str) -> str:
    """Vérifie la disponibilité et l'emplacement d'un article dans les entrepôts.
    
    Args:
        article_id: L'identifiant exact de l'article (ex: SERVEUR-N100, ECRAN-4K, CABLE-ETH-10M).
    """
    article = article_id.upper()
    if article in STOCKS:
        info = STOCKS[article]
        return f"Article {article} : {info['stock']} unités disponibles ({info['emplacement']})."
    return f"Article {article} non trouvé dans le catalogue."

@mcp.tool()
def obtenir_meteo_site(ville: str) -> str:
    """Obtient les conditions météo actuelles pour un site d'entreprise.
    
    Args:
        ville: Nom de la ville du site (marseille, aix, paris).
    """
    v = ville.lower()
    if v in METEO_SITES:
        return f"Météo actuelle pour le site de {ville.capitalize()} : {METEO_SITES[v]}"
    return f"Aucun site répertorié dans la ville de {ville}."

if __name__ == "__main__":
    # Lance le serveur sur les flux d'entrée/sortie standard (STDIO)
    mcp.run(transport="stdio")

```

---

## 4. Développement du Client LLM + MCP — `assets/mcp_client.py`

Le client se connecte au serveur MCP via des sous-processus (`stdio`), lit le catalogue d'outils déclarés par le serveur, et passe les schémas au LLM pour exécuter un **Function Calling** autonome.

```python
#!/usr/bin/env python3
# mcp_client.py — Client MCP + Inférence LLM locale

# === PROMPTS (modifiables sans toucher à la logique) ===
# Le prompt système est construit dynamiquement avec les outils découverts
TOOL_CALL_FORMAT = """Tu es un assistant connecté à un serveur MCP.
Tu as accès aux outils suivants :
{tools_json}

Pour utiliser un outil, réponds UNIQUEMENT sous la forme JSON suivante :
{{"outil": "nom_de_l_outil", "arguments": {{"parametre": "valeur"}}}}"""

FINAL_SYNTHESIS_PROMPT = "Rédige une réponse claire et courtoise à l'utilisateur à partir du résultat de l'outil."

# === LOGIQUE ===
import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llama_cpp import Llama

# 1. Charger le LLM local
MODEL_PATH = "/models/qwen2.5-3b-instruct-q4_k_m.gguf"
print("🔄 Chargement du modèle LLM local...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)


async def executer_demonstration_mcp():
    # Configuration pour lancer le serveur MCP en arrière-plan via STDIO
    server_params = StdioServerParameters(
        command="python3",
        args=["/home/appuser/mcp_server.py"],
        env=None
    )

    print("🔌 Connexion au serveur MCP...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Étape A : Initialisation de la session MCP
            await session.initialize()
            
            # Étape B : Découverte dynamique des outils MCP
            mcp_tools = await session.list_tools()
            tools_description = []
            
            for tool in mcp_tools.tools:
                tools_description.append({
                    "nom": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                })
            
            print(f"✅ Outils découverts sur le serveur MCP : {[t['nom'] for t in tools_description]}")

            # Étape C : Soumission d'une question nécessitant l'outil
            question = "Combien de serveurs N100 reste-t-il en stock et où sont-ils ?"
            print(f"\n❓ Question utilisateur : '{question}'")

            # Construction du prompt avec la définition des outils découverts
            system_prompt = TOOL_CALL_FORMAT.format(
                tools_json=json.dumps(tools_description, indent=2, ensure_ascii=False)
            )

            # Appel LLM pour choisir l'outil
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.0
            )

            choix_llm = response["choices"][0]["message"]["content"]
            print(f"🤖 Décision du LLM (Appel d'outil) :\n{choix_llm}")

            # Étape D : Exécution de l'outil via le protocole MCP
            try:
                action = json.loads(choix_llm)
                tool_name = action["outil"]
                tool_args = action["arguments"]

                print(f"⚙️ Exécution de l'outil MCP '{tool_name}' avec {tool_args}...")
                
                # Exécution via la session MCP
                mcp_result = await session.call_tool(tool_name, tool_args)
                result_text = mcp_result.content[0].text
                print(f"📊 Résultat renvoyé par le Serveur MCP :\n{result_text}")

                # Étape E : Synthèse finale par le LLM avec le résultat de l'outil
                final_response = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": FINAL_SYNTHESIS_PROMPT},
                        {"role": "user", "content": f"Question : {question}\nRésultat de l'outil MCP : {result_text}"}
                    ],
                    temperature=0.3
                )
                
                print(f"\n💬 Réponse finale à l'utilisateur :\n{final_response['choices'][0]['message']['content']}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"❌ Erreur lors de l'analyse de la décision de l'outil : {e}")

if __name__ == "__main__":
    asyncio.run(executer_demonstration_mcp())
```

---

## 5. Dockerfile — `assets/Dockerfile`

```dockerfile
FROM local-ia/base-llm:latest

LABEL description="Tuto 06 - Serveur et Client MCP local"

# Installation du SDK officiel MCP d'Anthropic
RUN pip install --no-cache-dir "mcp<2"

# Copie des scripts serveur et client
COPY --chown=appuser:appuser mcp_server.py /home/appuser/mcp_server.py
COPY --chown=appuser:appuser mcp_client.py /home/appuser/mcp_client.py

USER appuser
CMD ["python3", "mcp_client.py"]

```

---

## 6. Build et Exécution

Dans le dossier `06-mcp-server/assets` :

```bash
# Build de l'image
podman build -t local-ia/mcp-demo:tuto06 .

# Exécution avec le volume de modèles partagé
podman run --rm -v shared-models:/models local-ia/mcp-demo:tuto06

```

---

## 7. Résultat attendu dans la console

```text
🔄 Chargement du modèle LLM local...
🔌 Connexion au serveur MCP...
✅ Outils découverts sur le serveur MCP : ['verifier_stock', 'obtenir_meteo_site']

❓ Question utilisateur : 'Combien de serveurs N100 reste-t-il en stock et où sont-ils ?'
🤖 Décision du LLM (Appel d'outil) :
{
  "outil": "verifier_stock",
  "arguments": {
    "article_id": "SERVEUR-N100"
  }
}
⚙️ Exécution de l'outil MCP 'verifier_stock' avec {'article_id': 'SERVEUR-N100'}...
📊 Résultat renvoyé par le Serveur MCP :
Article SERVEUR-N100 : 14 unités disponibles (Entrepôt Marseille).

💬 Réponse finale à l'utilisateur :
Il reste actuellement 14 unités du serveur N100 en stock. Elles se trouvent à l'entrepôt de Marseille.

```

---

## 8. Bilan Pédagogique

1. **Découplage total** : Le LLM n'a aucune connaissance codée en dur de la base de données. C'est le serveur MCP qui expose les outils et exécute la logique métier.
2. **Standard ouvert** : Vous pouvez réutiliser ce même serveur `mcp_server.py` avec Claude Desktop, Cursor, Ollama ou n'importe quel client compatible MCP.
3. **Exécution sécurisée** : Les outils s'exécutent dans l'environnement du serveur MCP, garantissant un contrôle précis sur les droits d'accès aux données de l'entreprise.

---

## 9. Exercices + Corrections

### Exercice 1 — Ajouter un outil de conversion de devise

**Consigne** : Ajouter un outil `convertir_devise(montant: float, de: str, vers: str) -> str` dans `mcp_server.py` avec un taux de change fixe (ex: 1 EUR = 1.08 USD). Tester avec la question : "Combien font 500 EUR en USD ?".

**Correction** :
```python
# Dans mcp_server.py, ajouter :
TAUX = {"EUR_USD": 1.08, "USD_EUR": 0.93}

@mcp.tool()
def convertir_devise(montant: float, de: str, vers: str) -> str:
    \"\"\"Convertit un montant entre deux devises (EUR, USD).\"\"\"
    cle = f\"{de.upper()}_{vers.upper()}\"
    if cle in TAUX:
        res = round(montant * TAUX[cle], 2)
        return f\"{montant} {de.upper()} = {res} {vers.upper()}\"
    return f\"Conversion {de} → {vers} non supportée.\"
```
Rebuild + run. Le LLL découvrira l'outil automatiquement via `list_tools()`.

### Exercice 2 — Changer le transport : stdio → SSE (HTTP)

**Consigne** : Modifier le serveur pour écouter en HTTP/SSE (port 8000) au lieu de stdio, et adapter le client pour se connecter via `streamablehttp_client`.

**Correction** :
```python
# Dans mcp_server.py : changer la dernière ligne
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)

# Dans mcp_client.py : changer la connexion
from mcp.client.streamablehttp import streamablehttp_client
server_params = "http://localhost:8000/mcp"  # URL du serveur
async with streamablehttp_client(server_params) as (read, write):
    # ... reste identique
```

**Note** : Nécessite `podman run -p 8000:8000` pour exposer le port. Montre que MCP est transport-agnostique.

---

## 10. Nettoyage

```bash
podman rmi local-ia/mcp-demo:tuto06
```