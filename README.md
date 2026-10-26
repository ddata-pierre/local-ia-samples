# IA Locale — Démos Éducatives Complètes

> **But** : comprendre comment fonctionne l'IA en local, sans cloud, sur du matériel modeste (Intel N100, 800 MHz, 16 Go RAM).
> **Public** : développeurs, data engineers, curieux techniques.
> **Durée** : ~10–25 min par tutoriel.

---

## Structure du dépôt

```
local-ia/
├── 01-ollama/          # Ollama en conteneur (Podman) — Dockerfile + API REST
├── 02-raw-llama/       # LLM sans Ollama : image de base + llama.cpp (Python)
├── 03-classification   # Classification de tickets (département, frustration, urgence)
├── 04-extraction       # Extraction d'entités (prénom, nom, âge, email) depuis CSV
├── 05-rag/             # RAG minimal : embedding TF-IDF + recherche + génération
├── 06-mcp/             # Serveur MCP (Model Context Protocol) + client LLM
└── README.md           # Ce fichier
```

Chaque dossier contient :
- `docs/tutoriel.md` — explications pas à pas (français, ~10–25 min lecture)
- `assets/` — Dockerfiles, scripts Python, données, modèles

---

## Prérequis communs

| Outil | Installation | Vérification |
|-------|--------------|--------------|
| **WSL Ubuntu** | déjà là | `uname -a` |
| **Python 3.10+** | `sudo apt update && sudo apt install -y python3 python3-venv python3-pip` | `python3 --version` |
| **Git** | `sudo apt install -y git` | `git --version` |
| **Podman** | `sudo apt update && sudo apt install -y podman` | `podman --version` |
| **make** (optionnel) | `sudo apt install -y build-essential` | `make --version` |

> ⚠️ **N100 800 MHz** : tout est dimensionné pour tourner en CPU seul. Pas de GPU, pas de quantization complexe. Les modèles font < 1 Go.

---

## Ordre recommandé

| # | Tutoriel | Ce qu'on apprend | Temps |
|---|----------|------------------|-------|
| 01 | **Ollama** | Installer, pull, chat, API REST, Modelfile | 10 min |
| 02 | **Sans Ollama** | Image de base commune, `llama-cpp-python`, volume partagé modèles | 15 min |
| 03 | **Classification** | Few-shot prompting, JSON mode, classification structurée | 15 min |
| 04 | **Extraction** | NER structuré, CSV→JSON/CSV, few-shot, typage strict | 15 min |
| 05 | **RAG** | TF-IDF + cosinus, injection contexte, preuve d'origine | 20 min |
| 06 | **MCP** | Serveur FastMCP (stdio), découverte outils, function calling | 25 min |

---

## Démarrage rapide

### Étape 0 — Image de base commune (requise pour 02 à 06)

```bash
# Une seule fois : build de l'image de base + téléchargement modèle partagé
cd 02-raw-llama/assets/base-llm
podman build -f Dockerfile.base -t local-ia/base-llm:latest .
podman volume create shared-models
podman run --rm -v shared-models:/models local-ia/base-llm:latest \
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
    filename='qwen2.5-3b-instruct-q4_k_m.gguf',
    local_dir='/models',
    local_dir_use_symlinks=False
)
"
```

> Le volume `shared-models` est monté sur `/models` dans tous les conteneurs dérivés (03 à 06).

---

### 01 — Ollama (Podman, indépendant)

```bash
cd 01-ollama
podman build -t local-ia/ollama:tiny .           # build image (une fois, ~2-3 min)
podman volume create ollama-data                  # volume persistant
podman run -d --name ollama -p 11434:11434 -v ollama-data:/root/.ollama local-ia/ollama:tiny
podman exec -it ollama ollama run qwen2.5:0.5b   # REPL interactive
# API REST dispo sur http://localhost:11434
```

---

### 02 — Sans Ollama (image de base déjà faite à l'étape 0)

```bash
# REPL Python interactif avec le modèle partagé
podman run -v shared-models:/models -it local-ia/base-llm:latest python3 -c "
from llama_cpp import Llama
llm = Llama(model_path='/models/qwen2.5-3b-instruct-q4_k_m.gguf', n_ctx=512, n_threads=4, verbose=False)
while True:
    q = input('Vous: ')
    if q == 'quit': break
    r = llm(q, max_tokens=64, stream=True)
    for chunk in r:
        print(chunk['choices'][0]['text'], end='', flush=True)
    print()
"
```

---

### 03 — Classification de tickets

```bash
cd 03-classification/assets
podman build -t local-ia/classification:tuto03 .
podman run --rm -v shared-models:/models local-ia/classification:tuto03
# Résultats dans resultats.json (copiable via podman cp)
```

---

### 04 — Extraction d'entités (CSV → JSON/CSV)

```bash
cd 04-extraction/assets
podman build -t local-ia/extraction:tuto04 .
podman run --rm -v shared-models:/models local-ia/extraction:tuto04
# Génère fiches_profils.json et profils_extraits.csv
```

---

### 05 — RAG local avec preuve d'origine

```bash
cd 05-rag/assets
podman build -t local-ia/rag-demo:tuto05 .
podman run --rm -v shared-models:/models local-ia/rag-demo:tuto05
# 3 questions test avec citations de documents sources
```

---

### 06 — Serveur MCP + Client LLM

```bash
cd 06-mcp/assets
podman build -t local-ia/mcp-demo:tuto06 .
podman run --rm -v shared-models:/models local-ia/mcp-demo:tuto06
# Démontre : découverte outils → function calling → exécution → synthèse
```

---

## Modèles utilisés (pré-téléchargés)

| Modèle | Taille | Usage | Source |
|--------|--------|-------|--------|
| `smollm2:135m` | ~80 Mo | Chat rapide, test (Ollama seulement) | HuggingFace / Ollama |
| `qwen2.5:0.5b` | ~300 Mo | Ollama : RAG, agent, meilleur raisonnement | HuggingFace / Ollama |
| `tinyllama:1.1b` | ~650 Mo | Ollama : alternative plus capable | HuggingFace / Ollama |
| `qwen2.5-3b-instruct-q4_k_m.gguf` | ~1.8 Go | **Modèle partagé** pour tutos 02–06 (llama.cpp) | HuggingFace (Qwen) |
| `nano-jev` (sdmlai/nano-jev) | ~33M | RAG eval : relevance, sufficient, grounded | HuggingFace (sdmlai) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22M | Proxy Kev : choice/boolean/score | HuggingFace (cross-encoder) |

> Les scripts 02–06 utilisent **exclusivement** `qwen2.5-3b-instruct-q4_k_m.gguf` via le volume `shared-models` monté sur `/models`.  
> Le tutoriel 01 (Ollama) utilise ses propres modèles gérés par Ollama.  
> Le tutoriel 07 télécharge `nano-jev` et `cross-encoder/ms-marco-MiniLM-L-6-v2` automatiquement au premier run (cache HF).

---

## Problèmes fréquents

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `podman: command not found` | Podman non installé | `sudo apt install -y podman` |
| `Error: unable to pull image` | Réseau / registry | Vérifier proxy, `podman pull ollama/ollama:latest` |
| `port 11434 already in use` | Conflit port hôte | `podman run -p 11435:11434 ...` ou `podman stop <autre>` |
| `ModuleNotFoundError: llama_cpp` | venv non activé / image incomplète | Rebuild `local-ia/base-llm:latest` |
| Modèle non trouvé dans `/models` | Volume mal monté | Vérifier `-v shared-models:/models` (pas `/home/appuser/models`) |

---

## Licence

Code : MIT. Modèles : licences respectives (Apache 2.0 pour Qwen/SmolLM, Llama 2 pour TinyLlama).
Usage éducatif uniquement — pas de garantie de production.

---

## Auteur

Pierre Dupont — Architecte Data freelance (ddata.fr)
Créé pour démonstration pédagogique, octobre 2026.