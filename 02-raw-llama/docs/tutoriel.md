# 02-raw-llama — LLM sans Ollama : deux approches (llama.cpp + llamafile)

> **Objectif** : faire tourner un LLM en local sans Ollama, directement depuis Python.
> **Durée** : ~10 min.
> **Prérequis** : Podman installé, 16 Go RAM, modèles < 1 Go.
> **Note matériel** : Sur N100 800 MHz, l'inférence est lente (~1-5 tok/s). Le serveur HTTP peut timeout. Privilégiez le CLI.

---

## 1. Pourquoi cette approche ?

Montrer comment utiliser un LLM directement en Python.

> Les tutoriels tournent en **CPU pur** sur N100 (800 MHz, 4 cœurs).

---

## 2. Création d'une image de base pour les tutoriels

### 2.1 Dockerfile — Image avec modèles préchargés

**Fichier** : `assets/base-llm/Dockerfile.base`

```dockerfile
FROM docker.io/python:3.11-slim

LABEL maintainer="ddata.fr"
LABEL description="Image de base commune pour les tutos LLM local"

# 1. Dépendances système lourdes (compilateur C++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# 2. Utilisateur non-root
RUN useradd -m -s /bin/bash appuser
WORKDIR /home/appuser

# 3. Venv et installation de llama-cpp-python (compilation une seule fois !)
RUN python3 -m venv /home/appuser/.venv
ENV PATH="/home/appuser/.venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python huggingface-hub

# 4. Dossier partagé pour les modèles (+ symlink pour compatibilité)
RUN mkdir -p /models && ln -s /models /home/appuser/models
```

**Explications clés** :
| Ligne | Rôle |
|-------|------|
| `build-essential cmake` | Compile `llama-cpp-python` (bindings C++) |
| `CMAKE_ARGS="-DLLAMA_BLAS=ON..."` | Active OpenBLAS → accélération CPU (AVX2) |
| `python3 -m venv` | Isole les dépendances Python du système |
| `mkdir -p /models && ln -s` | Le volume `shared-models` monte sur `/models` ; le symlink assure la compatibilité avec `/home/appuser/models` |

### 2.2 Build

```bash
cd assets/base-llm
podman build -f Dockerfile.base -t local-ia/base-llm:latest .
podman volume create shared-models
```

Alimenter le volume une seule fois (via un petit conteneur éphémère de téléchargement) :

```bash
podman run --rm -v shared-models:/models local-ia/base-llm:latest \
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF', filename='qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/models', local_dir_use_symlinks=False)"
```

---

## 3. Alternative : llamafile (binaire unique, pas de build)

> **llamafile** = modèle GGUF + llama.cpp compilé en un seul exécutable portable (Cosmopolitan Libc). Zéro dépendance, zéro build.

```bash
# Télécharger (exemple avec SmolLM2 135M)
wget -O smollm2-135m-llamafile https://huggingface.co/Mozilla/SmolLM2-135M-Instruct-llamafile/resolve/main/SmolLM2-135M-Instruct.Q4_K_M.llamafile
chmod +x smollm2-135m-llamafile

# Lancer serveur OpenAI-compatible sur port 8080
./smollm2-135m-llamafile --server --port 8080 -ngl 0

# Tester
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"smollm2","messages":[{"role":"user","content":"test"}],"stream":false}' | jq
```

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| **Image base + llama.cpp** | Contrôle total, extensible, volume partagé | Build initial (~3 min) |
| **llamafile** | Zéro install, portable, instantané | Moins flexible, version figée |

---

## 4. Test rapide (non-interactif)

```bash
# Avec le modèle Qwen2.5-3B (déjà dans shared-models)
podman run -v shared-models:/models -it local-ia/base-llm:latest python3 -c "
from llama_cpp import Llama
llm = Llama(model_path='/models/qwen2.5-3b-instruct-q4_k_m.gguf', n_ctx=512, n_threads=4, verbose=False)
print(llm('Bonjour', max_tokens=32)['choices'][0]['text'])
"
```

---

## 5. Changer de modèle

```bash
# Exemple : SmolLM2 135M (plus rapide, moins capable)
podman run --rm -v shared-models:/models local-ia/base-llm:latest \
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/SmolLM2-135M-GGUF', filename='SmolLM2-135M-Q4_K_M.gguf', local_dir='/models', local_dir_use_symlinks=False)"
```

Puis tester :
```bash
podman run -v shared-models:/models -it local-ia/base-llm:latest python3 -c "
from llama_cpp import Llama
llm = Llama(model_path='/models/SmolLM2-135M-Q4_K_M.gguf', n_ctx=512, n_threads=4, verbose=False)
print(llm('Bonjour', max_tokens=32)['choices'][0]['text'])
"
```

---

## 6. Exercices + Corrections

### Exercice 1 — Mode streaming token par token

**Consigne** : Modifier le test rapide pour afficher la réponse token par token (streaming).

**Correction** :
```bash
podman run -v shared-models:/models -it local-ia/base-llm:latest python3 -c "
from llama_cpp import Llama
llm = Llama(model_path='/models/qwen2.5-3b-instruct-q4_k_m.gguf', n_ctx=512, n_threads=4, verbose=False)
for chunk in llm('Compte de 1 à 5', max_tokens=64, stream=True):
    print(chunk['choices'][0]['text'], end='', flush=True)
print()
"
```

### Exercice 2 — Comparer deux modèles

**Consigne** : Télécharger `Phi-3-mini-4k-instruct-q4.gguf` (plus capable) et comparer la réponse à une même question technique.

**Correction** :
```bash
# Télécharger Phi-3
podman run --rm -v shared-models:/models local-ia/base-llm:latest \
  python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='microsoft/Phi-3-mini-4k-instruct-gguf', filename='Phi-3-mini-4k-instruct-q4.gguf', local_dir='/models', local_dir_use_symlinks=False)"

# Comparer
QUESTION="Explique la différence entre HTTP et HTTPS en 2 phrases."
echo \"=== Qwen2.5-3B ===\"
podman run -e QUESTION="$QUESTION" -v shared-models:/models -it local-ia/base-llm:latest python3 -c '
import os
from llama_cpp import Llama
llm = Llama(model_path="/models/qwen2.5-3b-instruct-q4_k_m.gguf", n_ctx=512, n_threads=4, verbose=False)
print(llm(os.environ["QUESTION"], max_tokens=64)["choices"][0]["text"])
'

echo \"=== Phi-3-mini ===\"
podman run --rm -e QUESTION="$QUESTION" -v shared-models:/models -it local-ia/base-llm:latest python3 -c '
import os
from llama_cpp import Llama
llm = Llama(model_path="/models/Phi-3-mini-4k-instruct-q4.gguf",n_ctx=2048,n_threads=4,verbose=False)
prompt = ("<|user|>\n"+ os.environ["QUESTION"]+ "<|end|>\n""<|assistant|>\n")
result = llm(prompt,max_tokens=128,temperature=0.2,stop=["<|end|>", "<|user|>"],echo=False)
print("Texte :", repr(result["choices"][0]["text"]))
'
```

**Observation attendue** : Phi-3 plus précis techniquement, mais ~2x plus lent sur N100.

---

## 7. Nettoyage

```bash
podman rmi local-ia/base-llm:latest
podman volume rm shared-models
```

---

## 8. Aller plus loin

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python docs](https://github.com/abetlen/llama-cpp-python)
- [llamafile Mozilla](https://github.com/Mozilla-Ocho/llamafile)
- Prochaine étape : **03-classification** — classification structurée de tickets avec le modèle partagé.