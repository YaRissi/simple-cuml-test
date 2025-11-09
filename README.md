# Installationsanleitung

Diese Anleitung führt dich Schritt für Schritt durch die Installation

## Inhaltsverzeichnis

1. [CUDA-Installation (für beide Methoden)](#cuda-installation-für-beide-methoden)
2. [Lokale Installation](#lokale-installation)
3. [VS Code Dev Container (mit Docker)](#vs-code-dev-container-mit-docker)

---

## CUDA-Installation (für beide Methoden)

Die folgenden Schritte sind notwendig, unabhängig davon, ob du lokal oder im Dev Container arbeitest.

## Schritt 1: Ubuntu-Version überprüfen

Überprüfe zuerst deine Ubuntu-Version:

```bash
lsb_release -a
```

Dies zeigt dir deine genaue Ubuntu-Version. Je nach Version (Ubuntu 24.04, 22.04 oder 20.04) musst du möglicherweise die CUDA-Keyring-Datei anpassen.

---

## Schritt 2: CUDA-Keyring herunterladen

Wähle den Befehl basierend auf deiner Ubuntu-Version:

**Für Ubuntu 24.04:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
```

**Für Ubuntu 22.04:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```

**Für Ubuntu 20.04:**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
```

---

## Schritt 3: CUDA-Keyring installieren

Installiere die heruntergeladene Keyring-Datei:

```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

oder (bei Ubuntu 20.04):

```bash
sudo dpkg -i cuda-keyring_1.0-1_all.deb
```

---

## Schritt 4: CUDA Toolkit installieren  (dauert ein bisschen lol)

Aktualisiere die Paketlisten und installiere das CUDA Toolkit Version 12.6:

```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

---

## Schritt 5: Umgebungsvariablen konfigurieren

Füge die CUDA-Pfade zu deinen Shell-Umgebungsvariablen hinzu:

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Schritt 6: CUDA-Installation überprüfen

Überprüfe, ob CUDA korrekt installiert ist:

```bash
nvcc --version
```

Du solltest die NVIDIA CUDA Compiler-Version sehen (z.B. "Cuda compilation tools, release 12.6").

---

## Lokale Installation


## UV Package Manager installieren

Installiere den `uv` Package Manager von Astral:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Synchronisiere die Abhängigkeiten des Projekts:

```bash
uv sync
```

---

## Projekt testen

### Einfacher Test - Hauptprogramm ausführen:

```bash
uv run main.py
```

---

## GPU-Nutzung testen

Überprüfe, ob die GPU tatsächlich verwendet wird:

```bash
uv run gpu_test.py
```

---

### Fehlerbehebung (Lokale Installation)

Falls Probleme auftreten:

1. **CUDA wird nicht erkannt**: Stelle sicher, dass `nvcc --version` funktioniert und die LD_LIBRARY_PATH korrekt gesetzt ist
2. **GPU wird nicht erkannt**: Überprüfe, ob die GPU-Treiber installiert sind (`nvidia-smi` sollte funktionieren)

---

## VS Code Dev Container (mit Docker, falls du docker zu laufen bekommst 👀)

### Voraussetzungen

- **VS Code** installiert
- **Docker** mit GPU-Unterstützung (NVIDIA Container Toolkit)
- **Dev Containers Extension** für VS Code

### Dev Container starten

1. Öffne das Projekt in VS Code
2. Drücke **Cmd/Ctrl+Shift+P** und suche nach `Dev Containers: Reopen in Container`
3. VS Code startet den Container mit:
   - RAPIDS AI mit CUDA 12
   - Python 3.13
   - GPU-Unterstützung
   - Alle Dependencies vorinstalliert

### Was im Dev Container enthalten ist

- **NVIDIA RAPIDS AI Base Image** (25.10 mit CUDA 12)
- **Python 3.13** mit Conda
- **cuDF, cuML, cuGraph** für GPU-Datenverarbeitung
- **Git, Curl, Build Tools** für die Entwicklung
- **VS Code Extensions** (Python, Pylance, Debugger)
- **Jupyter & App Ports** (8888, 8080) automatisch weitergeleitet

### Dev Container Konfiguration

Die Konfiguration befindet sich in `.devcontainer/devcontainer.json`:

```json
{
  "image": "nvcr.io/nvidia/rapidsai/base:25.10-cuda12-py3.13",
  "runArgs": [
    "--gpus=all",
    "--shm-size=1g",
    "--ulimit=memlock=-1",
    "--ulimit=stack=67108864"
  ]
}
```

### Projekt im Container ausführen

Nachdem der Container gestartet ist:

```bash
uv run main.py        # Hauptprogramm
uv run gpu_test.py    # GPU-Test
```

---
