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

Dies installiert das CUDA Toolkit, das für GPU-Unterstützung erforderlich ist.

---

## Schritt 5: Umgebungsvariablen konfigurieren

Füge die CUDA-Pfade zu deinen Shell-Umgebungsvariablen hinzu:

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Diese Befehle:
- Fügen den CUDA-Bin-Ordner zum PATH hinzu
- Konfigurieren die Library-Pfade für CUDA
- Laden die aktualisierte Konfiguration sofort

---

## Schritt 6: CUDA-Installation überprüfen

Überprüfe, ob CUDA korrekt installiert ist:

```bash
nvcc --version
```

Du solltest die NVIDIA CUDA Compiler-Version sehen (z.B. "Cuda compilation tools, release 12.6").

---

## Schritt 7: UV Package Manager installieren

Installiere den `uv` Package Manager von Astral:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv` ist ein schneller Python-Package-Manager und wird für dieses Projekt verwendet.

Synchronisiere die Abhängigkeiten des Projekts:

```bash
uv sync
```

---

## Schritt 8: Projekt testen

### Einfacher Test - Hauptprogramm ausführen:

```bash
uv run main.py
```

Dies führt das Hauptprogramm aus und überprüft, ob die grundlegende Installation funktioniert.

---

## Schritt 9: GPU-Nutzung testen

Überprüfe, ob die GPU tatsächlich verwendet wird:

```bash
uv run gpu_test.py
```

Dieses Skript testet, ob die GPU korrekt erkannt und genutzt wird.

---

## Fehlerbehebung

Falls Probleme auftreten:

1. **CUDA wird nicht erkannt**: Stelle sicher, dass `nvcc --version` funktioniert und die LD_LIBRARY_PATH korrekt gesetzt ist
2. **GPU wird nicht erkannt**: Überprüfe, ob die GPU-Treiber installiert sind (`nvidia-smi` sollte funktionieren)

---
