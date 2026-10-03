"""Erzeugt drei Whisper-Fassungen einer Aufnahme und schreibt sie als JSON.

Aufruf im Container:
  fassungen.py /arbeit/<name>.wav     ->  /arbeit/<name>.json
  fassungen.py --vorbereiten          ->  lädt bzw. wandelt alle Modelle einmalig nach /cache

Fassungen: Whisper large-v3-turbo, large-v3 und CrisperWhisper 2.0 verbatim mit
Wort-Zeitstempeln. Die Modelle laufen nacheinander, damit auch eine Karte mit 5 GB reicht.

Umgebung: SPRACHE (Standard de), GERAET (auto | cuda | cpu), COMPUTE (Standard int8).
"""
import gc
import json
import os
import sys
import time
import wave

import numpy as np
from faster_whisper import WhisperModel

SPRACHE = os.environ.get("SPRACHE", "de")
GERAET = os.environ.get("GERAET", "auto")
COMPUTE = os.environ.get("COMPUTE", "int8")
HF_WHISPER = "/cache/hf-whisper"
CRISPER_ID = "nyralabs/CrisperWhisper2.0_large"
CRISPER_CACHE = "/cache/crisperwhisper"
TURBO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
LARGE = "Systran/faster-whisper-large-v3"


def whisper_modell(modell):
    return WhisperModel(modell, device=GERAET, compute_type=COMPUTE, download_root=HF_WHISPER)


def crisper_modell():
    from crisperwhisper import CrisperWhisperModel
    # Mit cache_dir findet das Paket ein schon umgewandeltes CT2-Modell wieder; ohne würde es
    # das Originalmodell (~3 GB) bei jedem Start neu umwandeln.
    return CrisperWhisperModel(CRISPER_ID, compute_type=COMPUTE, device=GERAET, cache_dir=CRISPER_CACHE)


if sys.argv[1:] == ["--vorbereiten"]:
    for modell in (TURBO, LARGE):
        print("lade", modell, flush=True)
        whisper_modell(modell)
        gc.collect()
    print("lade und wandle", CRISPER_ID, flush=True)
    crisper_modell()
    print("fertig: alle Modelle liegen in /cache")
    sys.exit(0)

WAV = sys.argv[1]
# Selbst einlesen: das von crisperwhisper gezogene PyAV passt nicht zu decode_audio
# von faster-whisper (open() ohne metadata_errors). Erwartet 16 kHz mono 16 bit.
with wave.open(WAV) as w:
    assert w.getframerate() == 16000 and w.getnchannels() == 1, "16 kHz mono erwartet"
    AUDIO = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
ZIEL = WAV.rsplit(".", 1)[0] + ".json"
OPTS = {"language": SPRACHE, "vad_filter": True, "condition_on_previous_text": False}
erg = {"datei": WAV, "fassungen": {}, "dauer_s": {}}


def whisper(name, modell, woerter=False):
    t = time.time()
    m = whisper_modell(modell)
    segmente, _ = m.transcribe(AUDIO, word_timestamps=woerter, **OPTS)
    segmente = list(segmente)
    erg["fassungen"][name] = " ".join(s.text.strip() for s in segmente).strip()
    if woerter:
        erg["woerter_" + name] = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                                  for s in segmente for w in (s.words or [])]
    erg["dauer_s"][name] = round(time.time() - t, 1)
    del m
    gc.collect()
    print(f"{name}: {erg['dauer_s'][name]} s", flush=True)


whisper("turbo", TURBO)
# large-v3 mit Wort-Zeitstempeln: Ersatz-Zeitbasis, falls CrisperWhisper nichts liefert
whisper("large-v3", LARGE, woerter=True)

t = time.time()
r = crisper_modell().transcribe(WAV, language=SPRACHE, mode="verbatim", word_timestamps=True)
erg["dauer_s"]["crisper"] = round(time.time() - t, 1)
lv3 = erg.pop("woerter_large-v3")
if r.text.strip() and r.words:
    erg["fassungen"]["crisper"] = r.text
    erg["woerter"] = [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2)} for w in r.words]
    erg["zeitbasis"] = "crisper"
else:
    # Bei sehr leisen, kurzen Aufnahmen liefert CrisperWhisper mitunter text='' und words=None
    erg["woerter"] = lv3
    erg["zeitbasis"] = "large-v3"
print(f"crisper: {erg['dauer_s']['crisper']} s, Zeitbasis {erg['zeitbasis']}, "
      f"{len(erg['woerter'])} Wörter", flush=True)

with open(ZIEL, "w") as fh:
    json.dump(erg, fh, ensure_ascii=False, indent=1)
print("fertig", ZIEL)
