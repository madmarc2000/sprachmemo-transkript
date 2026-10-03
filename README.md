# sprachmemo-transkript

Macht aus eigenen Sprachaufnahmen einen lesbaren Text, auch bei Dialekt, Stottern oder Fahrgeräusch. Unsichere Stellen werden nicht geraten: Jede kommt als kurzer Clip auf eine Korrekturseite, wo man sie anhört und das richtige Wort tippt.

*English summary below.*

## Wie es arbeitet

1. **Drei Fassungen:** Ein Docker-Worker transkribiert die Aufnahme dreimal hintereinander, mit Whisper large-v3-turbo, Whisper large-v3 und CrisperWhisper 2.0 im Verbatim-Modus mit Wort-Zeitstempeln. Die Modelle laufen nacheinander, deshalb reicht eine GPU mit 5 GB.
2. **Verschmelzen:** Claude bekommt nur die drei Texte, keine Werkzeuge und kein Audio. Daraus macht es einen bereinigten Text. Wo sich die Fassungen widersprechen, setzt es eine Marke `{{N}}` mit einem wörtlichen Zitat aus der Fassung mit Zeitstempeln.
3. **Verorten:** Das Zitat wird in den Wort-Zeitstempeln gesucht. Daraus wird ein Clip mit 1,2 s Rand geschnitten.
4. **Korrigieren:** Die Korrekturseite zeigt jede offene Stelle mit Clip und Umfeld. Mit **Passt** oder **Weglassen** wird der Markdown-Text sofort neu geschrieben. Weicht eine Korrektur von der Lesart ab, kommt sie in eine Wortliste. Die geht beim nächsten Verschmelzen als Kontext mit, so lernt das System Eigennamen und Dialektwörter.

Das Audio verlässt nie deine Rechner. An Claude gehen nur die maschinellen Texte.

## Voraussetzungen

- Python 3.11 oder neuer (nur Standardbibliothek), `ffmpeg`
- Docker, ideal mit NVIDIA-GPU und NVIDIA Container Toolkit, lokal oder auf einem Rechner, der per SSH erreichbar ist (`ssh host` ohne Passwort)
- [Claude Code](https://docs.claude.com/en/docs/claude-code) als `claude` im PATH, angemeldet oder mit `ANTHROPIC_API_KEY`. Jeder andere Befehl geht auch, wenn er den Prompt auf stdin nimmt und JSON auf stdout liefert (`[verschmelzen] befehl`).

## Einrichten

```sh
git clone https://github.com/madmarc2000/sprachmemo-transkript.git
cd sprachmemo-transkript
cp config.example.toml config.toml        # anpassen
```

Worker-Image bauen, auf dem Docker-Rechner:

```sh
docker build -t sprachmemo-transkript worker/
```

Modelle einmalig laden (ca. 8 GB). CrisperWhisper muss dabei einmal ins CTranslate2-Format umgewandelt werden. Das braucht torch, das nur in diesem einen Container installiert wird und nicht im Image bleibt:

```sh
docker run --rm --gpus all -v sprachmemo-cache:/cache --entrypoint sh sprachmemo-transkript -c \
  "pip install -q torch --index-url https://download.pytorch.org/whl/cpu && pip install -q transformers && python /app/fassungen.py --vorbereiten"
```

## Benutzen

```sh
python3 sprachmemo.py verarbeiten "2026-10-03 Einkauf.m4a"   # Datum und Titel aus dem Dateinamen
python3 sprachmemo.py server                                # Korrekturseite, Standard http://127.0.0.1:8795/
```

Ergebnis in `[pfade] ausgabe`: Die Rohfassungen werden einmal geschrieben und nie überschrieben. Der lesbare Text wird bei jeder Korrektur neu geschrieben. Offene Stellen stehen dort als `[?]`. Mit `[ausgabe] obsidian = true` verweisen die Dateien per `[[Wikilink]]` aufeinander.

**Eingangsordner:** Ist `[pfade] eingang` gesetzt, arbeitet `sprachmemo.py eingang` jede Audiodatei dort ab und verschiebt sie danach nach `erledigt/` bzw. `fehler/`. Den Takt gibt cron oder launchd vor:

```
*/5 * * * *  cd /pfad/zu/sprachmemo-transkript && python3 sprachmemo.py eingang >> eingang.log 2>&1
```

Liegt der Eingang in iCloud Drive, muss die Datei lokal vorhanden sein (`brctl download <datei>`). Platzhalter `.name.icloud` werden übersprungen.

**Meldung:** `[melden] befehl` wird nach jedem Lauf mit dem Meldetext als letztem Argument aufgerufen, z. B. `["ntfy", "publish", "mein-thema"]`.

## Sicherheit

Die Korrekturseite hat keine Anmeldung. Sie bindet deshalb standardmäßig an `127.0.0.1`. Für den Zugriff vom Handy nur an ein vertrauenswürdiges Netz binden (z. B. eine Tailscale-Adresse), nie ans offene Internet.

## Tests

```sh
python3 -m unittest -v     # ohne GPU, Docker und Claude
```

## Lizenz

MIT, siehe [LICENSE](LICENSE). Die Modelle (Whisper, CrisperWhisper) haben eigene Lizenzen. Vor der Nutzung bitte dort prüfen.

---

## English summary

Turns your own voice memos into readable text, built for dialect, stuttering and noisy recordings. A Docker worker (GPU recommended, 5 GB VRAM is enough) produces three transcripts (Whisper large-v3-turbo, large-v3, CrisperWhisper 2.0 verbatim with word timestamps). Claude merges them *blind* (text only, no tools, no audio) and marks every disagreement. Each uncertain spot is located via word timestamps and cut into a short audio clip. A small web page lets you listen and confirm or correct each one, and the Markdown output (Obsidian-friendly) updates instantly. Confirmed corrections feed a word list that improves the next merge. Audio never leaves your machines. The prompt and UI are German. Set `[verschmelzen] beschreibung` and `[worker] sprache` for other languages. Setup: see the commands above.
