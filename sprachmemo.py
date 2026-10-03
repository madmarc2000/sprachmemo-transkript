#!/usr/bin/env python3
"""Sprachmemo-Transkript: drei Whisper-Fassungen, Verschmelzen mit Claude, Clip-Korrektur.

  sprachmemo.py verarbeiten <audio> [--datum YYYY-MM-DD] [--slug S] [--titel T] [--neu]
  sprachmemo.py server            # Korrekturseite
  sprachmemo.py eingang           # Eingangsordner abarbeiten (z. B. per cron alle 5 min)

Konfiguration: --config <datei>, sonst $SPRACHMEMO_CONFIG, sonst config.toml neben dem Skript.

Ablauf `verarbeiten`:
  1. ffmpeg -> 16 kHz mono WAV; der Docker-Worker (lokal oder per SSH) erzeugt
     turbo, large-v3 und CrisperWhisper verbatim (mit Wort-Zeitstempeln)
  2. claude -p --tools "" verschmilzt blind; unsichere Stellen kommen als Liste zurück
  3. jede Stelle wird über ein Zitat aus der CrisperWhisper-Fassung zeitlich verortet
     und als Clip ausgeschnitten
  4. Rohfassungen als Markdown (einmal geschrieben, nie überschrieben),
     lesbarer Text als Markdown (wird bei jeder Korrektur neu geschrieben)
"""
import argparse
import datetime
import difflib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import unicodedata
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AUDIO = {".m4a", ".mp3", ".wav", ".aac", ".caf", ".mp4", ".ogg", ".flac"}
CLIP_RAND = 1.2                                            # Sekunden vor/nach der Stelle
C = {}                                                     # Konfiguration, siehe laden()

PROMPT = """Du bekommst mehrere maschinelle Transkripte DERSELBEN Sprachaufnahme (__BESCHREIBUNG__).
Erzeuge EIN bereinigtes, lesbares Transkript.

Regeln, strikt:
1. Verwende nur Wörter, die in mindestens einer Fassung vorkommen, oder offensichtliche
   Schreibvarianten davon. Erfinde keinen Inhalt.
2. Entferne Füllwörter ([UH], [UM] usw.), Stotter-Wiederholungen und abgebrochene Wortanfänge.
3. Keine inhaltliche Glättung, kein Umformulieren ins Hochdeutsche über die Rechtschreibung
   hinaus. Dialektwörter bleiben. Absätze bei Themenwechsel.
4. Jede Stelle, an der die Fassungen sich inhaltlich widersprechen und keine klar plausibel
   ist, markierst du im Text als {{N}} direkt HINTER der besten Lesart (N = laufende Nummer ab 1).
   Ist gar nichts Sinnvolles erkennbar, schreibst du nur {{N}} ohne Lesart.
   Die Lesart ist nur das unsichere Wort bzw. die kürzeste unsichere Wortgruppe (höchstens
   3 Wörter), genau so, wie sie unmittelbar vor {{N}} im Text steht — nie ein ganzer Halbsatz.
5. Für jede Stelle lieferst du ein ZITAT: 3 bis 8 aufeinanderfolgende Wörter, wörtlich aus der
   Fassung "crisper" kopiert, die genau diese Stelle abdecken. Nichts daran ändern.
6. Die Wortliste unten enthält bestätigte Begriffe des Sprechers. Passt eine davon, nimm sie.

Antworte NUR mit JSON, ohne Codeblock:
{"text": "...", "stellen": [{"n": 1, "lesart": "...", "zitat": "..."}]}

WORTLISTE:
"""


# ---------------------------------------------------------------- Konfiguration

def laden(datei):
    pfad = Path(datei or os.environ.get("SPRACHMEMO_CONFIG") or Path(__file__).with_name("config.toml"))
    if not pfad.exists():
        sys.exit(f"Keine Konfiguration unter {pfad} — config.example.toml kopieren und anpassen.")
    C.update(tomllib.loads(pfad.read_text()))


def wert(abschnitt, schluessel, standard=None):
    return C.get(abschnitt, {}).get(schluessel, standard)


def pfad(schluessel, standard):
    return Path(wert("pfade", schluessel, standard)).expanduser()


def daten():
    return pfad("daten", "~/sprachmemo/daten")


def ausgabe(schluessel, standard):
    return pfad("ausgabe", "~/sprachmemo/ausgabe") / wert("pfade", schluessel, standard)


def korrektur_url():
    return wert("server", "url") or f"http://{wert('server', 'host', '127.0.0.1')}:{wert('server', 'port', 8795)}/"


# ---------------------------------------------------------------- Verarbeiten

def lauf(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def wortliste_text():
    wl = ausgabe("wortliste", "Sprachmemos/Wortliste.md")
    if not wl.exists():
        return "(leer)"
    zeilen = [z for z in wl.read_text().splitlines() if z.startswith("| ") and "---" not in z]
    return "\n".join(zeilen[1:]) or "(leer)"


def worker_fassungen(audio, slug, ordner):
    """Erzeugt die drei Fassungen im Docker-Worker — lokal oder auf dem SSH-Host aus [worker]."""
    wav = ordner / f"{slug}.wav"
    lauf(["ffmpeg", "-loglevel", "error", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav)])
    ssh = wert("worker", "ssh", "")
    arbeit = wert("worker", "arbeit", "sprachmemo-arbeit")
    docker = ["docker", "run", "--rm", *(["--gpus", "all"] if wert("worker", "gpu", True) else []),
              "-e", f"SPRACHE={wert('worker', 'sprache', 'de')}",
              "-v", "__ARBEIT__:/arbeit", "-v", f"{wert('worker', 'cache_volume', 'sprachmemo-cache')}:/cache",
              wert("worker", "image", "sprachmemo-transkript"), f"/arbeit/{slug}.wav"]
    print("Worker: drei Fassungen …", flush=True)
    if ssh:
        entfernt = arbeit if arbeit.startswith("/") else f"$HOME/{arbeit}"
        befehl = shlex.join(docker).replace("__ARBEIT__", f'"{entfernt}"')
        lauf(["ssh", ssh, f'mkdir -p "{entfernt}"'])
        lauf(["scp", "-q", str(wav), f"{ssh}:{arbeit}/{slug}.wav"])
        lauf(["ssh", ssh, befehl])
        lauf(["scp", "-q", f"{ssh}:{arbeit}/{slug}.json", str(ordner / "fassungen.json")])
    else:
        lokal = Path(arbeit).expanduser()
        lokal = lokal if lokal.is_absolute() else Path.home() / lokal
        lokal.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, lokal / wav.name)
        lauf([x.replace("__ARBEIT__", str(lokal)) for x in docker])
        shutil.copy2(lokal / f"{slug}.json", ordner / "fassungen.json")
    return json.loads((ordner / "fassungen.json").read_text())


def verschmelzen(fassungen, zeitbasis="crisper"):
    # Zitate kommen aus der Fassung mit Wort-Zeitstempeln (Crisper, ersatzweise large-v3)
    prompt = PROMPT.replace("__BESCHREIBUNG__", wert("verschmelzen", "beschreibung", "deutsch"))
    teile = [prompt.replace('"crisper"', f'"{zeitbasis}"') + wortliste_text(), ""]
    for name, text in fassungen.items():
        teile += [f"=== Fassung {name}", text, ""]
    print("Claude verschmilzt …", flush=True)
    befehl = wert("verschmelzen", "befehl", ["claude", "-p", "--model", "sonnet", "--tools", ""])
    roh = lauf(befehl, input="\n".join(teile))
    roh = roh.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(roh)


def norm(w):
    return re.sub(r"[^\wäöüß]", "", w.lower())


def verorten(zitat, woerter):
    """Findet das Zitat in der Wort-Zeitliste (Crisper bzw. large-v3); liefert (start, ende) in s."""
    ziel = [norm(w) for w in zitat.split() if norm(w)]
    wl = [norm(w["w"]) for w in woerter]
    if not ziel:
        return None
    best, pos = 0.0, None
    for i in range(0, max(1, len(wl) - len(ziel) + 1)):
        q = difflib.SequenceMatcher(None, ziel, wl[i:i + len(ziel)]).ratio()
        if q > best:
            best, pos = q, i
    if pos is None or best < 0.5:
        return None
    return woerter[pos]["s"], woerter[min(pos + len(ziel), len(woerter)) - 1]["e"]


def clip(wav, start, ende, ziel):
    s = max(0.0, start - CLIP_RAND)
    lauf(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{s:.2f}", "-t", f"{ende + CLIP_RAND - s:.2f}",
          "-i", str(wav), "-c:a", "aac", "-b:a", "64k", str(ziel)])


def roh_pfad(z):
    return ausgabe("roh", "Sprachmemos/roh") / f"{z['datum']}-{z['slug']}-fassungen.md"


def text_pfad(z):
    return ausgabe("text", "Sprachmemos") / f"{z['datum']}-{z['slug']} - Text.md"


def verweis(von, nach):
    """Link von einer Ausgabedatei auf eine andere: Obsidian-Wikilink oder relativer Markdown-Link."""
    wurzel = pfad("ausgabe", "~/sprachmemo/ausgabe")
    if wert("ausgabe", "obsidian", False):
        return f"[[{nach.relative_to(wurzel).with_suffix('').as_posix()}]]"
    rel = os.path.relpath(nach, von.parent)
    return f"[{nach.stem}]({urllib.parse.quote(rel)})"


def vault_text(z):
    """Lesbarer Text: offene Stellen als '[?]', korrigierte als Korrektur."""
    def ersetze(m):
        st = z["stellen"].get(m.group(1))
        if not st:
            return ""
        if st.get("korrektur") is not None:
            return st["korrektur"]
        return "[?]" if st["lesart"] else "[unverständlich]"
    text = z["text"]
    # Lesart steht vor {{N}}; bei Korrektur wird die Lesart mit ersetzt
    for n, st in z["stellen"].items():
        if st.get("korrektur") is not None and st["lesart"]:
            text = text.replace(f"{st['lesart']} {{{{{n}}}}}", f"{{{{{n}}}}}", 1)
            text = text.replace(f"{st['lesart']}{{{{{n}}}}}", f"{{{{{n}}}}}", 1)
    text = re.sub(r"\{\{(\d+)\}\}", ersetze, text)
    text = re.sub(r"(?<=\S)\[\?\]", " [?]", text)
    offen = sum(1 for s in z["stellen"].values() if s.get("korrektur") is None)
    kopf = (f"---\ntags: [sprachmemo, transkript]\ndate: {z['datum']}\nstatus: "
            f"{'in-korrektur' if offen else 'korrigiert'}\n---\n\n# {z['titel']} — Sprachmemo {z['datum']}\n\n"
            f"Rohfassungen: {verweis(text_pfad(z), roh_pfad(z))} · "
            f"offene Stellen: {offen} von {len(z['stellen'])} · Korrektur: {korrektur_url()}\n\n")
    return kopf + re.sub(r"  +", " ", text).strip() + "\n"


def text_schreiben(z):
    ziel = text_pfad(z)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text(vault_text(z))
    return ziel


def roh_schreiben(z, fassungen):
    ziel = roh_pfad(z)
    if ziel.exists():
        return ziel                                   # Rohfassungen sind unveränderlich
    ziel.parent.mkdir(parents=True, exist_ok=True)
    teile = [f"---\ntitel: \"{z['titel']}\"\nherkunft: Sprachmemo {z['datum']}, maschinelle Fassungen "
             f"(turbo, large-v3, CrisperWhisper 2.0 verbatim)\n"
             f"erstellt: {z['erstellt']}\nstatus: erfasst\ntags: [quelle, sprachmemo, transkript]\n---\n",
             f"# {z['titel']} — Rohfassungen\n\nLesbarer Text: {verweis(ziel, text_pfad(z))}\n"]
    for name, text in fassungen.items():
        teile.append(f"## {name}\n\n```text\n{text}\n```\n")
    ziel.write_text("\n".join(teile))
    return ziel


def zustand_pfad(slug):
    return daten() / slug / "zustand.json"


def speichern(z):
    zustand_pfad(z["slug"]).write_text(json.dumps(z, ensure_ascii=False, indent=1))


def verarbeiten(a):
    ordner = daten() / a.slug
    (ordner / "clips").mkdir(parents=True, exist_ok=True)
    fj = ordner / "fassungen.json"
    erg = json.loads(fj.read_text()) if fj.exists() and not a.neu else worker_fassungen(Path(a.audio), a.slug, ordner)
    fassungen = dict(erg["fassungen"])
    m = verschmelzen(fassungen, erg.get("zeitbasis", "crisper"))
    wav = ordner / f"{a.slug}.wav"
    if not wav.exists():
        lauf(["ffmpeg", "-loglevel", "error", "-y", "-i", a.audio, "-ac", "1", "-ar", "16000", str(wav)])
    z = {"slug": a.slug, "datum": a.datum, "titel": a.titel or a.slug, "text": m["text"],
         "erstellt": datetime.date.today().isoformat(), "stellen": {}}
    for st in m["stellen"]:
        n = str(st["n"])
        zeit = verorten(st.get("zitat", ""), erg.get("woerter", []))
        eintrag = {"lesart": st.get("lesart", ""), "zitat": st.get("zitat", ""), "korrektur": None,
                   "zeit": zeit}
        if zeit:
            clip(wav, zeit[0], zeit[1], ordner / "clips" / f"{n}.m4a")
        z["stellen"][n] = eintrag
    speichern(z)
    print("Roh:", roh_schreiben(z, fassungen))
    print("Text:", text_schreiben(z))
    ohne = [n for n, s in z["stellen"].items() if not s["zeit"]]
    print(f"Stellen: {len(z['stellen'])}, davon ohne Zeit: {len(ohne)} {ohne}")
    return z


# ---------------------------------------------------------------- Eingang

def slug_aus(name):
    s = unicodedata.normalize("NFC", name).lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "memo"


def datum_titel(f):
    """Dateiname `YYYY-MM-DD Titel.m4a` → (Datum, Titel); ohne Datum zählt das Änderungsdatum."""
    stamm = unicodedata.normalize("NFC", f.stem)
    m = re.match(r"(\d{4}-\d{2}-\d{2})[\s_-]*(.*)", stamm)
    datum = m.group(1) if m else datetime.date.fromtimestamp(f.stat().st_mtime).isoformat()
    return datum, ((m.group(2) if m else stamm).strip() or stamm)


def melden(text):
    befehl = wert("melden", "befehl", [])
    if befehl:
        subprocess.run([*befehl, text], check=False, capture_output=True, timeout=30)


def eingang(a):
    """Jede Audiodatei im Eingangsordner verarbeiten, danach nach erledigt/ bzw. fehler/ schieben."""
    import fcntl
    ordner_ein = pfad("eingang", "")
    if not wert("pfade", "eingang"):
        sys.exit("[pfade] eingang ist nicht gesetzt.")
    daten().mkdir(parents=True, exist_ok=True)
    sperre = open(daten() / ".eingang.lock", "w")
    try:
        fcntl.flock(sperre, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return print("Eingang läuft schon.")
    for unter in ("erledigt", "fehler"):
        (ordner_ein / unter).mkdir(parents=True, exist_ok=True)
    for f in sorted(ordner_ein.iterdir()):
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() not in AUDIO:
            continue
        groesse = f.stat().st_size
        time.sleep(10)
        if f.stat().st_size != groesse:                          # wird noch geschrieben
            continue
        datum, titel = datum_titel(f)
        slug = slug_aus(titel)
        if zustand_pfad(slug).exists():          # Ordner ohne Zustand = abgebrochener Lauf → wiederverwenden
            slug = f"{slug}-{datum}"
        ordner = daten() / slug
        ordner.mkdir(parents=True, exist_ok=True)
        kopie = ordner / f"original{f.suffix.lower()}"
        print(f"{f.name} → {slug} ({datum})", flush=True)
        try:
            shutil.copy2(f, kopie)
            z = verarbeiten(argparse.Namespace(audio=str(kopie), datum=datum, slug=slug, titel=titel, neu=False))
            shutil.move(str(f), str(ordner_ein / "erledigt" / f.name))
            melden(f"„{titel}“ ({datum}) ist transkribiert: {len(z['stellen'])} Stellen zum Gegenhören.\n"
                   f"{korrektur_url()}?m={slug}")
        except Exception as e:
            shutil.move(str(f), str(ordner_ein / "fehler" / f.name))
            print(f"FEHLER {f.name}: {e}", file=sys.stderr, flush=True)
            melden(f"„{titel}“ ist fehlgeschlagen ({type(e).__name__}). Datei liegt in {ordner_ein / 'fehler'}.")


# ---------------------------------------------------------------- Korrekturseite

SEITE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sprachmemo-Korrektur</title>
<style>
body{font:20px -apple-system,sans-serif;margin:0;padding:16px;background:#111;color:#eee}
.k{background:#1e1e1e;border-radius:14px;padding:16px;margin:0 0 16px}
.kt{color:#aaa;font-size:17px;margin:8px 0}
.l{color:#ffd166}
audio{width:100%;margin:8px 0}
input{width:100%;box-sizing:border-box;font-size:22px;padding:12px;border-radius:10px;border:1px solid #555;background:#000;color:#fff}
button{font-size:22px;padding:14px 18px;border-radius:12px;border:0;margin:10px 8px 0 0}
.ok{background:#2a9d8f;color:#fff}.weg{background:#444;color:#fff}
h1{font-size:24px} a{color:#8ecae6}
.fertig{opacity:.45}
.wahl{display:flex;flex-direction:column;gap:8px;margin:0 0 16px}
.wahl a{background:#2b2b2b;color:#eee;text-decoration:none;padding:12px 16px;border-radius:12px;font-size:18px}
.wahl a.akt{background:#2a9d8f;color:#fff}
</style></head><body>__INHALT__
<script>
async function senden(slug,n,wert){
  const r=await fetch('/korrektur',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({slug:slug,n:n,korrektur:wert})});
  if(r.ok){document.getElementById('k'+slug+n).classList.add('fertig');}
}
</script></body></html>"""


def umfeld(z, n, breite=60):
    t = z["text"]
    i = t.find("{{%s}}" % n)
    if i < 0:
        return ""
    vor = re.sub(r"\{\{\d+\}\}", "", t[max(0, i - breite - len(z["stellen"][n]["lesart"])):i])
    nach = re.sub(r"\{\{\d+\}\}", "", t[i + len(n) + 4:i + len(n) + 4 + breite])
    les = z["stellen"][n]["lesart"]
    if les and vor.rstrip().endswith(les):          # rstrip vor dem Kürzen, sonst bleibt ein Buchstabe doppelt
        vor = vor.rstrip()[: -len(les)]
    return f"…{html.escape(vor)}<span class=l>{html.escape(les) or '???'}</span>{html.escape(nach)}…"


def seite(auswahl=None):
    memos = [json.loads(zp.read_text()) for zp in daten().glob("*/zustand.json")]
    memos.sort(key=lambda z: (z["datum"], z["slug"]), reverse=True)    # neueste zuerst
    if not memos:
        return SEITE.replace("__INHALT__", "<h1>Nichts offen.</h1>")

    def offen(z):
        return [n for n, s in z["stellen"].items() if s.get("korrektur") is None]
    z = next((m for m in memos if m["slug"] == auswahl), None) \
        or next((m for m in memos if offen(m)), memos[0])
    knoepfe = "".join(
        f"<a href='/?m={m['slug']}' class='{'akt' if m is z else ''}'>"
        f"{html.escape(m['titel'])} · {m['datum']} · {len(offen(m))}/{len(m['stellen'])}</a>"
        for m in memos)
    teile = [f"<nav class=wahl>{knoepfe}</nav>",
             f"<h1>{html.escape(z['titel'])} · {z['datum']} · offen {len(offen(z))}/{len(z['stellen'])}</h1>"]
    for n in offen(z):
        s = z["stellen"][n]
        audio = (f"<audio controls preload=none src='/clip/{z['slug']}/{n}.m4a'></audio>"
                 if s.get("zeit") else "<div class=kt>kein Clip (Stelle nicht verortet)</div>")
        wert_ = html.escape(s["lesart"], quote=True)
        teile.append(
            f"<div class=k id='k{z['slug']}{n}'><div class=kt>Stelle {n}</div>{audio}"
            f"<div class=kt>{umfeld(z, n)}</div>"
            f"<input id='i{z['slug']}{n}' value='{wert_}' autocomplete=off autocorrect=off>"
            f"<button class=ok onclick=\"senden('{z['slug']}','{n}',document.getElementById('i{z['slug']}{n}').value)\">Passt</button>"
            f"<button class=weg onclick=\"senden('{z['slug']}','{n}','')\">Weglassen</button></div>")
    if not offen(z):
        teile.append("<div class=kt>Alle Stellen dieser Aufnahme sind erledigt.</div>")
    return SEITE.replace("__INHALT__", "\n".join(teile))


def wortliste_ergaenzen(wort, slug):
    wl = ausgabe("wortliste", "Sprachmemos/Wortliste.md")
    if not wort or len(wort) > 40:
        return
    if not wl.exists():
        wl.parent.mkdir(parents=True, exist_ok=True)
        wl.write_text(
            f"---\ntags: [sprache, transkript]\ndate: {datetime.date.today().isoformat()}\n---\n\n"
            "# Sprach-Wortliste\n\nBestätigte Begriffe aus Sprachmemo-Korrekturen. Wird automatisch ergänzt "
            "und beim Verschmelzen als Kontext mitgegeben.\n\n| Begriff | aus |\n|---|---|\n")
    if f"| {wort} |" in wl.read_text():
        return
    with wl.open("a") as fh:
        fh.write(f"| {wort} | {slug} |\n")


def trainingspaar(z, n, korrektur):
    """Clip + bestätigter Text als Trainingsbeispiel, falls [training] an = true."""
    c = daten() / z["slug"] / "clips" / f"{n}.m4a"
    if not wert("training", "an", False) or not c.exists():
        return
    ziel = daten() / "training"
    ziel.mkdir(exist_ok=True)
    shutil.copy2(c, ziel / f"{z['slug']}-{n}.m4a")
    (ziel / f"{z['slug']}-{n}.txt").write_text(korrektur)


class Handler(BaseHTTPRequestHandler):
    def _senden(self, code, body, typ="text/html; charset=utf-8"):
        roh = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", typ)
        self.send_header("Content-Length", str(len(roh)))
        self.end_headers()
        self.wfile.write(roh)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p == "/":
            return self._senden(200, seite(urllib.parse.parse_qs(u.query).get("m", [None])[0]))
        m = re.fullmatch(r"/clip/([\w-]+)/(\d+)\.m4a", p)
        if m:
            f = daten() / m.group(1) / "clips" / f"{m.group(2)}.m4a"
            if f.exists():
                return self._senden(200, f.read_bytes(), "audio/mp4")
        self._senden(404, "nicht gefunden")

    def do_POST(self):
        if self.path != "/korrektur":
            return self._senden(404, "nicht gefunden")
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        slug, n, korr = d["slug"], str(d["n"]), d["korrektur"].strip()
        if not re.fullmatch(r"[\w-]+", slug) or not zustand_pfad(slug).exists():
            return self._senden(400, "unbekannt")
        z = json.loads(zustand_pfad(slug).read_text())
        if n not in z["stellen"]:
            return self._senden(400, "unbekannt")
        z["stellen"][n]["korrektur"] = korr
        speichern(z)
        text_schreiben(z)
        if korr and korr != z["stellen"][n]["lesart"]:
            wortliste_ergaenzen(korr, slug)
        if korr:
            trainingspaar(z, n, korr)
        self._senden(200, "ok")

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verarbeiten")
    v.add_argument("audio")
    v.add_argument("--datum", help="YYYY-MM-DD, sonst aus dem Dateinamen bzw. Änderungsdatum")
    v.add_argument("--slug")
    v.add_argument("--titel")
    v.add_argument("--neu", action="store_true", help="Fassungen neu erzeugen statt wiederverwenden")
    sub.add_parser("server")
    sub.add_parser("eingang")
    a = ap.parse_args()
    laden(a.config)
    if a.cmd == "verarbeiten":
        datum, titel = datum_titel(Path(a.audio))
        a.datum, a.titel = a.datum or datum, a.titel or titel
        a.slug = a.slug or slug_aus(a.titel)
        verarbeiten(a)
    elif a.cmd == "eingang":
        eingang(a)
    else:
        host, port = wert("server", "host", "127.0.0.1"), wert("server", "port", 8795)
        print(f"Korrekturseite: {korrektur_url()}", flush=True)
        ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
