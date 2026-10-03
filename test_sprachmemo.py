"""Tests ohne GPU, Docker und Claude:  python3 -m unittest -v"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sprachmemo as sm


class Grundlagen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        sm.C.clear()
        sm.C.update({"pfade": {"daten": str(t / "daten"), "ausgabe": str(t / "aus")},
                     "server": {"host": "127.0.0.1", "port": 8795}})

    def tearDown(self):
        self.tmp.cleanup()

    def test_slug(self):
        self.assertEqual(sm.slug_aus("Hauptstraße 2"), "hauptstrasse-2")
        self.assertEqual(sm.slug_aus("???"), "memo")

    def test_datum_titel(self):
        f = Path(self.tmp.name) / "2024-02-01 Einkauf Graz.m4a"
        f.write_bytes(b"")
        self.assertEqual(sm.datum_titel(f), ("2024-02-01", "Einkauf Graz"))

    def test_verorten(self):
        woerter = [{"w": w, "s": i, "e": i + 0.5} for i, w in enumerate("ich bin heute nach Graz gefahren".split())]
        self.assertEqual(sm.verorten("nach Graz gefahren", woerter), (3, 5.5))
        self.assertIsNone(sm.verorten("völlig anderer Satz hier", woerter))

    def test_text_offen_und_korrigiert(self):
        z = {"slug": "t", "datum": "2026-01-01", "titel": "T", "erstellt": "2026-01-01",
             "text": "Ich war in Linz {{1}} beim Bäcker {{2}}.",
             "stellen": {"1": {"lesart": "Linz", "korrektur": None},
                         "2": {"lesart": "", "korrektur": None}}}
        t = sm.vault_text(z)
        self.assertIn("Ich war in Linz [?] beim Bäcker [unverständlich].", t)
        self.assertIn("offene Stellen: 2 von 2", t)
        self.assertIn("[2026-01-01-t-fassungen](roh/2026-01-01-t-fassungen.md)", t)
        z["stellen"]["1"]["korrektur"] = "Gleisdorf"
        self.assertIn("Ich war in Gleisdorf beim Bäcker", sm.vault_text(z))

    def test_obsidian_link(self):
        sm.C["ausgabe"] = {"obsidian": True}
        z = {"slug": "t", "datum": "2026-01-01"}
        self.assertEqual(sm.verweis(sm.text_pfad(z), sm.roh_pfad(z)), "[[Sprachmemos/roh/2026-01-01-t-fassungen]]")

    def test_worker_ssh_befehl(self):
        sm.C["worker"] = {"ssh": "gpu-host", "arbeit": "sm-arbeit", "gpu": True}
        ordner = Path(self.tmp.name) / "o"
        ordner.mkdir()
        aufrufe = []

        def falsch(cmd, **kw):
            aufrufe.append(cmd)
            if cmd[0] == "scp" and cmd[-1].endswith("fassungen.json"):
                Path(cmd[-1]).write_text('{"fassungen": {}}')
            return ""
        with mock.patch.object(sm, "lauf", falsch):
            sm.worker_fassungen(Path("x.m4a"), "memo", ordner)
        docker = next(c[2] for c in aufrufe if c[0] == "ssh" and "docker" in c[2])
        self.assertIn('-v "$HOME/sm-arbeit":/arbeit', docker)
        self.assertIn("--gpus all", docker)
        self.assertIn("/arbeit/memo.wav", docker)


if __name__ == "__main__":
    unittest.main()
