"""Copy the demo into a site checkout (apps/credit-lab/) with cache-busting version stamps.

Usage: python scripts/sync_site.py ../website
Writes <site>/apps/credit-lab/{lab-app.js, lab-engine.js, lab.css, data/cases.json} and prints the snippet the
project page embeds. The version is a hash of the four files, so a changed file gets a new URL and browsers never
mix an old engine with a new app.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["lab-app.js", "lab-engine.js", "lab.css", "data/cases.json"]


def main(site: str):
    web = ROOT / "web"
    h = hashlib.sha256()
    for f in FILES:
        h.update((web / f).read_bytes())
    v = h.hexdigest()[:10]
    dest = Path(site) / "apps" / "credit-lab"
    (dest / "data").mkdir(parents=True, exist_ok=True)
    app = (web / "lab-app.js").read_text(encoding="utf-8")
    app = app.replace('from "./lab-engine.js"', f'from "./lab-engine.js?v={v}"')
    app = app.replace('new URL("./data/cases.json", import.meta.url)', f'new URL("./data/cases.json?v={v}", import.meta.url)')
    (dest / "lab-app.js").write_text(app, encoding="utf-8", newline="\n")
    for f in FILES[1:]:
        (dest / f).write_bytes((web / f).read_bytes())
    print(f"synced to {dest} (version {v})")
    print(f'<link rel="stylesheet" href="/apps/credit-lab/lab.css?v={v}">')
    print('<div id="credit-lab" class="clab"></div>')
    print(f'<script type="module" src="/apps/credit-lab/lab-app.js?v={v}"></script>')
    return v


if __name__ == "__main__":
    main(sys.argv[1])
