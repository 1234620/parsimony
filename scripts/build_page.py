"""Inject measured results into the demo page; emit artifact + standalone builds."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
body = (root / "app" / "page_body.html").read_text()
res_path = root / "results" / "ablation_small" / "results.json"

if res_path.exists():
    rows = json.loads(res_path.read_text())
    keep = ("vocab_actual","n_layers","embed_fraction","bytes_per_token","steps",
            "bits_per_byte","token_ppl","reasoning_acc","reasoning_choice_acc")
    rows = [{k: r.get(k) for k in keep} for r in rows]
else:
    rows = []

body = body.replace("__RESULTS_JSON__", json.dumps(rows))
(root / "app" / "artifact_page.html").write_text(body)

standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
              + body.replace("<title>Parsimony</title>", "<title>Parsimony</title>") +
              "\n</body>\n</html>\n")
# move the <div class="wrap"> content inside body properly
standalone = standalone.replace('<div class="wrap">', '</head>\n<body>\n<div class="wrap">', 1)
(root / "app" / "index.html").write_text(standalone)
print(f"injected {len(rows)} result rows")
print("wrote app/artifact_page.html and app/index.html")
