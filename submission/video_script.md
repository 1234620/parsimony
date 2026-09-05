# Demo video script — 3 minutes

GIBC requires 2–5 minutes, English audio or subtitles. Screen recording with voiceover is fine;
you do not need to be on camera. **Record the demo page live** — judges can tell a real
interaction from a slideshow.

---

### 0:00–0:25 · The hook (screen: demo page, top)

> "The rules for this track cap models at 50 million parameters — and the cap *includes* token
> embeddings. That sounds like a footnote. It isn't."
>
> "With GPT-2's fifty-thousand-token vocabulary, the embedding table alone is 25.7 million
> parameters. Fifty-four percent of the entire budget is a lookup table, before you have a
> single transformer layer."

### 0:25–0:55 · The interactive moment (screen: drag the slider)

> "So vocabulary size isn't a preprocessing detail here — it's a capacity allocation decision."

**Drag the slider from 50k down to 8k. Let the layer blocks fill in.**

> "Same fifty million parameters. Seven layers becomes fourteen. The obvious guess is that depth
> wins, because depth is where the computation happens. That was our hypothesis — and it was
> wrong."

### 0:55–1:40 · The method (screen: results table, then scroll to the method note)

> "To answer it properly we had to control three things."
>
> "First, parameters — every configuration is re-solved to the same total count."
>
> "Second, compute — steps are solved per configuration so every run burns the same FLOPs.
> Matching steps would quietly favour shallow models, because they're cheaper per step."
>
> "Third, and this is the one people get wrong: you cannot compare perplexity across
> tokenizers. A bigger vocabulary packs more text into each token, so it predicts fewer, harder
> tokens and its perplexity improves without the model being any better. In our own sweep, token
> perplexity varies several-fold while bits-per-byte barely moves."

**Point at the two columns.**

> "So everything here is bits-per-byte — negative log likelihood normalised by the raw bytes of
> held-out text. The tokenizer drops out of the metric entirely."

### 1:40–2:20 · The finding (screen: the dual-axis chart)

> "Here's what we found. The teal line is compression. It's almost flat — how you split the
> budget barely affects it."
>
> "The amber line is reasoning. That is not flat."

**Pause on the divergence.**

> "And we nearly published a confounded version of this. Free-form generation made large
> vocabularies look dramatically better — but with a bigger vocabulary the answer is fewer
> tokens, so greedy decoding has fewer chances to slip. When we rescored every candidate with
> byte-normalised likelihood, the way ARC and HellaSwag are scored, the ranking changed. We
> report both numbers, because the gap between them is itself a result."

### 2:20–2:45 · The build (screen: terminal, run `python scripts/verify_params.py`)

> "Trained from scratch — no pretrained weights, no distillation. Modern small-scale stack:
> RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings."
>
> "Initialisation was verified against theory, not assumed: loss at init 9.065 against a
> predicted ln-V of 9.011."

**Let the parameter table print on screen — it proves every config is under the cap.**

### 2:45–3:00 · Close (screen: back to the demo page)

> "Two of the three mistakes that nearly cost us this result were measurement errors, not
> modelling errors. In a fixed-budget regime, the hard part isn't building the model. It's
> refusing to compare numbers that aren't comparable."

---

## Recording checklist

- [ ] Close every other tab; hide bookmarks bar
- [ ] Browser zoom 110–125% so text is readable when compressed
- [ ] Drag the slider **slowly** — the layer animation is the moment that lands
- [ ] Record at 1080p; upload to YouTube as **public or unlisted** (private fails judging)
- [ ] Watch it back once with sound off to confirm subtitles/visuals carry it
