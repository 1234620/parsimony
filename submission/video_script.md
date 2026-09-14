# Demo video script — ~3:20

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

### 1:40–2:40 · The finding (screen: the dual-axis chart, then flagship numbers)

> "Here's what we found. The teal line is compression, the amber line is reasoning — and they
> agree. Both peak at the same vocabulary: sixteen thousand three hundred eighty-four tokens.
> Not the smallest vocabulary we tested — that's worst on both. Not the largest we finished
> either — that gives a little back on both."

**Pause on the peak.**

> "A smaller pilot on synthetic data first suggested these two capabilities trade off against
> each other. At full scale, on real text, they don't. We trained that winning configuration at
> the full fifty-million-parameter budget — Kaggle's twelve-hour session cap actually killed the
> first run at step ten thousand of twelve thousand, and our checkpoint-resume path picked it
> back up and finished. Final numbers: point nine four bits-per-byte, ninety-nine percent
> reasoning accuracy."
>
> "One thing we're upfront about: that reasoning number is free-generation accuracy, not the
> byte-normalised rescoring our own pilot showed can inflate large-vocabulary scores by masking
> a tokenization artifact. We didn't re-run that check at this scale — so some of this gap might
> still be that same artifact, not pure capability."

### 2:40–3:05 · The build (screen: terminal, run `python scripts/verify_params.py`)

> "Trained from scratch — no pretrained weights, no distillation. Modern small-scale stack:
> RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings."
>
> "Initialisation was verified against theory, not assumed: loss at init 9.065 against a
> predicted ln-V of 9.011."

**Let the parameter table print on screen — it proves every config is under the cap.**

### 3:05–3:20 · Close (screen: back to the demo page)

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
