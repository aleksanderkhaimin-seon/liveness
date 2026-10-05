# Anonymised production data for document liveness — feasibility by attack type

> **Status: DRAFT** · 2026-10-05 · Owner: Aleksandr Khaimin · Space: IDV ML
> Related: *Datasets impact evaluation*; repo `liveness` → `docs/anonymisation_evidence.md`, `docs/anonymisation_summary.md`

## TL;DR

We want to train document presentation-attack detectors on production captures without moving identity content out of the trusted zone. Whether that is possible depends on **where the evidence of the attack lives relative to the identity content**. For screen replay it lives in the scene and the coarse frame; for print it is mostly fine texture; for portrait substitution and digital manipulation it *is* the identity content.

| Attack type | Where the evidence lives | Survives anonymisation? | Verdict | Evidence status |
|---|---|---|---|---|
| **Screen replay** | Surroundings (bezel, glare, framing), coarse colour cast, document-in-frame geometry | Yes — coarse signal transfers to production intact | **Feasible.** Within seed noise of unmodified frames on production data | Measured (Sept–Oct 2026, 25+ runs) |
| **Printed copy** | Halftone / dot pattern, paper texture, missing optically variable features, flat specular behaviour | Partially — fine texture is destroyed, coarse reflectance cues remain | **Feasible with substantial loss** (preliminary) | Not yet measured — reasoning only, see §3 |
| **Portrait substitution** | The portrait region: paste edges, lighting / grain mismatch, photo-on-photo texture | No — the portrait is the identity content | **Not feasible, even in principle** | By construction, see §4 |
| **Digital manipulation** | Pixel-level forensics in the fields: resampling traces, double compression, font / kerning inconsistencies, edited glyphs | No — the fields are the identity content and the traces are sub-pixel | **Not feasible, even in principle** | By construction, see §5 |

**The general rule:** anonymisation of training images is possible exactly when the attack signal is spatially or spectrally separable from the identity content. Screen replay is; print is marginal; the other two are not. For those, the route to production data is not anonymised images but a different mechanism (§6).

---

## 1. What "anonymisation" means here

The representation under evaluation is the **whole frame downscaled so the document's long side is 96 px** (`export_lowres_frames.py --mode doc --size 96`). A typical ID card becomes 96 × 60 px; the frame is 160–220 px wide.

- **Identity content at that scale** (geometric proxy, all three splits, both classes): field text p90 ≤ 3.1 px (100 % under the 5 px OCR threshold), face p90 ≤ 27 px (100 % under the 40 px face-matching threshold). An empirical OCR / face-match audit is pending the availability of those detectors internally and is a condition of use beyond the research team.
- **What survives:** document type and layout, dominant colours, screen bezel / glare shape, the room or desk, how large the document is in the frame.
- **Custody:** opaque identifiers, shuffled order; the manifest mapping identifier → source is the re-identification map and never leaves the trusted zone.

Alternatives tested and rejected (production EER, 3 epochs; unmodified baseline 14.95 %): document interior masked 19.7 %, interior pixelated 19.6 %, document only without surroundings 24.8 %, native-resolution texture patches 38.0 % (chance). Details in `docs/anonymisation_summary.md`.

---

## 2. Screen replay — feasible

**Evidence.** Production validation = `ProdTest-0.3` (3,739 frames, two independent reviewers, all attacks screen replays). EER, lower is better; seed-to-seed spread at identical settings is ~2 points on production.

| Training input | Production EER |
|---|---|
| Full frame, unmodified (baseline), 10 epochs, 2 seeds | 13.7 / 12.2 |
| **Document long side → 96 px, 10 epochs, 2 seeds** | **12.1 / 14.3** |
| Same, trained on the exported files rather than a training-time transform, 3 epochs | 15.4 (vs 16.2 for the transform) |
| Same, network input reduced from 512 to 224 px, 3 epochs, 2 seeds | 13.7 / 13.8 — and 3× training throughput |

**Why it works.** The decisive experiment is the one that failed: native-resolution 512 px patches reach 1.7 % EER per document on the in-distribution test set — better than the baseline — and 38 % (chance) on production. The fine texture (moiré, pixel grid) that patches capture belongs to the ten phone × monitor combinations of the collection, not to screen replay as a phenomenon. What transfers to production is coarse: bezel, glare, colour cast, framing. Removing the fine detail therefore removes nothing that generalises, and the document interior contributes little beyond its border (masking it costs ~4 points; removing the *surroundings* costs ~10).

**Caveats.** (a) The current production EER is 12–15 % regardless of anonymisation — the training data does not transfer; production data in training is the remedy, which is what anonymisation enables. (b) At a 0.5 threshold 55–75 % of production replays are missed in every configuration: a production-derived operating point is needed whatever the training data. (c) `ProdTest-0.3` currently serves both checkpoint selection and reporting; a split or a second production set is needed for unbiased numbers.

---

## 3. Printed copy — feasible with substantial loss (preliminary, not measured)

**Where the evidence lives.** A printed reproduction differs from a genuine document in (i) **halftone / dithering pattern** of the printer, (ii) **paper texture** and ink spread, (iii) **absence of optically variable features** — holograms, OVI, laser-engraved relief that catch light, (iv) **flat, diffuse specular behaviour** instead of the laminate's sharp highlights, (v) colour gamut shift.

**What the anonymisation does to it.** (i) and (ii) are fine texture — exactly what a downscale to 96 px removes, and exactly the class of signal that failed to transfer for screen replay. (iii)–(v) are coarse: the shape and sharpness of highlights across the document, the overall colour, the way the surface responds to light — these survive the downscale and are the kind of cue that did transfer for screen replay.

**Preliminary assessment.** The coarse cues are weaker and less distinctive for print than the bezel-and-glare cues are for screen replay (a print on a desk has no bezel; its glare is just dimmer), so we expect a working detector with a materially higher error rate than a full-resolution one. "Possible, but with big losses." Whether the loss is 3 points or 15 is unknown until the same matrix is run on a print dataset.

**To quantify:** `[TBD]` print-attack training set and a production-labelled print validation set; then the baseline vs `downscale_doc:96` comparison at two seeds, exactly as in §2. The degrade family and the launcher already support it.

---

## 4. Portrait substitution — not feasible, even in principle

**The attack.** A different person's photograph pasted or printed over the portrait of a genuine physical document.

**Where the evidence lives.** In the portrait region, at fine scale: the edge of the pasted photo, a mismatch in grain, lighting, colour temperature or print process between the portrait and the rest of the document, photo-on-photo texture, misalignment with security printing that overlaps the portrait.

**Why anonymisation cannot help.** The portrait *is* the identity content. Any transform that makes the face unmatchable removes the texture, edges and lighting detail the detector needs; at 96 px document width the portrait is ~22 px tall, below both the matching threshold and the detection evidence. Masking or pixelating the portrait removes the evidence outright. There is no representation that keeps the evidence and removes the identity, because they are the same pixels. This holds regardless of model architecture.

**Routes that do not require anonymised images:** see §6.

---

## 5. Digital manipulation — not feasible, even in principle

**The attack.** A genuine capture edited after the fact: changed name, date of birth, number or expiry; a replaced portrait inserted digitally; sometimes a wholly generated document image.

**Where the evidence lives.** In the pixel statistics of the fields: resampling and interpolation traces, double JPEG compression grids, inconsistent noise levels between edited and untouched regions, font rendering and kerning that does not match the issuer's template, misaligned baselines, copy-move repetitions. Much of it is sub-pixel or single-pixel in nature.

**Why anonymisation cannot help.** The fields are the identity content, and the forensic traces live at a finer scale than the text itself. A downscale that makes a 28 px name illegible (to ~2.5 px) has destroyed every compression grid and resampling trace in it many times over; masking the fields removes the edited region entirely. Unlike screen replay, nothing in the surroundings carries the signal — a digitally edited document sits in an ordinary scene. "Not even in theory" is exact here: the information-theoretic content needed for detection is contained in the content that must be removed.

---

## 6. If anonymised images are not the route — what is

For portrait substitution and digital manipulation (and to shore up print), production data can still inform training without identity content leaving the trusted zone:

| Mechanism | How | Notes |
|---|---|---|
| **Train inside the production VPC, export weights only** | Ephemeral GPU in the trusted zone, existing containerised pipeline, only the model leaves | Cleanest legally; organisational, not technical. Weights can memorise — a DPIA question, mitigable with differential privacy at an accuracy cost |
| **Synthetic attacks on synthetic documents** | Apply the manipulation (portrait paste, field edit, recompression) to generated documents (DocXpand) and to *consented* collection captures | Especially natural for digital manipulation: the attack is a procedure, so it can be synthesised without any real PII. Production statistics (device, compression, resolution) can be matched without moving images |
| **Feature / score telemetry** | Export model scores, decisions, device class, image quality statistics — no pixels | Drives threshold calibration, drift detection and failure-mode clustering; does not train a CNN |
| **Human review of a small, access-controlled sample** | Viewer-only, logged access, short retention, evaluation only | Measures production error without creating a training corpus |

---

## 7. Open questions / next steps

1. DPO classification of the 96 px representation for screen-replay training (`docs/anonymisation_evidence.md`): anonymised vs pseudonymised, and approval of the export procedure.
2. Empirical OCR / face-match audit of exported frames once detectors are available.
3. Print: assemble a print-attack training set and production-labelled print validation; run the §2 comparison.
4. Portrait substitution / digital manipulation: decide between in-VPC training and synthetic generation as the production-data route; both can start without waiting on the DPO outcome.
5. Production operating point: derive the threshold from production data; the EER-threshold on production is 10⁻⁴–10⁻², so a 0.5 threshold misses most replays.
6. `ProdTest-0.3`: split into selection and reporting halves, or obtain a second production set.

---

*Reproducibility: repository `liveness` on `main`; run reports under `runs/train_reports/` (Sept-28, Sept-29, Oct-01); export summaries `runs/train_reports/Sept-28/summary-{train,val,test}.json`; one-table summary `docs/anonymisation_summary.md`; evidence write-up `docs/anonymisation_evidence.md`.*
