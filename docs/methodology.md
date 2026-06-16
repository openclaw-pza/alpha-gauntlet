# The gauntlet methodology

> TL;DR: factor discovery is an industrial-scale false-positive machine. This
> framework's design assumes that almost everything you find is overfit, and
> makes survival expensive on purpose. The result is fewer "discoveries" and a
> much higher base rate of the survivors actually working out of sample.

This document describes the end-to-end pipeline a candidate factor (or entry
filter) must survive before it is trusted. It is deliberately adversarial: every
stage is a different way to *kill* a candidate, not to bless it.

## 0. The problem: false champions

If you evaluate enough candidate signals against the same historical data, some
will look excellent purely by chance. Standard significance tests assume one
hypothesis; a factor mine tests thousands. The "best" factor from a wide search
is, by construction, the one that most exploited the noise in your sample. We
call it a **false champion**: it tops the leaderboard in-sample and dies the
moment you look at fresh data.

The gauntlet is six stages designed to make a false champion improbable.

## 1. Blind generation

Candidate factors are generated **without touching the data they will be judged
on**. The generator never sees the IC table, the returns, or the leaderboard. A
generator that can see its own scoreboard will (consciously or not) curve-fit to
it; blinding removes that feedback loop. Generated candidates are mechanical
formulas over OHLCV-derived fields, not hand-tuned-against-the-answer constructs.

## 2. Pre-registration + hash-freeze

Before any evaluation:

- The full candidate set is written down and its count `m` is fixed.
- The survival threshold is computed for `m` (see stage 3) and frozen.
- The candidate set + threshold are serialised and **hashed (SHA-256)**.

The hash is recorded. From this point the bar cannot move and candidates cannot
be quietly added or dropped to flatter the survivors. This is the same discipline
as pre-registering a clinical trial: decide the endpoint before you see the data.

## 3. Bonferroni-anchored t-threshold

With `m` candidates tested, a per-test significance of α inflates the
family-wise error rate. We anchor the survival threshold using a Bonferroni-style
correction: the required `|t|` is set for the *family* of `m` tests, not for one.
For example, with `m ≈ 49` candidates and a family-wise α, the per-candidate
threshold lands around `|t| ≳ 4.4` rather than the naive `2.0`. Anchoring the
threshold to `m` is what stops "I tried 50 things and one had t = 2.1" from
counting as a discovery.

## 4. Multi-round survival

A candidate that clears the pre-registered threshold then has to survive several
independent screens, each cutting from a different angle:

1. **Full-sample significance** -- the anchored `|t|` threshold on the whole
   sample, with the t-stat computed on **non-overlapping** forward windows and
   shrunk for IC-series autocorrelation (overlapping windows manufacture fake
   significance; we explicitly correct the effective sample size).
2. **Year-by-year sign stability** -- the factor's IC sign must agree across most
   individual years. A factor that is strongly positive one year and strongly
   negative the next is not a stable edge, it is regime noise.
3. **Low correlation to incumbents** -- a survivor must be reasonably orthogonal
   to factors already in the book (e.g. `|ρ| < 0.7`). A new factor that is just a
   re-expression of momentum adds nothing.

## 5. Decay-lens kill-shot

Two final, ruthless checks aimed at the most common ways a "real" factor is
actually a mirage:

- **Re-skin check** -- is the candidate just a non-linear re-skin of an existing
  factor (high rank-correlation to an incumbent through a different formula)? If
  the apparent edge collapses once you control for the incumbent, it is killed.
- **Recent-decay check** -- has the edge already decayed? A factor with a great
  full-sample t-stat but a flat or negative recent window has likely been arbed
  away; it is killed (or flagged for live monitoring, never silently trusted).

## 6. Walk-forward ratchet (the promotion gate)

Surviving factors do not get "deployed". They become *candidates* for the
**ratchet evolution engine** (`alphagauntlet.evolution`), which governs the only
thing that ever changes a champion. The ratchet enforces, in code:

- **Strict-greater-than promotion (G1/G5).** A challenger is the merge of the
  current champion with the candidate, re-evaluated against the *current*
  champion. It is promoted only if its pooled Sharpe improvement exceeds an
  epsilon that **inflates with the number of attempts in the same family**
  (`ε ∝ √(1 + ln(k+1))`) -- multiple-testing control baked into the gate itself.
- **Multi-segment, non-overlapping out-of-sample (G2).** Several selection
  segments + a forward recheck segment + a **never-touched holdout segment**,
  with aligned boundaries so they cannot overlap.
- **Conjunctive guardrails (G3).** Per-segment Sharpe and max-drawdown must not
  regress, a minimum trade count must be met, signal retention must stay above a
  floor, and pooled total profit must not shrink (you cannot fake a Sharpe lift
  by cutting exposure).
- **The holdout kill-shot.** Even after passing every selection gate, the
  challenger is burned once against the holdout segment it has never seen. If it
  degrades there -- the textbook overfit signature -- it is rejected. This is the
  single most effective false-champion filter in the pipeline.
- **Tamper-evident, replayable ledger (G4).** Every promotion, rejection,
  demotion and recheck is appended to a SHA-256 **hash chain**. The champion is a
  pure projection of the ledger and can be deterministically replayed; any edit
  to history breaks the chain and is detected on read. Writes are atomic
  (`os.replace`) and serialised behind a single-writer lock.
- **Auto-safe rollback.** A periodic recheck re-evaluates the champion against a
  fixed baseline on the forward-OOS segment. After a hysteresis window of
  consecutive degradations it **demotes** the most recently promoted keys,
  ratcheting back toward the safer state -- the engine can only move toward
  monotonic improvement, never silently drift worse.

**Rejection is knowledge.** Every kill is recorded with a structured reason code
and a *direction-only* hint (never the exact failing threshold, which would let a
search learn the gate's shape and overfit to it). The ledger is the institutional
memory of everything the gauntlet has already killed and why.

## Why this is honest, not pessimistic

The gauntlet rejects most candidates, and that is the point. A pipeline that
promotes often is a pipeline that has not corrected for how many things it tried.
The honest output of factor research is usually "nothing survived" -- and a
framework that can say that confidently is more valuable than one that always
finds a champion.

---

*This is a research methodology, not financial advice. Surviving the gauntlet
means a candidate is less likely to be a statistical artefact -- it does not mean
it will be profitable in live trading.*
