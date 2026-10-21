"""Head scores for the exemplar-label ICL task (Singh et al., 2024). NumPy only.

Token layout produced by SequenceClassifier (models.py): items are interleaved
as  x_0 y_0 x_1 y_1 ... x_k  with the last item's label NOT embedded, so
  item i's exemplar sits at token 2i, its label at token 2i+1, T = 2*n_items - 1.
Base runs (train_context_len 2):  x0 y0 x1 y1 xq            T = 5, query = item 2
Extended runs:                    x0 y0 x1 y1 xq yq xn      T = 7, query = item 2,
                                                            next symbol = item 3

attn convention: attn[l, b, h, dest, src], rows sum to 1 over causal sources.

For a "reader" item k (the query, or the extended next symbol), the candidate
label tokens are the labels of items j < k; the correct ones are those whose
exemplar has the reader's class.

Names follow the team guide (section 3.1; Olsson et al. 2022 adapted to our sequences):
  prev_token[l,h]    previous-token score: mean over label tokens t of attn[t -> t-1]
                     (each label reads the exemplar just before it)
  induction[l,h]     induction score: attention from the reader (query) to the correct label token(s)
  matching[l,h]      matching score: attention to the correct label minus attention to the
                     distractor's label (= Singh et al.'s "induction strength"; 1 perfect, 0 indifferent)
  exemplar_attn[l,h] attention from the reader to the matching *exemplar* (a control: a head that
                     finds the match but does not read the label)
Layers are 0-indexed here (L0, L1). The team guide counts from 1 ("layer 1" = L0, "layer 2" = L1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Reader:
  name: str
  pos: int                  # reader token position (2k)
  correct: np.ndarray       # [B, T] bool, correct label tokens
  incorrect: np.ndarray     # [B, T] bool, other candidate label tokens

  @property
  def match_exemplar(self) -> np.ndarray:
    m = np.zeros_like(self.correct)
    m[:, :-1] = self.correct[:, 1:]          # exemplar is one token before its label
    return m


def make_reader(class_idxs: np.ndarray, k: int, name: str) -> Reader:
  """class_idxs [B, n_items]; reader = item k (k >= 1)."""
  B, n = class_idxs.shape
  T = 2 * n - 1
  correct = np.zeros((B, T), bool)
  incorrect = np.zeros((B, T), bool)
  same = class_idxs[:, :k] == class_idxs[:, k:k + 1]          # [B, k]
  correct[:, 1:2 * k:2] = same
  incorrect[:, 1:2 * k:2] = ~same
  return Reader(name, 2 * k, correct, incorrect)


def label_positions(n_items: int) -> np.ndarray:
  """Embedded label tokens (all labels except the last item's)."""
  return np.arange(1, 2 * n_items - 1, 2)


def prev_token_score(attn: np.ndarray, n_items: int) -> np.ndarray:
  t = label_positions(n_items)
  return attn[:, :, :, t, t - 1].mean(axis=(1, 3))           # [L, H]


def _mass(attn, pos, mask):
  row = attn[:, :, :, pos, :]                                  # [L, B, H, T]
  m = (row * mask[None, :, None, :]).sum(-1)                   # [L, B, H]
  has = mask.any(1) if mask.ndim == 2 else np.ones(row.shape[1], bool)
  return m[:, has].mean(1)


def reader_scores(attn: np.ndarray, r: Reader) -> dict:
  ind = _mass(attn, r.pos, r.correct)
  wrong = _mass(attn, r.pos, r.incorrect) if r.incorrect.any() else np.zeros_like(ind)
  return {"induction": ind, "matching": ind - wrong, "exemplar_attn": _mass(attn, r.pos, r.match_exemplar)}


def chance(r: Reader) -> float:
  """Mass on correct labels under uniform causal attention."""
  has = r.correct.any(1)
  return float(r.correct[has].sum(1).mean() / (r.pos + 1))


def all_scores(attn: np.ndarray, readers: list[Reader], n_items: int) -> dict:
  out = {"prev_token": prev_token_score(attn, n_items)}
  for r in readers:
    for k, v in reader_scores(attn, r).items():
      out[k if r.name == "query" else f"{r.name}_{k}"] = v
    out[("" if r.name == "query" else r.name + "_") + "chance"] = chance(r)
  return out


def pick_heads(scores: dict, ind_key: str = "matching", ind_min_layer: int = 1, pt_set_min: float = 0.5) -> dict:
  """Rule (state it in the write-up): induction head = argmax of the matching score in
  layers >= ind_min_layer; previous-token head = argmax prev-token score in earlier layers;
  prev_token_set = every earlier-layer head with prev-token score >= pt_set_min (there can
  be several previous-token heads; Singh et al. report three)."""
  ind = np.array(scores[ind_key], float)
  ind[:ind_min_layer] = -np.inf
  li, hi = np.unravel_index(np.argmax(ind), ind.shape)
  pt = scores["prev_token"][:max(li, 1)]
  lp, hp = np.unravel_index(np.argmax(pt), pt.shape)
  pt_set = [[int(a), int(b)] for a, b in zip(*np.nonzero(pt >= pt_set_min))] or [[int(lp), int(hp)]]
  return {"induction": [int(li), int(hi)], "prev_token": [int(lp), int(hp)], "prev_token_set": pt_set}


# ---------------------------------------------------------------- per-sequence values
def per_sequence(attn: np.ndarray, r: Reader, n_items: int, ih: tuple, pt_heads: list) -> dict:
  """Per-probe-sequence values [B] for the chosen heads (for paired tests across generations)."""
  l, h = ih
  row = attn[l, :, h, r.pos, :]                                   # [B, T]
  ind = (row * r.correct).sum(-1)
  wrong = (row * r.incorrect).sum(-1)
  t = label_positions(n_items)
  pt = np.mean([attn[a, :, b][:, t, t - 1].mean(-1) for a, b in pt_heads], axis=0)
  return {"induction": ind, "matching": ind - wrong, "prev_token": pt}


def first_move(x0: np.ndarray, xs: list[np.ndarray], worse: str, z: float = 3.0):
  """First generation g >= 1 whose per-sequence values are worse than generation 0's by more
  than z paired standard errors (same probe sequences in every generation).
  worse = 'down' (accuracy, scores) or 'up' (loss). Returns (gen or None, list of z per gen)."""
  zs = [0.0]
  first = None
  for g, x in enumerate(xs[1:], start=1):
    d = np.asarray(x, float) - np.asarray(x0, float)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    zz = float(d.mean() / se) if se > 0 else (0.0 if d.mean() == 0 else np.sign(d.mean()) * np.inf)
    zz = -zz if worse == "down" else zz                          # positive = worse
    zs.append(zz)
    if first is None and zz > z:
      first = g
  return first, zs


def rise_points(iters, y, fracs=(0.1, 0.5, 0.9)) -> dict:
  """Iterations at which y completes 10/50/90% of its rise from the first value to its peak
  (same definition as find_phases.py), linearly interpolated."""
  it, y = np.asarray(iters, float), np.asarray(y, float)
  start, peak = y[0], y.max()
  out = {"start": float(start), "peak": float(peak)}
  for f in fracs:
    level = start + f * (peak - start)
    above = np.nonzero(y >= level)[0]
    if peak - start <= 0 or not above.size:
      out[f"iter_{int(f * 100)}pct"] = None
      continue
    j = above[0]
    out[f"iter_{int(f * 100)}pct"] = float(it[0]) if j == 0 else float(
      it[j - 1] + (level - y[j - 1]) * (it[j] - it[j - 1]) / (y[j] - y[j - 1]))
  return out


def first_crossing(series, frac: float = 0.9):
  """First index g >= 1 with series[g] < frac * series[0]; None if never."""
  s = np.asarray(series, float)
  below = np.where(s[1:] < frac * s[0])[0]
  return int(below[0] + 1) if below.size else None


def lead_time(metric, outcome, frac: float = 0.9) -> dict:
  """Generations between the head metric's drop and the outcome's drop.
  Positive lead = the head metric dropped first (an early warning)."""
  gm, go = first_crossing(metric, frac), first_crossing(outcome, frac)
  return {"metric_drop_gen": gm, "outcome_drop_gen": go,
          "lead": None if gm is None or go is None else go - gm, "threshold": frac}


def lagged_spearman(metric, outcome, lag: int = 1) -> float:
  """Spearman corr of change in metric at g with change in outcome at g+lag.
  With ~10 generations this is descriptive only."""
  from scipy.stats import spearmanr

  dm, do = np.diff(np.asarray(metric, float)), np.diff(np.asarray(outcome, float))
  if lag:
    dm, do = dm[:-lag], do[lag:]
  if len(dm) < 3 or np.std(dm) == 0 or np.std(do) == 0:
    return float("nan")
  return float(spearmanr(dm, do).statistic)


def formation_iter(iters, series, level: float = 0.5):
  """First training iteration where a score reaches `level` (linear interpolation)."""
  it, s = np.asarray(iters, float), np.asarray(series, float)
  above = np.nonzero(s >= level)[0]
  if not above.size:
    return None
  j = above[0]
  if j == 0:
    return float(it[0])
  return float(it[j - 1] + (level - s[j - 1]) * (it[j] - it[j - 1]) / (s[j] - s[j - 1]))
