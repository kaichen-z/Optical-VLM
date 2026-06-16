# May 31 — Progress Report

Model (both methods): **MedGemma-27b-it, zero-shot (no fine-tuning)**.
Test set: 120 balanced images (40 not-likely / 40 borderline / 40 likely).

## How the model is called (shared by both methods)

```
┌─────────────────────────────────────────────┐
│  VLM INPUT PROMPT                            │
│                                             │
│  [1] 6 worked CoT examples  (2 per class)   │
│       each = 🖼 fundus image                 │
│             + 6-step reasoning              │
│             + final diagnosis               │
│                                             │
│  [2] Current case                           │
│       = 🖼 fundus image                      │
│         + predicted clinical indicators:    │
│             • CDR (vertical / horizontal)   │
│             • rim (ISNT)   ◄── differs A/B  │
│             • 6 glaucomatous signs          │
└─────────────────────────────────────────────┘
              │
              ▼
   MedGemma-27b  →  6-step CoT  →  diagnosis
```

- **[1] Few-shot teaching** — 6 examples, 2 per class (not-likely / borderline / likely); each = image → reasoning → answer.
- **[2] The case to diagnose** — its image + indicators **predicted by our RETFound heads** (deployable, no ground truth).
- **Only difference between Method A and B → how the `rim (ISNT)` field is encoded.** Everything else identical.

### What a current case actually looks like (real example, test id 026)

Same image + same CDR + same 6 signs in both. **Only the rim line changes:**

**Method A — rim = order only**
```
[🖼 fundus image]
Measured clinical indicators for THIS patient:
  Vertical CDR: 0.737 | Horizontal CDR: 0.658
  ISNT observed order: N>I>T>S            ◄── rim (A)
  Notching: present | RNFL defect: present | Disc hemorrhage: absent
  Bayoneting: present | Beta-zone atrophy: present | PPA: present
```

**Method B — rim = order + per-quadrant status**
```
[🖼 fundus image]
Measured clinical indicators for THIS patient:
  Vertical CDR: 0.737 | Horizontal CDR: 0.658
  ISNT observed order: N>I>T>S            ◄── rim (B, line 1, same as A)
  Per-quadrant rim status: Inferior: severe; Superior: mild;
                           Nasal: severe; Temporal: severe   ◄── rim (B, NEW line)
  Notching: present | RNFL defect: present | Disc hemorrhage: absent
  Bayoneting: present | Beta-zone atrophy: present | PPA: present
```

---

## Part 1 — Two currently usable inference methods

### Inference Method A — rim = ISNT order   → **65.8%**
Indicators fed: CDR + **rim as ISNT order only** + 6 signs.

| Class | Recall | Misclassified as |
|-------|:------:|------------------|
| not likely | 0.95 | borderline 1, likely 1 |
| borderline | 0.40 | not likely 17, likely 7 |
| likely | 0.625 | borderline 13, not likely 2 |

### Inference Method B — rim = ISNT order + per-quadrant status   → **68.3%**
Indicators fed: CDR + **rim as ISNT order + per-quadrant thinning status** + 6 signs.

| Class | Recall | Misclassified as |
|-------|:------:|------------------|
| not likely | 0.725 | borderline 11 |
| borderline | 0.575 | not likely 8, likely 9 |
| likely | 0.75 | borderline 9, not likely 1 |

→ Method B is more balanced: borderline **+17.5 pts**, likely **+12.5 pts** (small drop on not-likely).

---

## Part 2 — Rim scoring methods tried

| Method | Used in Part 1? | Note |
|--------|:---------------:|------|
| 1. Absolute values | ❌ No | Weak signal (corr w/ diagnosis ≈ 0.05). Same 0.30 = normal in Temporal but abnormal in Superior → raw number meaningless without per-quadrant threshold. Dropped. |
| 2. ISNT order | ✅ Method A | Relative order of the 4 rims, e.g. `N>I>S>T`. Moderate signal (corr ≈ 0.21). |
| 3. Order + per-quadrant status | ✅ Method B | Order plus each quadrant labeled normal / mild / severe thinning. Strongest signal (corr ≈ 0.87). |

### Examples (one per class), showing the 3 methods

**🟢 Not likely** (id 116)
- Absolute: `I:0.367  S:0.375  N:0.267  T:0.213`
- Order: `S>I>N>T`
- Order+status: `S>I>N>T` + `Inferior: normal; Superior: normal; Nasal: normal; Temporal: normal`

**🟡 Borderline** (id 635)
- Absolute: `I:0.261  S:0.246  N:0.346  T:0.195`
- Order: `N>I>S>T`
- Order+status: `N>I>S>T` + `Inferior: borderline; Superior: mild thinning; Nasal: normal; Temporal: normal`

**🔴 Likely** (id 180)
- Absolute: `I:0.156  S:0.185  N:0.160  T:0.093`
- Order: `S>N>I>T`
- Order+status: `S>N>I>T` + `Inferior: thinning; Superior: thinning; Nasal: thinning; Temporal: severe thinning`

→ Absolute numbers look similar/ambiguous, order partially helps, but **order + per-quadrant status separates the 3 classes most clearly** — which is why Method B wins.
