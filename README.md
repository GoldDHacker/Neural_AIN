# Adaptive Invariant Neuron (AIN)

AIN is a “zero pre-wired expert” architecture designed to **discover invariants** through an iterative **System-2 trial-and-error** loop.

Core ideas implemented in this repository:

- **Ontological pathways (9 voies)**: the encoder exposes 9 distinct pathways (e.g., continuous/spline, discrete/spin, relational/attention, variational/statistical, temporal/Chronos, geometric, curvature/Riemann, combinatorial, algorithmic/automate).
- **Combinatorial routing**: instead of selecting a single expert, the router selects **coalitions** (binary recipes over pathways).
- **System-2 deliberation**: routing is refined over multiple internal iterations by testing candidate invariants/programs on the support and feeding back an error signal.
- **Multi-oracle self-consistency**: a legacy heuristic oracle (`support_std`) is combined with an auto-supervised **masked-support prediction** oracle, mixed by a learned gate and aggregated across System-2 iterations via softmin.
- **ProgramBank (external memory)**: stores a context signature, the discovered invariant `z`, the compiled program (`forged`), and optional replay episodes. It enables reuse and helps reduce catastrophic forgetting in continual/curriculum settings.

## Repository layout (high level)

- **`ain_neuron.py`**: main AIN model implementation (pathways, combinatorial router, System-2 loop, multi-oracle, routing stability mechanisms).
- **`program_bank.py`**: ProgramBank memory module and related policies/orchestrator utilities.
- **`arxiv/`**: LaTeX article source (`main.tex`, `refs.bib`).
- **Markdown reports / notes**:
  - `GENESE_CONCEPTION_AIN.md`
  - `RESULTATS_TESTS_AIN_SET_SEQUENCE.md`
  - `RUBIKS_2x2_EXPERIENCES.md`
  - `RESULTATS_PROGRAM_BANK_CONTINUAL_AIN.md`
  - `RAPPORT_DIAGNOSTIC_CHRONOS_ROUTING.md`

## Demos and experiments

This repo includes Python demo scripts used to generate the reported results (names as present in the project):

- `demo_ain.py` (general diagnostic battery)
- `demo_ain_set.py` (SET / permutation-invariant regime)
- `demo_ain_sequence.py` (SEQUENCE / order-dependent regime)
- `demo_ain_program_bank_continual.py` (continual learning with ProgramBank + replay)

See the corresponding Markdown files and saved logs for protocols, metrics, and consolidated results.

## ProgramBank usage notes

- ProgramBank entries store:
  - a **context signature** computed from expanded features `[x, x^2]` using mean/std over nodes and mean/std over consecutive deltas,
  - the invariant **`z`**,
  - the compiled program **`forged`**,
  - optional replay mini-episodes (`support`, `query`, `target`).
- Matching is performed primarily by cosine similarity on `z` (threshold `z_threshold`), with an optional signature-based fallback (`signature_threshold`).

## Building the arXiv PDF

The paper sources are in `arxiv/`.

Typical LaTeX build (from inside `arxiv/`):

- `pdflatex main.tex`
- (optional if you change citations) `bibtex main` then `pdflatex main.tex` twice

## License

No license file is provided in this repository snapshot. If you plan to distribute the code publicly, add a license.
