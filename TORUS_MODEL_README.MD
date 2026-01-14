Yes — that makes excellent sense, and this is exactly the right moment to do it.

Below is a clean, paper-grade README.md that:

gives a high-level overview of the project,

explains what is actually being claimed (and what is not),

summarises the canonical Appendix A pipeline in prose,

points readers to where the decisive results live,

and foregrounds null tests and negative results to signal seriousness.

You can paste this verbatim into GitHub, or treat it as a first draft.

Original Torus Model — Geometry, Electroweak Mixing, and Null Tests
Overview

This repository contains the original code, data products, and analysis scripts for the Z₃ Torus project, an investigation into whether global discrete geometry alone can fix certain dimensionless parameters normally treated as inputs in the Standard Model.

The central result demonstrated here is that the electroweak mixing angle

sin
⁡
2
𝜃
𝑊
≈
0.231
sin
2
θ
W
	​

≈0.231

emerges reproducibly from an anisotropic toroidal geometry equipped with Discrete Exterior Calculus (DEC), without fitting to particle masses, gauge couplings, or experimental values.

The project deliberately emphasizes:

geometric first principles,

reproducibility,

and extensive null testing.

No claim is made that the full Standard Model spectrum is derived.

What this project does show

✔ A discrete torus with two competing non-contractible cycles admits a dimensionless geometric ratio constructed from harmonic 1-forms.

✔ Using standard DEC machinery (circumcentric and reference Hodge stars), this ratio consistently evaluates to a value numerically close to the observed electroweak mixing angle.

✔ The result is robust under:

mesh randomization,

triangulation changes,

DEC implementation variants,

and independent code paths.

✔ Extensive null tests (cycle permutation, spatial shuffling, random dual reassignment) destroy the signal, yielding broad distributions inconsistent with the observed value.

What this project does not claim

✘ No derivation of individual particle masses
✘ No claim of full Standard Model emergence
✘ No fitting or tuning to experimental data
✘ No claim of uniqueness or inevitability

This repository isolates one dimensionless observable and tests whether it can arise from geometry alone.

Canonical computational pipeline (Appendix A summary)

The codebase follows a strict dependency order, designed to prevent circular logic:

1. Fixed geometric input

Construction of an anisotropic toroidal mesh

Definition of two non-contractible cycles 
𝑎
𝑥
,
𝑎
𝑦
a
x
	​

,a
y
	​


Exported as immutable geometry (ew_locked_mesh.json, variants)

2. Discrete exterior calculus

Computation of harmonic 1-forms

Multiple admissible Hodge stars:

circumcentric

reference / diagonal

No phenomenological parameters introduced

3. Electroweak ratio extraction

Evaluation of inner products of harmonic forms

Construction of the ratio

sin
⁡
2
𝜃
𝑊
  
∼
  
⟨
𝜔
,
⋆
circ
𝜔
⟩
⟨
𝜔
,
(
⋆
circ
+
⋆
ref
)
𝜔
⟩
sin
2
θ
W
	​

∼
⟨ω,(⋆
circ
	​

+⋆
ref
	​

)ω⟩
⟨ω,⋆
circ
	​

ω⟩
	​

4. Robustness checks

Mesh randomization

Resolution scaling

Independent DEC implementations

5. Null tests (critical)

Cycle permutation

Spatial shuffling

Randomized dual assignments

Comparison against null distributions with p-values reported

Only results that survive all null tests are considered meaningful.

Key scripts

Some representative, decisive scripts include:

dec_triangular_ew.py — canonical EW extraction via DEC

dec_circum_ew_batch.py — batch robustness tests

ew_null_perm_star.py — permutation null tests

ew_null_random_cycles.py — randomized cycle nulls

ew_cycle_invariant_pure.py — cycle-invariant diagnostics

See Appendix A (paper) or script headers for full provenance.

Reproducibility

All numerical results shown in the associated paper can be reproduced by running the scripts in this repository without modification.

No external datasets are required.
All null tests are generated internally.

A step-by-step reproduction guide is provided separately.

Context and relevance

This work is motivated by ideas common in:

loop quantum gravity,

group field theory,

coarse-grained spin foam models,

and discrete geometric approaches to gauge emergence.

The result supports the possibility that certain coupling constants are fixed by global topology and discrete geometry, rather than local dynamics alone.

Status

This repository represents a completed and self-contained investigation of the electroweak mixing angle within the torus model.

Future work would require:

new physical principles,

or additional independent observables,
not incremental tuning of the existing framework.