
# Single-feature fits report

**Rows:** 10

## Results: λ₁ (principal Laplacian eigenvalue)
- **Linear:** `v² = -1.445e+04·λ₁ + 2.628e+05`, **RMSE = 187032.289**
- **Inverse:** `v² = 3.179e+05·(1/λ₁) + -4.219e+00`, **RMSE = 3.633**
- **Log–log:** `log v² = -1.000·log λ₁ + 12.670`, **RMSE = 12.275`  *(expect slope ≈ −1 if v² ∝ 1/λ₁)*

## Results: base_spec (spectral-shape proxy)
- **Linear:** `v² = 9.053e+04·base_spec + -1.541e+05`, **RMSE = 40156.406**

## Results: ρ_v (vertex count proxy — replace with true density when available)
- **Linear:** `v² = 4.083e+02·ρ_v + 5.880e+04`, **RMSE = 217869.905`

---

### Interpretation
- The **inverse** and **log–log** fits for **λ₁** are the key diagnostics for the geometric law `v² ∝ 1/λ₁`.
- **base_spec** can act as a *shape correction*; if helpful, try a two-term law:

  `v² ≈ C/λ₁ · (1 + α (base_spec − s̄))`.

- The plotted PNGs in this folder show the regressions and residual scales.

**Generated files**
- `fit_lambda1.png` — linear vs inverse vs log–log fits
- `fit_base_spec.png`
- `fit_rho_v.png`
- `report.md`
