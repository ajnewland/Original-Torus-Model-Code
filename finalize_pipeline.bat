@echo off
REM ============================
REM  Z3 Torus — finalize pipeline
REM  One-click: param search -> build slices -> constraints report -> CKM/PMNS
REM  UTF-8 console
REM ============================
chcp 65001 >nul

REM ---------- EDIT THESE PATHS ----------
set PY=python
set SCRIPTS="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts"
set DATA_IN="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results"
set OUTROOT="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\FINAL_RUN"

REM Inputs you already have:
set SPECIES_T0=%DATA_IN%\torsion_asymmetry_by_species.csv
set LOCKED_CSV=%DATA_IN%\Predicted Masses\all_particles_locked.csv

REM Where the current t0 rich slice lives (from your earlier build):
set RICH_T0_H5=%DATA_IN%\rich\torus_rich_t0_fields.h5

REM ---------- NUMERICS (you can tweak) ----------
REM Parameter search grids (coarse→fine as needed)
set ALPHAS=3.5,4.0,4.25,4.5
set DTS=0.08,0.1,0.12
set MUS=0.002,0.003,0.004

REM Refinements that help the Hamiltonian residual
set LAPLACE=nine
set SMOOTH_PX=1.2
set MASK_MARGIN=6

REM CKM / PMNS sweep knobs
set CKM_SIGMA=0.020
set CKM_GRID=240
set PMNS_SIGMA=0.024

REM Make output folders
if not exist %OUTROOT% mkdir %OUTROOT%
set SEARCH_OUT=%OUTROOT%\param_search_refined
set BUILD_OUT_PREFIX=%OUTROOT%\rich\torus_rich
set REPORT_OUT=%OUTROOT%\einstein_momentum_report
set CKM_OUT=%OUTROOT%\ckm_out
set PMNS_OUT=%OUTROOT%\pmns_out
for %%D in ("%OUTROOT%\rich" "%SEARCH_OUT%" "%REPORT_OUT%" "%CKM_OUT%" "%PMNS_OUT%") do if not exist "%%~D" mkdir "%%~D"

echo.
echo === 1) Parameter search (alpha, dt, mu) with refined numerics ===
%PY% %SCRIPTS%\torus_parameter_search.py ^
  --h5_t0 %RICH_T0_H5% ^
  --alpha_list %ALPHAS% ^
  --dt_list %DTS% ^
  --mu_list %MUS% ^
  --laplace %LAPLACE% --smooth_px %SMOOTH_PX% --mask_margin %MASK_MARGIN% ^
  --outdir %SEARCH_OUT%
if errorlevel 1 goto :fail

echo.
echo === 2) Extract best (alpha, dt, mu) from search_results.csv ===
REM We’ll have PowerShell parse the CSV top row and emit a temporary .bat that sets env vars.
powershell -NoProfile -Command ^
  "$p = Import-Csv '%SEARCH_OUT%\search_results.csv' | Sort-Object {[double]$_.score} | Select-Object -First 1; " ^
  "Set-Content -Encoding Ascii '%SEARCH_OUT%\best_vars.bat' (@(" ^
  "'set BEST_ALPHA=' + $p.alpha," ^
  "'set BEST_DT='    + $p.dt," ^
  "'set BEST_MU='    + $p.mu" ^
  "))"
if errorlevel 1 goto :fail

call "%SEARCH_OUT%\best_vars.bat"
echo Best alpha=%BEST_ALPHA%  dt=%BEST_DT%  mu=%BEST_MU%

echo.
echo === 3) Build rich slices at the best (alpha, dt, mu) ===
%PY% %SCRIPTS%\build_rich_torus_timeslices.py ^
  --species_csv_t0 %SPECIES_T0% ^
  --locked_csv     %LOCKED_CSV% ^
  --grid_n 400 --alpha %BEST_ALPHA% --dt %BEST_DT% ^
  --evolve_mode gradflow --mu %BEST_MU% ^
  --outprefix %BUILD_OUT_PREFIX%
if errorlevel 1 goto :fail

set H5_T0=%BUILD_OUT_PREFIX%_t0_fields.h5
set H5_T1=%BUILD_OUT_PREFIX%_t1_fields.h5

echo.
echo === 4) Einstein + Momentum report (alpha sweep around best) ===
REM Sweep around the best alpha by ±0.25 (adjust if you like)
setlocal EnableDelayedExpansion
for /f %%A in ('powershell -NoProfile -Command "[double]$env:BEST_ALPHA - 0.25"') do set A_MINUS=%%A
for /f %%A in ('powershell -NoProfile -Command "$env:BEST_ALPHA"') do set A_MID=%%A
for /f %%A in ('powershell -NoProfile -Command "[double]$env:BEST_ALPHA + 0.25"') do set A_PLUS=%%A
endlocal & set ALPHA_SWEEP=%A_MINUS%,%A_MID%,%A_PLUS%

%PY% %SCRIPTS%\torus_einstein_momentum_report.py ^
  --h5_t0 %H5_T0% ^
  --h5_t1 %H5_T1% ^
  --dt %BEST_DT% --lapse 1.0 ^
  --alpha_sweep %ALPHA_SWEEP% ^
  --laplace %LAPLACE% --smooth_px %SMOOTH_PX% --mask_margin %MASK_MARGIN% ^
  --outdir %REPORT_OUT%
if errorlevel 1 goto :fail

echo.
echo === 5) CKM with the same geometry (torsion+locked CSV) ===
%PY% %SCRIPTS%\predict_ckm_from_torsion_v2.py ^
  --torsion_csv %SPECIES_T0% ^
  --locked_csv  %LOCKED_CSV% ^
  --sigma %CKM_SIGMA% --grid_n %CKM_GRID% ^
  --outdir %CKM_OUT%
if errorlevel 1 goto :fail

echo.
echo === 6) (Optional) PMNS — if you have the script ready ===
REM Uncomment the next block if you want to run PMNS too.
REM %PY% %SCRIPTS%\predict_pmns_from_torsion.py ^
REM   --torsion_csv %SPECIES_T0% ^
REM   --locked_csv  %LOCKED_CSV% ^
REM   --sigma %PMNS_SIGMA% ^
REM   --outdir %PMNS_OUT%
REM if errorlevel 1 goto :fail

echo.
echo ==========================================================
echo  DONE.
echo  Results folders:
echo    %SEARCH_OUT%
echo    %OUTROOT%\rich
echo    %REPORT_OUT%
echo    %CKM_OUT%
echo    %PMNS_OUT%  (if run)
echo.
echo  Key files to share with a reviewer:
echo    - %REPORT_OUT%\report.pdf
echo    - %REPORT_OUT%\summary.csv
echo    - %SEARCH_OUT%\search_results.csv
echo    - %CKM_OUT%\ckm_comparison_v2.csv  (and predicted_ckm_v2.csv)
echo    - %OUTROOT%\rich\torus_rich_t0_fields.h5  (and _t1_fields.h5)
echo ==========================================================
exit /b 0

:fail
echo.
echo !!! PIPELINE FAILED — see the console for the step that errored.
exit /b 1