@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ========= USER SETTINGS (edit these) =========
REM Base grid to deform (your existing refined grid)
set GRID_IN=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses\dark_band_mirror_map_fine\grid_ax_ay_z.csv

REM Locked particles list (to enforce separation)
set KNOWN_LOCKS=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses\all_particles_locked.csv

REM Scripts (adjust if you keep them elsewhere)
set S_EPOCH=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\epoch_deform_grid.py
set S_PEAKS=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\find_maxima_with_boundary.py
set S_GAP  =C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\compute_plateau_separation.py

REM Output base folder
set OUTBASE=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses\epoch_dark_sweep

REM Candidate selection knobs
set MIN_SEP=0.006
set TOP_K=20

REM Tau values to test (space-separated)
set TAUS=0.25 0.50 0.75
REM ==============================================


REM --- sanity checks ---
if not exist "%GRID_IN%" (
  echo [ERROR] GRID_IN not found: %GRID_IN%
  exit /b 1
)
if not exist "%KNOWN_LOCKS%" (
  echo [ERROR] KNOWN_LOCKS not found: %KNOWN_LOCKS%
  exit /b 1
)
for %%P in ("%S_EPOCH%" "%S_PEAKS%" "%S_GAP%") do (
  if not exist %%~fP (
    echo [ERROR] Script missing: %%~fP
    exit /b 1
  )
)

REM --- create out dirs ---
if not exist "%OUTBASE%" mkdir "%OUTBASE%"
set LOG=%OUTBASE%\run_log.txt
echo [%date% %time%] epoch_dark_sweep start > "%LOG%"

REM === loop over tau values ===
for %%T in (%TAUS%) do (
  set "TAU=%%T"

  REM Map tau -> deformation params (you can tweak the coefficients):
  REM scale = 1 + 0.4*tau ; tilt = 0.02*tau ; offset = 0
  for /f %%A in ('python -c "print(1+0.4*float(%TAU%))"') do set "SCALE=%%A"
  for /f %%A in ('python -c "print(0.02*float(%TAU%))"') do set "TILT=%%A"
  set "OFFSET=0.0"

  REM make a tau label without dot (e.g. 0.25 -> 025)
  set "TAULABEL=%%T"
  set "TAULABEL=!TAULABEL:.=!"

  set "GRID_TAU=%OUTBASE%\grid_tau!TAULABEL!.csv"
  set "CAND_DIR=%OUTBASE%\candidates_tau!TAULABEL!"
  set "GAP_DIR=%OUTBASE%\gapcheck_tau!TAULABEL!"

  echo.
  echo --- TAU=%%T  SCALE=!SCALE!  TILT=!TILT!  OFFSET=!OFFSET! ---
  echo [EPOCH] writing: !GRID_TAU!
  python "%S_EPOCH%" --in "%GRID_IN%" --out "!GRID_TAU!" --tau %%T --scale !SCALE! --tilt !TILT! --offset !OFFSET! 1>>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [WARN] epoch_deform failed for tau=%%T (see log). Skipping…
    goto :NEXTTAU
  )

  echo [PEAKS] outdir: !CAND_DIR!
  if not exist "!CAND_DIR!" mkdir "!CAND_DIR!"
  python "%S_PEAKS%" --grid "!GRID_TAU!" --known "%KNOWN_LOCKS%" --min_sep %MIN_SEP% --top_k %TOP_K% --outdir "!CAND_DIR!" 1>>"%LOG%" 2>&1
  if errorlevel 1 (
    echo [WARN] peaks finder failed for tau=%%T (see log). Continuing…
  )

  REM Optional gap/isolation check if peaks file exists
  set "PEAKSCSV=!CAND_DIR!\peaks_or_boundary.csv"
  if exist "!PEAKSCSV!" (
    echo [GAP] checking isolation: !GAP_DIR!
    if not exist "!GAP_DIR!" mkdir "!GAP_DIR!"
    python "%S_GAP%" --csv "!PEAKSCSV!" --outdir "!GAP_DIR!" 1>>"%LOG%" 2>&1
  ) else (
    echo [INFO] No peaks_or_boundary.csv found for tau=%%T
  )

  :NEXTTAU
)

echo.
echo [DONE] Results in: %OUTBASE%
echo       - Deformed grids: grid_tau*.csv
echo       - Candidates:     candidates_tau*\*
echo       - Gap checks:     gapcheck_tau*\*
echo [LOG ] Full log: %LOG%
endlocal