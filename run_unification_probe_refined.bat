@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Paths (edit only if your files moved) ===
set PYTHON=python
set SWEEP_IN="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ew_sweep_files_combined.csv"
set PREFILTER_OUT="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ew_band_refined.csv"
set LOCKS="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses\all_particles_locked.csv"
set OUTDIR="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\unification_probe_refined"

REM === Scripts (edit only if you renamed or moved them) ===
set PREFILTER_PY="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\prefilter_relax_band.py"
set PROBE_PY="C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\torus_unification_probe.py"

REM === Ensure output directory exists ===
if not exist %OUTDIR% mkdir %OUTDIR%

echo.
echo ===================== STEP 1: Prefilter (ay in [0.90, 1.00], S* in [0.220, 0.235]) =====================
%PYTHON% %PREFILTER_PY% ^
  %SWEEP_IN% ^
  %PREFILTER_OUT% ^
  0.90 1.00 0.220 0.235

if errorlevel 1 (
  echo [ERROR] Prefilter step failed. Aborting.
  exit /b 1
)

echo.
echo ===================== STEP 2: Probe (robust interior trimming) =====================
REM Primary run: percentile interior, 5%% erosion, min 200 interior points
%PYTHON% %PROBE_PY% ^
  --latent-csv  %PREFILTER_OUT% ^
  --locks       %LOCKS% ^
  --outdir      %OUTDIR% ^
  --ax-col ax --ay-col ay --z-col S_star --r-col r ^
  --interior-policy percentile ^
  --erosion 0.05 ^
  --min-interior 200

if errorlevel 1 (
  echo.
  echo [WARN] Primary probe failed (likely not enough interior points). Trying a more aggressive trim...
  REM Fallback: increase erosion to 0.10 and lower min-interior to 100
  %PYTHON% %PROBE_PY% ^
    --latent-csv  %PREFILTER_OUT% ^
    --locks       %LOCKS% ^
    --outdir      %OUTDIR% ^
    --ax-col ax --ay-col ay --z-col S_star --r-col r ^
    --interior-policy percentile ^
    --erosion 0.10 ^
    --min-interior 100

  if errorlevel 1 (
    echo [ERROR] Fallback probe also failed. Consider widening the prefilter windows.
    exit /b 1
  )
)

echo.
echo [DONE] Results written under: %OUTDIR%
endlocal