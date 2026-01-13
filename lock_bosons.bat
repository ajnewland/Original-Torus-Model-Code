@echo off
REM Avoid Unicode console errors from Python prints
set PYTHONIOENCODING=utf-8

set PY=C:\Users\anthoni.newland\AppData\Local\Programs\Python\Python313\python.exe
set SCRIPTS=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts
set LATENT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\geom_fit_many\latent_z_merged2.csv
set ISO=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\anchor_free_iso_full_many\anchor_free_isotonic_masses.csv
set OUTDIR=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses
set OUT=%OUTDIR%\bosons_locked.csv

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

"%PY%" "%SCRIPTS%\predict_ax_ay_for_mass.py" ^
  --latent "%LATENT%" ^
  --iso "%ISO%" ^
  --sin2 0.231 --alpha 0.0647 --beta 0.5529 ^
  --masses "H:125.1,W:80.4,Z:91.2" ^
  --out "%OUT%"

echo.
echo [DONE] Wrote boson locks to:
echo   %OUT%
pause