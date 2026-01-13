@echo off
set PYTHONIOENCODING=utf-8

set PY=C:\Users\anthoni.newland\AppData\Local\Programs\Python\Python313\python.exe
set SCRIPTS=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts
set LATENT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\geom_fit_many\latent_z_merged2.csv
set LOCKED=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses\bosons_locked.csv
set OUTDIR=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\mini_grids_bosons

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

"%PY%" "%SCRIPTS%\mini_grid_sweeps.py" ^
  --latent "%LATENT%" ^
  --locked "%LOCKED%" ^
  --step 0.004 ^
  --radius 2 ^
  --outdir "%OUTDIR%"

echo.
echo [DONE] Mini-grids saved to:
echo   %OUTDIR%
pause