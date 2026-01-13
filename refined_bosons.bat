@echo off
set PYTHONIOENCODING=utf-8

set PY=C:\Users\anthoni.newland\AppData\Local\Programs\Python\Python313\python.exe
set SCRIPTS=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts
set LATENT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\geom_fit_many\latent_z_merged2.csv
set OUTBASE=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses

REM === Higgs (center 2.5671, 0.9014375) target z = 1.6830744610458144 ===
"%PY%" "%SCRIPTS%\refined_neutrino_sweep.py" ^
  --latent "%LATENT%" ^
  --ax_min 2.5471 --ax_max 2.5871 --ax_steps 161 ^
  --ay_min 0.8814375 --ay_max 0.9214375 --ay_steps 161 ^
  --targets 1.6830744610458144 ^
  --min_sep 0.001 ^
  --outdir "%OUTBASE%\boson_refined_H"

REM === W (center 2.4840, 0.8775) target z = 1.5538675472927237 ===
"%PY%" "%SCRIPTS%\refined_neutrino_sweep.py" ^
  --latent "%LATENT%" ^
  --ax_min 2.4640 --ax_max 2.5040 --ax_steps 161 ^
  --ay_min 0.8575  --ay_max 0.8975  --ay_steps 161 ^
  --targets 1.5538675472927237 ^
  --min_sep 0.001 ^
  --outdir "%OUTBASE%\boson_refined_W"

REM === Z (center 2.5668, 0.89565625) target z = 1.595805142840617 ===
"%PY%" "%SCRIPTS%\refined_neutrino_sweep.py" ^
  --latent "%LATENT%" ^
  --ax_min 2.5468 --ax_max 2.5868 --ax_steps 161 ^
  --ay_min 0.87565625 --ay_max 0.91565625 --ay_steps 161 ^
  --targets 1.595805142840617 ^
  --min_sep 0.001 ^
  --outdir "%OUTBASE%\boson_refined_Z"

echo.
echo [DONE] Refined boson sweeps finished. Check the heatmap_contours.png files in:
echo   %OUTBASE%\boson_refined_H
echo   %OUTBASE%\boson_refined_W
echo   %OUTBASE%\boson_refined_Z
pause