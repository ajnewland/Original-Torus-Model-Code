REM === scan_sin2_fit.bat ===
@echo off
setlocal

set PY=C:\Users\anthoni.newland\AppData\Local\Programs\Python\Python313\python.exe
set PREDICT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\predict_ax_ay_for_mass.py
set LATENT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\geom_fit_many\latent_z_merged2.csv
set ISO=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\anchor_free_iso_full_many\anchor_free_isotonic_masses.csv

REM Your current reference couplings (alpha,beta) – keep as you used before:
set ALPHA=0.0647
set BETA=0.5529

REM Target masses (GeV) – full SM set you’ve been using:
set MASSES=u:0.0022,d:0.0047,s:0.095,e:0.000511,mu:0.10566,c:1.27,tau:1.77686,b:4.18,t:172.76,H:125.1,W:80.4,Z:91.2

REM Output folder:
set OUTBASE=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\sin2_scan
if not exist "%OUTBASE%" mkdir "%OUTBASE%"

echo sin2,total_abs_err > "%OUTBASE%\summary.csv"

for %%S in (0.218 0.220 0.222 0.224 0.226 0.228 0.229 0.230 0.231 0.232 0.233 0.235) do (
  set OUTCSV=%OUTBASE%\fit_sin2_%%S.csv
  "%PY%" "%PREDICT%" --latent "%LATENT%" --iso "%ISO%" --sin2 %%S --alpha %ALPHA% --beta %BETA% --masses %MASSES% --out "%OUTCSV%" > "%OUTBASE%\log_%%S.txt" 2>&1

  REM accumulate total absolute z-error across rows (simple and robust)
  "%PY%" - <<PYCODE
import pandas as pd, sys
p = r"%OUTCSV%"
try:
    df = pd.read_csv(p)
    tot = float(df["abs_err"].abs().sum())
except Exception as e:
    tot = float("nan")
print(f"{p.split('_')[-1][:-4]},{tot}")
PYCODE
)>> "%OUTBASE%\summary.csv"
)

echo Done. See %OUTBASE%\summary.csv
endlocal