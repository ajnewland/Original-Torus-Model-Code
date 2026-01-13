@echo off
setlocal EnableExtensions

REM --- 1) Point to your files (quoted; paths have spaces) ---
set "EW=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ew_band.csv"
set "STR=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\strong_band.csv"

REM --- 2) Where to write the report ---
for %%I in ("%EW%") do set "OUTDIR=%%~dpI"
set "REPORT=%OUTDIR%strong_probe_report.txt"

REM --- 3) Make a temporary Python script ---
set "TMPPY=%TEMP%\strong_probe_%RANDOM%%RANDOM%.py"

REM Build the Python file using PowerShell to avoid echo/escaping pain
powershell -NoProfile -Command ^
  "$code = @'
import csv, json, sys, os, math
from statistics import mean

def load_r_z(path):
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        cols = [c.strip() for c in r.fieldnames]
        need = {'r','z'}
        if not need.issubset(set(cols)):
            raise SystemExit(f\"{os.path.basename(path)} must have columns: {sorted(need)} (found: {cols})\")
        R, Z = [], []
        for row in r:
            try:
                R.append(float(row['r']))
                Z.append(float(row['z']))
            except Exception:
                pass  # skip bad rows
        if not R:
            raise SystemExit(f\"No numeric rows found in {path}\")
        return R, Z

ew_path, str_path, report_path = sys.argv[1], sys.argv[2], sys.argv[3]
R_ew, Z_ew = load_r_z(ew_path)
R_str, Z_str = load_r_z(str_path)

# Basic stats
n_ew, n_str = len(R_ew), len(R_str)
mu_ew, mu_str = mean(R_ew), mean(R_str)
mu_z_ew, mu_z_str = mean(Z_ew), mean(Z_str)

# Pair the first min(n) samples for a simple cross-sector fit
n = min(n_ew, n_str)
x = R_str[:n]
y = R_ew[:n]

# Linear least-squares fit: y ≈ A + B*x
mx = mean(x)
my = mean(y)
num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
den = sum((xi-mx)**2 for xi in x)
B = (num/den) if den != 0 else float('nan')
A = my - B*mx

# Residuals / MSE
res = [(yi - (A + B*xi)) for xi,yi in zip(x,y)]
mse = sum(r*r for r in res)/n

# Compact JSON result for programmatic use
result = {
    'counts': {'ew': n_ew, 'strong': n_str, 'paired': n},
    'r_means': {'ew': mu_ew, 'strong': mu_str, 'delta': mu_ew - mu_str},
    'z_means': {'ew': mu_z_ew, 'strong': mu_z_str, 'delta': mu_z_ew - mu_z_str},
    'affine_map_strong_to_ew': {'A': A, 'B': B, 'mse': mse}
}

# Pretty text report
lines = []
lines.append('=== Strong↔EW probe report ===')
lines.append(f'EW file     : {ew_path}')
lines.append(f'Strong file : {str_path}')
lines.append(f'Samples (EW / STR / paired): {n_ew} / {n_str} / {n}')
lines.append('')
lines.append('r-statistics')
lines.append(f'  mean(r)_EW     = {mu_ew:.9f}')
lines.append(f'  mean(r)_STR    = {mu_str:.9f}')
lines.append(f'  delta means    = {mu_ew - mu_str:.9f}')
lines.append('')
lines.append('z-statistics (sanity check)')
lines.append(f'  mean(z)_EW     = {mu_z_ew:.9f}')
lines.append(f'  mean(z)_STR    = {mu_z_str:.9f}')
lines.append(f'  delta means    = {mu_z_ew - mu_z_str:.9f}')
lines.append('')
lines.append('Affine map y_EW ≈ A + B * x_STR (paired by index)')
lines.append(f'  A = {A:.12g}')
lines.append(f'  B = {B:.12g}')
lines.append(f'  MSE = {mse:.12g}')
lines.append('')
lines.append('JSON:')
lines.append(json.dumps(result, indent=2))

text = '\n'.join(lines)
print(text)

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(text + '\n')

'@; Set-Content -Path $env:TMPPY -Value $code -Encoding UTF8"

if errorlevel 1 (
  echo Failed to create temp Python script.
  exit /b 1
)

REM --- 4) Run it with your files and save the report ---
python "%TMPPY%" "%EW%" "%STR%" "%REPORT%"
set ERR=%ERRORLEVEL%

echo(
if %ERR% EQU 0 (
  echo ---------------------------------------------------------
  echo Saved report to:
  echo   "%REPORT%"
  echo ---------------------------------------------------------
) else (
  echo There was an error (exit code %ERR%). See messages above.
)

REM --- 5) Cleanup temp script (optional) ---
del "%TMPPY%" >nul 2>nul

endlocal