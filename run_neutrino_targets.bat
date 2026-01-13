@echo off
setlocal enabledelayedexpansion

REM ======= CONFIG (edit) =======
REM Command template to run ONE sweep. Replace with your real sweep command if needed.
set "SIMTEMPLATE=python \"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\ew_sweep.py\" --ax {ax} --ay {ay} --outdir \"{outdir}\""

set "LATENT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\geom_fit_many\latent_z_merged2.csv"
set "OUTMERGED=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\Predicted Masses\neutrino_three_locks_local_grids.csv"

set "PY=C:\Users\anthoni.newland\AppData\Local\Programs\Python\Python313\python.exe"
set "EXTRACT=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\z_from_runs.py"
REM ==============================

REM Header for merged CSV
echo source_path,ax,ay,z,r> "%OUTMERGED%"

REM Helper: format x.yyyy -> xpyyyy (4 dp)
for /f "delims=" %%A in ('powershell -NoProfile -Command "function fmt([double]$x){('{0:F4}' -f $x).Replace('.', 'p')}; fmt 2.0"') do set TESTFMT=%%A

REM Define targets (ax;ay). One per line.
set "TARGETS="
REM ν1 block
set "TARGETS=%TARGETS% 2.5795;0.7405"
set "TARGETS=%TARGETS% 2.5715;0.7405"
set "TARGETS=%TARGETS% 2.5755;0.7405"
set "TARGETS=%TARGETS% 2.5835;0.7405"
set "TARGETS=%TARGETS% 2.5875;0.7405"
set "TARGETS=%TARGETS% 2.5795;0.7325"
set "TARGETS=%TARGETS% 2.5795;0.7365"
set "TARGETS=%TARGETS% 2.5795;0.7445"
set "TARGETS=%TARGETS% 2.5795;0.7485"
REM ν2 block
set "TARGETS=%TARGETS% 2.55875;0.73375"
set "TARGETS=%TARGETS% 2.55075;0.73375"
set "TARGETS=%TARGETS% 2.55475;0.73375"
set "TARGETS=%TARGETS% 2.56275;0.73375"
set "TARGETS=%TARGETS% 2.56675;0.73375"
set "TARGETS=%TARGETS% 2.55875;0.72575"
set "TARGETS=%TARGETS% 2.55875;0.72975"
set "TARGETS=%TARGETS% 2.55875;0.73775"
set "TARGETS=%TARGETS% 2.55875;0.74175"
REM ν3 block
set "TARGETS=%TARGETS% 2.57025;0.7375"
set "TARGETS=%TARGETS% 2.56225;0.7375"
set "TARGETS=%TARGETS% 2.56625;0.7375"
set "TARGETS=%TARGETS% 2.57425;0.7375"
set "TARGETS=%TARGETS% 2.57825;0.7375"
set "TARGETS=%TARGETS% 2.57025;0.7295"
set "TARGETS=%TARGETS% 2.57025;0.7335"
set "TARGETS=%TARGETS% 2.57025;0.7415"
set "TARGETS=%TARGETS% 2.57025;0.7455"

for %%T in (%TARGETS%) do (
  for /f "tokens=1,2 delims=;" %%a in ("%%~T") do (
    set ax=%%a
    set ay=%%b

    for /f "delims=" %%F in ('powershell -NoProfile -Command "('{0:F4}' -f [double]!ax!).Replace('.', 'p')"') do set axname=%%F
    for /f "delims=" %%G in ('powershell -NoProfile -Command "('{0:F4}' -f [double]!ay!).Replace('.', 'p')"') do set ayname=%%G

    set "outdir=C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ew_sweep_ax!axname!_ay!ayname!"
    if not exist "!outdir!" mkdir "!outdir!"

    set "SIMCMD=%SIMTEMPLATE%"
    set "SIMCMD=!SIMCMD:{ax}=!ax!!"
    set "SIMCMD=!SIMCMD:{ay}=!ay!!"
    set "SIMCMD=!SIMCMD:{outdir}=!outdir!!"

    echo >>> Running sweep: !SIMCMD!
    call !SIMCMD!
    if errorlevel 1 (
      echo WARNING: sweep failed for ax=!ax! ay=!ay!
      goto :continue
    )

    set "inCsv=!outdir!\cycle_rows_r.csv"
    if not exist "!inCsv!" (
      echo WARNING: missing !inCsv!
      goto :continue
    )

    set "tmpOut=!outdir!\z_extracted.csv"
    "%PY%" "%EXTRACT%" --latent "%LATENT%" --in_csv "!inCsv!" --out_csv "!tmpOut!"
    if errorlevel 1 (
      echo WARNING: z_from_runs failed for ax=!ax! ay=!ay!
      goto :continue
    )

    for /f "skip=1 delims=" %%L in ('type "!tmpOut!"') do (
      echo %%L>> "%OUTMERGED%"
    )

    :continue
  )
)

echo.
echo All done.
echo Merged results: %OUTMERGED%