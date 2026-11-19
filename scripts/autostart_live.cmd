@echo off
REM PocketDiffusion live server - started automatically at Windows logon (see install_autostart.cmd).
REM  * waits until the internet is reachable (Wi-Fi / hotspot may connect after logon)
REM  * starts the full-stack app with the public link (serve.py --share --publish-link), which keeps
REM    https://arpreran.github.io/pocketdiffusion/ pointing at it
REM  * restarts the server if it ever stops; close this window to stop everything
title PocketDiffusion live
setlocal
set "ROOT=%~dp0.."
set "PY=%USERPROFILE%\.venvs\pocketdiff\Scripts\python.exe"
if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
cd /d "%ROOT%"

REM already running (e.g. started by hand)? then do nothing
netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul && (
    echo PocketDiffusion is already running on port 8000 - nothing to do.
    timeout /t 10 >nul
    exit /b 0
)

:waitnet
curl.exe -s -m 10 -o NUL https://api.gradio.app/v3/tunnel-request
if errorlevel 1 (
    echo %date% %time%  waiting for internet ...
    timeout /t 15 /nobreak >nul
    goto waitnet
)

:run
echo %date% %time%  starting PocketDiffusion (log: data\server.log)
echo ===== %date% %time% autostart ===== >> "data\server.log"
"%PY%" serve.py --share --publish-link >> "data\server.log" 2>&1
echo %date% %time%  server stopped (exit %errorlevel%) - restarting in 30 s, close this window to stop
timeout /t 30 /nobreak >nul
goto waitnet
