@echo off
REM One-click start on Windows (full-stack app: web frontend + FastAPI backend + SQLite).
REM   run.bat            -> http://127.0.0.1:8000
REM   run.bat --share    -> also prints a public https link for phones / judges
REM   run.bat --share --publish-link -> + keeps https://arpreran.github.io/pocketdiffusion/ pointing at it
REM   run.bat --gradio   -> the alternative Gradio interface (python app.py)
setlocal
if exist "%~dp0.venv\Scripts\python.exe" (
    set PY="%~dp0.venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\.venvs\pocketdiff\Scripts\python.exe" (
    set PY="%USERPROFILE%\.venvs\pocketdiff\Scripts\python.exe"
) else (
    set PY=python
)
cd /d "%~dp0"
if "%1"=="--gradio" (
    %PY% app.py %2 %3 %4
) else (
    %PY% serve.py %*
)
