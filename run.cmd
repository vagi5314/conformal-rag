@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found.
    echo Create it first:  python -m venv .venv  ^&^&  .venv\Scripts\python.exe -m pip install -e ".[dev]"
    exit /b 1
)

rem Demo configuration: fast NLI model, demo tolerance, persisted state.
set PYTHONPATH=src
set CONFORMAL_RAG_NLI_MODEL=small
set CONFORMAL_RAG_ALPHA=0.15
set CONFORMAL_RAG_STATE_DIR=.state

if not exist ".state\engine_state.json" (
    echo First run: seeding the demo corpus and calibration ^(about a minute^) ...
    ".venv\Scripts\python.exe" scripts\seed_demo.py
    if errorlevel 1 (
        echo Seeding failed. Fix the error above, then run this file again.
        exit /b 1
    )
)

echo.
echo   Conformal RAG demo is starting at http://127.0.0.1:8000
echo   The browser opens on its own once the models are warm.
echo   Press Ctrl+C in this window to stop.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 12; Start-Process 'http://127.0.0.1:8000'"

".venv\Scripts\python.exe" -m uvicorn conformal_rag.api:create_app --factory --host 127.0.0.1 --port 8000

endlocal
