@echo off
rem Run the full pipeline. Pass --sample to use synthetic test data.
cd /d "%~dp0scripts"
set PY=..\.venv\Scripts\python.exe
%PY% 01_ingest_clean.py %1 || exit /b 1
%PY% 02_enrich_features.py || exit /b 1
%PY% 03_eda.py || exit /b 1
%PY% 04_model.py || exit /b 1
%PY% 05_dashboard.py || exit /b 1
echo.
echo Done. See the outputs folder.
