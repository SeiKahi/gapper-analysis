@echo off
REM Setup-Script: Kopiert alle ML-Dateien an die richtigen Stellen
REM Ausfuehren: cd gapper-analysis && setup_ml.bat

echo Erstelle Ordner...
if not exist "docs" mkdir docs
if not exist "scripts" mkdir scripts
if not exist "results" mkdir results
if not exist "models" mkdir models

echo.
echo Pruefe XGBoost Installation...
.\gapper_env\Scripts\python.exe -c "import xgboost; print('XGBoost OK:', xgboost.__version__)" 2>nul
if errorlevel 1 (
    echo XGBoost nicht gefunden. Installiere...
    .\gapper_env\Scripts\pip.exe install xgboost scikit-learn
)

echo.
echo Setup fertig! Starte Pipeline mit:
echo   .\gapper_env\Scripts\python.exe scripts\ml_pipeline.py
echo.
echo Oder fuer Token-effiziente Agent-Nutzung:
echo   1. CLAUDE.md lesen lassen
echo   2. Agent liest docs\ bei Bedarf
echo   3. Ergebnisse landen in results\ und models\
pause
