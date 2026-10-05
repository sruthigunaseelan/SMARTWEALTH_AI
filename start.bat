@echo off
echo ===================================================
echo   SmartWealth AI — Starting Server on Localhost
echo ===================================================
echo Access Application at: http://localhost:8000
echo Access API Docs at:    http://localhost:8000/docs
echo.
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000
pause
