@echo off
echo ============================================================
echo   UMAPLAY UPDATE SCRIPT
echo ============================================================
echo.

cd /d "%~dp0"

:: Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Run the update script
python update.py %*

echo.
pause
