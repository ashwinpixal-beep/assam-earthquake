@echo off
REM Double-click launcher for Windows: starts the site and opens the browser.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo Python 3 was not found. Install it from https://python.org and tick "Add Python to PATH".
    pause
    exit /b 1
  )
)

echo Starting the damage-risk site...
%PY% start.py
if errorlevel 1 (
  echo.
  echo The site stopped with an error. The message above explains why.
  pause
)
endlocal
