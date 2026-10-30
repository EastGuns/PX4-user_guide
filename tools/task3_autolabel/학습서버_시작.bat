@echo off
rem Task3 click labeler - local training server (double-click to start).
rem Opens http://127.0.0.1:8765 in the browser; the page's panel 6 then trains YOLO11n on this PC's GPU.
rem Keep this window open while training. Closing it stops the server and any running training.
setlocal
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

rem Use Anaconda/Miniconda base if installed (that is where torch+CUDA and ultralytics live on the lab PC).
rem To use another Python, set T3_PYTHON to its python.exe before running this file.
if defined T3_PYTHON goto run
set "CONDA_ROOT="
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "C:\anaconda3" "C:\miniconda3") do (
  if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
)
if defined CONDA_ROOT call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%"
set "T3_PYTHON=python"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Anaconda, or Python 3.9+ with: pip install ultralytics
  pause
  exit /b 1
)

:run
"%T3_PYTHON%" "%~dp0trainer\task3_train_server.py" %*
echo.
echo Server stopped.
pause
