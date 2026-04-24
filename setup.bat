@echo off
REM Lab 1 CV - Setup Script for Windows
REM This script sets up the Python environment for the Street Style Classification project

echo ============================================
echo Lab 1 CV - Environment Setup (Windows)
echo ============================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [2/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [2/4] Virtual environment already exists, skipping creation...
)

REM Activate virtual environment
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [4/4] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ============================================
echo Setup complete!
echo ============================================
echo.
echo To activate the environment, run:
echo   call venv\Scripts\activate.bat
echo.
echo To start training, run:
echo   python src\train.py --model resnet50 --epochs 20
echo ============================================
