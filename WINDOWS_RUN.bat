@echo off
title AutoSub
cd /d "%~dp0"

REM ====================================================================
REM  Step 1: Check for uv
REM ====================================================================
echo.
echo === [1/2] Checking uv ===

where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv is not installed. Installing...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo uv installed. Please close this window and run WINDOWS_RUN.BAT again.
    pause
    exit /b 0
)

REM ====================================================================
REM  Step 2: Install dependencies and run
REM ====================================================================
echo.
echo === [2/2] Launching AutoSub ===

uv sync

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

uv run python autosub_gui.py
if %errorlevel% neq 0 (
    echo.
    echo The application exited with an error.
    pause
)
