@echo off
REM Wells launcher (Windows) - runs the harness from the cloned repo, no install needed.
REM
REM Usage:
REM   wells.bat "your goal"
REM   wells.bat --workspace C:\path\to\project "fix bug"
REM   wells.bat config
REM   wells.bat info
REM
REM This wrapper AVOIDS building the local package (no hatchling needed from
REM PyPI) by running the harness module directly with PYTHONPATH=src. It sets
REM UV_NATIVE_TLS=1 so uv uses the Windows system certificate store - needed on
REM corporate networks whose proxy presents a self-signed cert.

setlocal
cd /d "%~dp0"

REM Use the OS certificate store so corporate TLS-intercepting proxies work.
if not defined UV_NATIVE_TLS set UV_NATIVE_TLS=1

REM Sync dependencies WITHOUT building the local package.
uv sync --no-install-project --quiet >nul 2>&1
if %errorlevel%==0 goto :run

echo [wells] installing dependencies (first run may take a minute) ...
uv sync --no-install-project
if errorlevel 1 goto :syncfail

:run
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
uv run --no-sync python -m coding_harness.main %*
goto :eof

:syncfail
echo.
echo [wells] Dependency install failed. This is usually a network/TLS issue. 1>&2
echo [wells] If behind a corporate proxy, try setting these environment vars: 1>&2
echo [wells]   set SSL_CERT_FILE=C:\path\to\your-corp-ca-bundle.pem 1>&2
echo [wells]   set REQUESTS_CA_BUNDLE=C:\path\to\your-corp-ca-bundle.pem 1>&2
exit /b 1
