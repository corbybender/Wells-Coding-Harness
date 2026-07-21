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
REM PyPI) by running the harness module directly with PYTHONPATH=src. It tells
REM uv to use the Windows system certificate store (UV_SYSTEM_CERTS, or
REM UV_NATIVE_TLS on older uv) - needed on corporate networks whose proxy
REM presents a self-signed cert.

setlocal
REM Save the user's CWD before we cd into the repo root.
set "USER_CWD=%CD%"
cd /d "%~dp0"

REM Use the OS certificate store so corporate TLS-intercepting proxies work.
REM Newer uv renamed UV_NATIVE_TLS to UV_SYSTEM_CERTS (the old name still
REM works but prints a deprecation warning) - set whichever this uv knows.
if defined UV_SYSTEM_CERTS goto :tlsdone
if defined UV_NATIVE_TLS goto :tlsdone
uv sync --help 2>nul | findstr /C:"UV_SYSTEM_CERTS" >nul
if %errorlevel%==0 (set UV_SYSTEM_CERTS=1) else (set UV_NATIVE_TLS=1)
:tlsdone

REM Sync dependencies WITHOUT building the local package — but only when
REM uv.lock has actually changed since the last successful sync. A no-op
REM `uv sync` still costs ~1s on every launch (full lockfile resolution
REM check); skipping it when nothing changed is most of a second saved on
REM every single "wells" invocation, not just the first.
set "STAMP=.venv\.sync-stamp"
set "LOCKSTAMP="
for %%F in ("uv.lock") do set "LOCKSTAMP=%%~tF;%%~zF"
set "CACHEDSTAMP="
if exist "%STAMP%" set /p CACHEDSTAMP=<"%STAMP%"

if "%LOCKSTAMP%"=="%CACHEDSTAMP%" if exist ".venv\Scripts\python.exe" goto :run

uv sync --no-install-project --quiet >nul 2>&1
if %errorlevel%==0 (
    >"%STAMP%" echo %LOCKSTAMP%
    goto :run
)

echo [wells] installing dependencies (first run may take a minute) ...
uv sync --no-install-project
if errorlevel 1 goto :syncfail
>"%STAMP%" echo %LOCKSTAMP%

:run
REM Auto-deploy the wells-index .pyd from the repo into the venv.
REM After a git pull the repo copy is newer; this keeps the venv in sync
REM without any manual copy step. /D = only copy if source is newer, so
REM this is a no-op stat check (not a real copy) on most launches.
set "PYD_SRC=%~dp0wells-index\python\wells_index\_core.cp312-win_amd64.pyd"
set "PYD_DST=%~dp0.venv\Lib\site-packages\wells_index\_core.cp312-win_amd64.pyd"
if exist "%PYD_SRC%" (
    xcopy /Y /Q /D "%PYD_SRC%" "%PYD_DST%" >nul 2>&1
)

REM Install optional semantic-search libraries (fastembed + sqlite-vec) on
REM first launch. Best-effort: stamp-file cached so it only runs once per
REM pyproject.toml change, skips if the libs already import, and never
REM blocks the launch on failure. Set WELLS_NO_EMBEDDINGS=1 to opt out.
if /i "%WELLS_NO_EMBEDDINGS%"=="1" goto :embeddone

set "EMBED_STAMP=.venv\.embed-stamp"
set "EMBED_SPEC="
for %%F in ("pyproject.toml") do set "EMBED_SPEC=%%~tF;%%~zF"
set "EMBED_CACHED="
if exist "%EMBED_STAMP%" set /p EMBED_CACHED=<"%EMBED_STAMP%"

if "%EMBED_SPEC%"=="%EMBED_CACHED%" goto :embeddone

REM Verify libs actually importable (handles manual installs / partial state).
"%~dp0.venv\Scripts\python.exe" -c "import fastembed, sqlite_vec" >nul 2>&1
if %errorlevel%==0 (
    >"%EMBED_STAMP%" echo %EMBED_SPEC%
    goto :embeddone
)

echo.
echo [wells] Installing semantic-search libraries (one-time, ~60s)...
echo [wells]   - fastembed  ^(local ONNX embeddings, ~150 MB^)
echo [wells]   - sqlite-vec ^(vector storage extension^)
echo.
uv pip install fastembed sqlite-vec
if errorlevel 1 (
    echo.
    echo [wells] Semantic-search libraries could not be installed ^(network/TLS error?^).
    echo [wells] Wells will start normally; semantic_search will return a hint
    echo [wells] to install manually. Other tools are unaffected.
    echo.
    goto :embeddone
)
>"%EMBED_STAMP%" echo %EMBED_SPEC%
echo.
echo [wells] Semantic search ready.
echo.

:embeddone
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
if not defined WORKSPACE_ROOT set "WORKSPACE_ROOT=%USER_CWD%"
uv run --no-sync python -m wells.main %*
goto :eof

:syncfail
echo.
echo [wells] Dependency install failed. This is usually a network/TLS issue. 1>&2
echo [wells] If behind a corporate proxy, try setting these environment vars: 1>&2
echo [wells]   set SSL_CERT_FILE=C:\path\to\your-corp-ca-bundle.pem 1>&2
echo [wells]   set REQUESTS_CA_BUNDLE=C:\path\to\your-corp-ca-bundle.pem 1>&2
exit /b 1
