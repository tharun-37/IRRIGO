@echo off
REM ---------------------------------------------------------------------------
REM IRRIGO V2 - start the backend and the dashboard.
REM
REM Two windows are opened, one per process, so each can be stopped with Ctrl-C
REM without taking the other down. Closing THIS launcher does not stop either -
REM each server owns its own window. The dashboard is at
REM http://127.0.0.1:5173 and the API, with interactive docs, is at
REM http://127.0.0.1:8000/docs
REM
REM The virtual environment is this repo's own .venv. It used to live only in
REM the sibling MP3 tree, which made this launcher depend on a checkout outside
REM the project; .venv is now copied here (and git-ignored), so the fallback below
REM only matters if someone deletes it.
REM
REM The window stays open until a key is pressed. It used to `exit /b 1` on the
REM first problem, which for a double-clicked launcher meant an error flashed for
REM a frame and the console was gone: the message existed, nobody could read it,
REM and the launcher looked like it had simply closed itself. `pause` on every
REM path that reports something is what makes the launcher usable.
REM ---------------------------------------------------------------------------

setlocal
set ROOT=%~dp0
set VENV=%ROOT%.venv\Scripts\python.exe
if not exist "%VENV%" set "VENV=%ROOT%..\MP3\.venv\Scripts\python.exe"

if not exist "%VENV%" (
  echo [error] virtual environment not found.
  echo         create one:        python -m venv .venv
  echo         then install:      .venv\Scripts\python -m pip install -r backend\requirements.txt
  goto fail
)

if not exist "%ROOT%frontend\node_modules\" (
  echo [setup] installing dashboard dependencies, this runs once
  pushd "%ROOT%frontend"
  call npm install
  popd
)

REM ---------------------------------------------------------------------------
REM Pre-flight: a port already in use fails with [winerror 10048] after the app
REM has already loaded its models, which reads like a bug in the app rather than
REM a port clash.
REM
REM Two outcomes rather than one. A port held by *another* program is a real
REM clash: name it and stop. A port held by this project's own server - a
REM launcher left over from an earlier run, or a window nobody closed - means
REM half the app is already up, and the useful response is to keep that half and
REM start only what is missing. Reporting it as an error turned every second
REM run into a failure the user could do nothing about.
REM
REM findstr, not find: Git for Windows puts its usr\bin ahead of System32, so
REM `find` can resolve to GNU find - which takes a file argument, errors out with
REM "No such file or directory", and returns a nonzero level for EVERY line. The
REM test then reads "free" on a port that is held, the check silently never
REM fires, and vite is the thing that reports the clash instead. findstr has no
REM GNU namesake, so the lookup always lands on the Windows one.
REM ---------------------------------------------------------------------------
set "START_API=1"
set "START_WEB=1"

netstat -ano | findstr /c:"LISTENING" | findstr /c:":8000 " >nul 2>&1
if not errorlevel 1 (
  curl -s -m 2 http://127.0.0.1:8000/api/health | findstr /c:"status" >nul 2>&1
  if not errorlevel 1 (
    echo [ok] port 8000 - this project's API is already running, keeping it.
    set "START_API=0"
  ) else (
    echo [error] port 8000 is already in use by another program.
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr /c:"LISTENING" ^| findstr /c:":8000 "') do (
      echo         held by PID %%I ^(tasklist /FI "PID eq %%I" to identify it^)
    )
    echo         stop the other instance first, or close its console window.
    goto fail
  )
)

netstat -ano | findstr /c:"LISTENING" | findstr /c:":5173 " >nul 2>&1
if not errorlevel 1 (
  curl -s -m 2 http://127.0.0.1:5173/ | findstr /c:"IRRIGO" >nul 2>&1
  if not errorlevel 1 (
    echo [ok] port 5173 - this project's dashboard is already running, keeping it.
    set "START_WEB=0"
  ) else (
    echo [error] port 5173 is already in use by another program.
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr /c:"LISTENING" ^| findstr /c:":5173 "') do (
      echo         held by PID %%I ^(tasklist /FI "PID eq %%I" to identify it^)
    )
    echo         stop the other instance first, or close its console window.
    goto fail
  )
)

REM ---------------------------------------------------------------------------
REM Both servers get their own console window.
REM
REM `/k` keeps that window open for as long as the server runs, which is what
REM makes the two independent of this launcher: closing THIS window stops
REM neither. `/c` would be the opposite - the console would exit with the server
REM as its child, so closing the window would kill it with no warning.
REM
REM `/D` sets the working directory instead of `cd /d "..." &&`, which nests
REM quotes inside quotes and is parsed unreliably by `start`.
REM ---------------------------------------------------------------------------
if "%START_API%"=="1" (
  echo [1/2] backend    http://127.0.0.1:8000
  start "IRRIGO V2 API" /D "%ROOT%backend" cmd /k ""%VENV%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
) else (
  echo [1/2] backend    already running
)

if "%START_WEB%"=="1" (
  echo [2/2] dashboard  http://127.0.0.1:5173
  start "IRRIGO V2 Dashboard" /D "%ROOT%frontend" cmd /k "npx vite --host 127.0.0.1 --port 5173 --strictPort"
) else (
  echo [2/2] dashboard  already running
)

if "%START_API%"=="0" if "%START_WEB%"=="0" goto ready

REM Wait for the backend to answer before reporting success, so the dashboard is
REM never pointed at an API that is still loading its models. Skipped when the
REM API was already up - it answered the health check above.
set /a TRIES=0
if "%START_API%"=="0" goto ready
:wait
timeout /t 1 /nobreak >nul
set /a TRIES+=1
curl -s -o nul -m 2 http://127.0.0.1:8000/api/health && goto ready
if %TRIES% lss 60 goto wait

echo.
echo [warn] the backend did not answer within 60s; check the API window for errors.
goto done

:ready
echo.
echo Both servers are up. Dashboard: http://127.0.0.1:5173   API docs: http://127.0.0.1:8000/docs
echo This launcher can be closed; the servers keep running.

:done
echo.
echo Press any key to close this window.
pause >nul
endlocal
exit /b 0

:fail
echo.
echo The servers were NOT started. This window stays open so the message above
echo can be read; a double-clicked launcher that exits immediately hides it.
pause >nul
endlocal
exit /b 1
