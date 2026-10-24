@echo off
REM ---------------------------------------------------------------------------
REM IRRIGO V2 - start the backend and the dashboard, in ONE console.
REM
REM Both processes belong to THIS window: one Ctrl-C stops both, and closing
REM the window stops both. That is the opposite of the old launcher, which
REM spawned a second and a third console ("each can be stopped on its own") and
REM left three windows on the taskbar for one double-click.
REM
REM The API runs in the background of this console and writes to
REM logs\backend.log, so its startup chatter - the advisor snapshots are
REM cached on disk, so the warm-up is a few seconds after the first-ever
REM run - does not interleave with vite's output. The dashboard runs in the
REM foreground, because that is the output worth watching; the API's log stays
REM on disk and its path is printed below.
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
REM One window. /b starts the API as a child of THIS console instead of a new
REM window, so the console's Ctrl-C and its close button reach it. Its output is
REM redirected to a file rather than to the screen: uvicorn logs every request,
REM and mixed into vite's output that is unreadable. A truncated `>` per launch
REM keeps the file to the current run.
REM
REM /D sets the working directory instead of `cd /d "..." &&`, which nests
REM quotes inside quotes and is parsed unreliably by `start`.
REM ---------------------------------------------------------------------------
if not exist "%ROOT%logs" mkdir "%ROOT%logs"

if "%START_API%"=="1" (
  echo [1/2] backend    http://127.0.0.1:8000   log: logs\backend.log
  REM --host 0.0.0.0, not loopback: the bench node posts from the LAN, and a
  REM loopback-only bind never sees its POST. The dashboard keeps using 127.0.0.1
  REM without change; the wider bind only adds the LAN interface.
  start "IRRIGO V2 API" /b /D "%ROOT%backend" cmd /c ""%VENV%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 1>>"%ROOT%logs\backend.log" 2>&1"
) else (
  echo [1/2] backend    already running
)

if "%START_API%"=="0" goto ready

REM Wait for the backend to answer before starting the dashboard, so vite is
REM never pointed at an API that is still loading its models. Skipped when the
REM API was already up - it answered the health check above.
set /a TRIES=0
:wait
REM ping, not timeout: timeout reads the console and exits immediately with
REM "Input redirection is not supported" when stdin is not a terminal, which
REM turned the wait into a tight retry loop.
ping -n 2 127.0.0.1 >nul 2>&1
set /a TRIES+=1
curl -s -o nul -m 2 http://127.0.0.1:8000/api/health && goto ready
if %TRIES% lss 60 goto wait

echo.
echo [warn] the backend did not answer within 60s. Last lines of logs\backend.log:
powershell -NoProfile -Command "Get-Content -Tail 20 '%ROOT%logs\backend.log'" 2>nul
call :stop_api
goto fail

:ready
echo.
if "%START_WEB%"=="1" (
  echo Both servers are up. Dashboard: http://127.0.0.1:5173   API docs: http://127.0.0.1:8000/docs
  echo Ctrl-C stops both. Closing this window stops both.
) else (
  echo API is up: http://127.0.0.1:8000/docs
  echo The dashboard was already running. This window keeps the API alive:
  echo closing it, or Ctrl-C, stops the API.
)

REM The dashboard runs in the foreground of this console, so its output - HMR
REM errors, the port it bound - appears where the user is looking. Vite is
REM invoked through its own entry script rather than `npx`: npx is a batch shim
REM that adds a cmd.exe layer between this window and node, and when that layer
REM is interrupted mid-run the relaunch is confusing. One node process, one
REM Ctrl-C.
if not "%START_WEB%"=="1" goto idle
echo.
pushd "%ROOT%frontend"
node "%ROOT%frontend\node_modules\vite\bin\vite.js" --host 127.0.0.1 --port 5173 --strictPort
popd
echo.
echo Dashboard stopped.
call :stop_api
goto done

:idle
echo.
echo Nothing left to run in this window.
pause >nul
call :stop_api
goto done

REM Kill only the API this launcher started, and only by the port it holds:
REM the pre-flight proved the port was free, so whatever is listening on 8000
REM now is ours. Skipped when the API was already running before we got here.
:stop_api
if not "%START_API%"=="1" exit /b 0
for /f "tokens=5" %%I in ('netstat -ano ^| findstr /c:"LISTENING" ^| findstr /c:":8000 "') do taskkill /f /pid %%I >nul 2>&1
exit /b 0

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
