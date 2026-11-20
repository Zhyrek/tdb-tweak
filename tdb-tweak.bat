@echo off
rem Double-click to start tdb-tweak (the first run installs it).
cd /d "%~dp0"
where py >nul 2>nul && (py -3 launch.py %*) || (python launch.py %*)
if errorlevel 1 pause
