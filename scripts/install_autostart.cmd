@echo off
REM Start the PocketDiffusion live server automatically at every Windows logon (current user only,
REM no administrator rights needed): puts a small launcher into the user's Startup folder.
REM   scripts\install_autostart.cmd            install
REM   scripts\install_autostart.cmd /remove    uninstall
setlocal
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "ENTRY=%STARTUP%\PocketDiffusion live.cmd"
if /i "%~1"=="/remove" (
    del "%ENTRY%" 2>nul
    echo Removed: "%ENTRY%"
    exit /b 0
)
> "%ENTRY%" echo @echo off
>> "%ENTRY%" echo REM PocketDiffusion live server (remove with: E:\PocketDiffusion\scripts\install_autostart.cmd /remove)
>> "%ENTRY%" echo start "PocketDiffusion live" /min cmd /c ""%~dp0autostart_live.cmd""
echo Installed: "%ENTRY%"
echo PocketDiffusion will start automatically (minimized window "PocketDiffusion live") at every logon.
