@echo off
echo ============================================
echo    INFOSIGHT 3.0 - Security Suite
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [INFO] Python detected: 
python --version
echo.

REM Check Nmap installation
echo [INFO] Checking Nmap installation...
nmap --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Nmap is not installed!
    echo [WARNING] PortScanner tool will not work without Nmap
    echo [WARNING] Download from: https://nmap.org/download.html
    echo.
) else (
    echo [SUCCESS] Nmap is installed
    echo.
)

REM Check Npcap installation (Windows)
echo [INFO] Checking Npcap installation...
if exist "C:\Windows\System32\Npcap" (
    echo [SUCCESS] Npcap is installed
    echo.
) else (
    echo [WARNING] Npcap is not installed!
    echo [WARNING] PortScanner may not work correctly without Npcap
    echo [WARNING] Download from: https://npcap.com/#download
    echo.
)

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
)

REM Check for waitress (required for production server)
echo [INFO] Checking for waitress (production server)...
pip show waitress >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Waitress not found! Installing...
    pip install waitress
    if errorlevel 1 (
        echo [ERROR] Failed to install waitress
        echo [ERROR] Server may not start correctly
        pause
        exit /b 1
    )
    echo [SUCCESS] Waitress installed
    echo.
) else (
    echo [SUCCESS] Waitress is installed
    echo.
)

REM Check for python-dotenv (required for config module)
echo [INFO] Checking for python-dotenv (config module)...
pip show python-dotenv >nul 2>&1
if errorlevel 1 (
    echo [WARNING] python-dotenv not found! Installing...
    pip install python-dotenv
    if errorlevel 1 (
        echo [ERROR] Failed to install python-dotenv
        echo [ERROR] Config module may not work correctly
        pause
        exit /b 1
    )
    echo [SUCCESS] python-dotenv installed
    echo.
) else (
    echo [SUCCESS] python-dotenv is installed
    echo.
)

REM Check for .env file
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo [WARNING] Some features may not work without API keys
    echo [INFO] Create .env file with required API keys
    echo.
)

echo ============================================
echo    Starting INFOSIGHT 3.0 Server...
echo ============================================
echo.
echo [INFO] Server will start using Waitress (production server)
echo [INFO] Default address: http://127.0.0.1:5000
echo [INFO] Server will automatically check and start Ollama/Llama server if needed
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start the application
python server.py

if errorlevel 1 (
    echo.
    echo [ERROR] Server failed to start
    echo [ERROR] Check the error messages above
    pause
    exit /b 1
)

pause