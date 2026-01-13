#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================"
echo "   INFOSIGHT 3.0 - Security Suite"
echo "============================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo -e "${GREEN}[INFO]${NC} Python detected:"
python3 --version
echo ""

# Check Nmap installation
echo -e "${BLUE}[INFO]${NC} Checking Nmap installation..."
if ! command -v nmap &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Nmap is not installed!"
    echo -e "${YELLOW}[WARNING]${NC} PortScanner tool will not work without Nmap"
    echo ""
    echo "Install Nmap:"
    echo "  Ubuntu/Debian: sudo apt-get install nmap"
    echo "  macOS: brew install nmap"
    echo "  Fedora: sudo dnf install nmap"
    echo ""
else
    echo -e "${GREEN}[SUCCESS]${NC} Nmap is installed"
    nmap --version | head -n 1
    echo ""
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}[INFO]${NC} Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
echo -e "${BLUE}[INFO]${NC} Activating virtual environment..."
source venv/bin/activate
echo ""

# Install/Update dependencies
echo -e "${BLUE}[INFO]${NC} Checking dependencies..."
if ! pip show flask &> /dev/null; then
    echo -e "${BLUE}[INFO]${NC} Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Failed to install dependencies"
        exit 1
    fi
    echo ""
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}[WARNING]${NC} .env file not found!"
    echo -e "${YELLOW}[WARNING]${NC} Some features may not work without API keys"
    echo -e "${BLUE}[INFO]${NC} Create .env file with required API keys"
    echo ""
fi

# Check permissions for network operations
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Not running as root"
    echo -e "${YELLOW}[WARNING]${NC} Some PortScanner features may require sudo"
    echo ""
fi

echo "============================================"
echo "   Starting INFOSIGHT 3.0 Server..."
echo "============================================"
echo ""
echo -e "${BLUE}[INFO]${NC} Server will start on http://127.0.0.1:5000"
echo -e "${BLUE}[INFO]${NC} Press Ctrl+C to stop the server"
echo ""

# Start the application
python3 server.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} Server failed to start"
    echo -e "${RED}[ERROR]${NC} Check the error messages above"
    exit 1
fi