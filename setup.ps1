# Math Agent System - Quick Setup Script
# PowerShell script for Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Math Routing Agent - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠ Docker not found. You'll need it for Qdrant" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Backend..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Setup backend
cd backend

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Copy environment file
if (Test-Path .env) {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
} else {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "⚠ Please edit backend/.env and add your API keys!" -ForegroundColor Yellow
}

cd ..

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Frontend..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if frontend directory needs setup
if (Test-Path frontend/package.json) {
    cd frontend
    Write-Host "Installing Node dependencies..." -ForegroundColor Yellow
    npm install
    cd ..
} else {
    Write-Host "⚠ Frontend package.json not found. Run ``'npm create vite@latest frontend -- --template react``' manually" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Qdrant (Vector Database)..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Starting Qdrant with Docker..." -ForegroundColor Yellow
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Qdrant started successfully on port 6333" -ForegroundColor Green
} else {
    Write-Host "⚠ Failed to start Qdrant. Please run manually: docker run -p 6333:6333 qdrant/qdrant" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Edit backend/.env and add your API keys:" -ForegroundColor White
Write-Host "   - OPENAI_API_KEY" -ForegroundColor Gray
Write-Host "   - ANTHROPIC_API_KEY" -ForegroundColor Gray
Write-Host "   - TAVILY_API_KEY" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Populate the knowledge base:" -ForegroundColor White
Write-Host "   cd backend" -ForegroundColor Gray
Write-Host "   python scripts/populate_kb.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start the backend:" -ForegroundColor White
Write-Host "   cd backend" -ForegroundColor Gray
Write-Host "   uvicorn app.main:app --reload" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Start the MCP server (in new terminal):" -ForegroundColor White
Write-Host "   cd mcp-server" -ForegroundColor Gray
Write-Host "   python search_server.py" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Start the frontend (in new terminal):" -ForegroundColor White
Write-Host "   cd frontend" -ForegroundColor Gray
Write-Host "   npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "6. Open http://localhost:5173 in your browser" -ForegroundColor White
Write-Host ""

Write-Host "Documentation:" -ForegroundColor Green
Write-Host "- README.md - Project overview" -ForegroundColor Gray
Write-Host "- docs/IMPLEMENTATION_GUIDE.md - Detailed implementation" -ForegroundColor Gray
Write-Host "- docs/FINAL_PROPOSAL.md - Complete proposal with rationale" -ForegroundColor Gray
Write-Host ""

Write-Host "To run benchmarks:" -ForegroundColor Green
Write-Host "   cd benchmarks" -ForegroundColor Gray
Write-Host "   python jee_benchmark.py --dataset data/jee_problems.json" -ForegroundColor Gray
Write-Host ""

Write-Host "Happy building! 🚀" -ForegroundColor Cyan
