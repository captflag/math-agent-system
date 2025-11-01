# Implementation Checklist

This checklist tracks which files from `docs/IMPLEMENTATION_GUIDE.md` need to be created as actual files.

## ✅ Already Created Files

- [x] `backend/requirements.txt`
- [x] `backend/.env.example`
- [x] `backend/app/main.py`
- [x] `backend/app/core/config.py`
- [x] `backend/app/models/schemas.py`
- [x] `backend/app/guardrails/input_guardrails.py`
- [x] `backend/app/guardrails/output_guardrails.py`
- [x] `backend/app/agents/math_routing_agent.py`
- [x] `README.md`
- [x] `docs/IMPLEMENTATION_GUIDE.md`
- [x] `docs/FINAL_PROPOSAL.md`
- [x] `PROJECT_SUMMARY.md`
- [x] `setup.ps1`

## 📋 Files to Create from Implementation Guide

### Backend - Knowledge Base
- [ ] `backend/app/knowledge/__init__.py`
- [ ] `backend/app/knowledge/vector_store.py` (Code in Implementation Guide Section 3)

### Backend - MCP Client
- [ ] `backend/app/mcp_client/__init__.py`
- [ ] `backend/app/mcp_client/search_client.py` (Code in Implementation Guide Section 4)

### Backend - Feedback System
- [ ] `backend/app/feedback/__init__.py`
- [ ] `backend/app/feedback/dspy_optimizer.py` (Code in Implementation Guide Section 5)
- [ ] `backend/app/feedback/storage.py` (Code in Implementation Guide Section 5)

### Backend - API Routes
- [ ] `backend/app/api/__init__.py`
- [ ] `backend/app/api/health.py` (Code in Implementation Guide Section 6)
- [ ] `backend/app/api/questions.py` (Code in Implementation Guide Section 6)
- [ ] `backend/app/api/feedback.py` (Code in Implementation Guide Section 6)

### Backend - Scripts
- [ ] `backend/scripts/__init__.py`
- [ ] `backend/scripts/populate_kb.py` (Code in Implementation Guide Section 3)

### MCP Server
- [ ] `mcp-server/requirements.txt` (Add: mcp, tavily-python, httpx)
- [ ] `mcp-server/search_server.py` (Code in Implementation Guide Section 4)

### Frontend
- [ ] Initialize React project: `npm create vite@latest frontend -- --template react`
- [ ] `frontend/package.json` (Dependencies listed in Implementation Guide Section 7)
- [ ] `frontend/src/App.jsx` (Code in Implementation Guide Section 7)
- [ ] `frontend/src/main.jsx` (Standard Vite setup)
- [ ] `frontend/index.html` (Standard Vite setup)
- [ ] `frontend/tailwind.config.js` (Tailwind configuration)

### Benchmarks
- [ ] `benchmarks/jee_benchmark.py` (Code in Implementation Guide Section 8)
- [ ] `data/jee_problems.json` (Sample JEE problems dataset)

### Data Files
- [ ] `data/math_dataset.json` (Sample math problems for KB)
- [ ] `data/feedback.json` (Auto-created by feedback storage)

### Docker & Deployment
- [ ] `Dockerfile` (Code in Implementation Guide Section 9)
- [ ] `docker-compose.yml` (Code in Implementation Guide Section 9)

### Tests
- [ ] `backend/tests/__init__.py`
- [ ] `backend/tests/test_guardrails.py`
- [ ] `backend/tests/test_agent.py`
- [ ] `backend/tests/test_api.py`

## 🔧 Implementation Steps

### Step 1: Create Missing Init Files
```powershell
# Backend package init files
New-Item -ItemType File -Path backend/app/knowledge/__init__.py
New-Item -ItemType File -Path backend/app/mcp_client/__init__.py
New-Item -ItemType File -Path backend/app/feedback/__init__.py
New-Item -ItemType File -Path backend/app/api/__init__.py
New-Item -ItemType File -Path backend/scripts/__init__.py
New-Item -ItemType File -Path backend/tests/__init__.py
```

### Step 2: Copy Code from Implementation Guide

For each file marked with "Code in Implementation Guide", open `docs/IMPLEMENTATION_GUIDE.md` and:
1. Find the relevant section number
2. Copy the Python/JavaScript code block
3. Create the file and paste the code
4. Save the file

**Example:**
```powershell
# For vector_store.py, copy code from Section 3 of Implementation Guide
code backend/app/knowledge/vector_store.py
# Paste code from Implementation Guide Section 3
```

### Step 3: Setup Frontend

```powershell
# Create React project with Vite
npm create vite@latest frontend -- --template react
cd frontend

# Install dependencies
npm install axios react-router-dom

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Copy App.jsx code from Implementation Guide Section 7
# Replace src/App.jsx with the code provided
```

### Step 4: Create Sample Data

Create `data/jee_problems.json`:
```json
[
  {
    "id": "jee_001",
    "question": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
    "answer": "3x^2 + 4x - 5",
    "topic": "calculus"
  },
  {
    "id": "jee_002",
    "question": "Solve x^2 + 5x + 6 = 0",
    "answer": "x = -2 or x = -3",
    "topic": "algebra"
  }
]
```

Create `data/math_dataset.json` with similar structure but more problems.

### Step 5: MCP Server Requirements

Create `mcp-server/requirements.txt`:
```
mcp==0.1.0
tavily-python==0.3.0
httpx==0.26.0
python-dotenv==1.0.0
```

### Step 6: Run Setup Script

```powershell
.\setup.ps1
```

### Step 7: Test the System

```powershell
# Terminal 1: Backend
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

# Terminal 2: MCP Server
cd mcp-server
python search_server.py

# Terminal 3: Frontend
cd frontend
npm run dev

# Browser
http://localhost:5173
```

## 📝 Quick Reference: Where to Find Code

| File | Implementation Guide Section |
|------|------------------------------|
| `vector_store.py` | Section 3: Knowledge Base Setup |
| `search_client.py` | Section 4.2: MCP Client Code |
| `search_server.py` | Section 4.1: MCP Server Code |
| `dspy_optimizer.py` | Section 5: Feedback System Code |
| `storage.py` | Section 5: Feedback Storage |
| `health.py` | Section 6: Health Check |
| `questions.py` | Section 6: Question Routes |
| `feedback.py` | Section 6: Feedback Routes |
| `populate_kb.py` | Section 3: Population Script |
| `App.jsx` | Section 7: Main Component |
| `jee_benchmark.py` | Section 8: Benchmark Script |
| `Dockerfile` | Section 9: Backend Dockerfile |
| `docker-compose.yml` | Section 9: Docker Compose |

## ✅ Verification Steps

After implementing all files, verify:

1. [ ] Backend starts without errors: `uvicorn app.main:app --reload`
2. [ ] MCP server starts: `python mcp-server/search_server.py`
3. [ ] Frontend builds: `npm run dev`
4. [ ] Health check returns 200: `curl http://localhost:8000/api/v1/health`
5. [ ] Qdrant is running: `curl http://localhost:6333`
6. [ ] API docs accessible: `http://localhost:8000/api/docs`
7. [ ] Frontend loads: `http://localhost:5173`
8. [ ] Submit a test question successfully
9. [ ] Feedback submission works
10. [ ] Benchmark script runs: `python benchmarks/jee_benchmark.py`

## 🎯 Priority Order for Implementation

**Phase 1 - Core Backend (Can test immediately):**
1. `vector_store.py`
2. `search_client.py`
3. `health.py`
4. `questions.py`
5. `populate_kb.py`

**Phase 2 - MCP & Feedback:**
6. `search_server.py` (MCP server)
7. `storage.py`
8. `feedback.py`
9. `dspy_optimizer.py`

**Phase 3 - Frontend:**
10. Initialize React project
11. `App.jsx`
12. Tailwind setup

**Phase 4 - Testing & Benchmarks:**
13. `jee_benchmark.py`
14. Test files
15. Docker setup

## 💡 Tips

- **Start with Phase 1** to get a working backend quickly
- **Test each file** as you create it
- **Use the guide**: All code is provided, just copy-paste carefully
- **Check imports**: Make sure all `__init__.py` files exist
- **Environment variables**: Don't forget to set API keys in `.env`

## 🆘 Troubleshooting

**Import errors?**
- Make sure all `__init__.py` files exist
- Check Python path: `export PYTHONPATH=$PYTHONPATH:/path/to/backend`

**Module not found?**
- Reinstall requirements: `pip install -r requirements.txt`
- Check virtual environment is activated

**API not responding?**
- Check logs in terminal
- Verify port 8000 is not in use
- Check .env file has API keys

**Frontend errors?**
- Run `npm install` again
- Check Node version: `node --version` (should be 18+)
- Clear cache: `npm cache clean --force`

---

**Note**: All code for these files is provided in `docs/IMPLEMENTATION_GUIDE.md`. You just need to create the files and copy the code. No additional coding required!
