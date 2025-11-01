# 🚀 Quick Start Guide

## How to See Your Math Agent Working

You now have **3 ways** to interact with your math agent:

---

## Option 1: Visual Web Interface (Recommended) 🎨

### Step 1: Start the Backend Server
```powershell
cd D:\projects\math-agent-system\backend
python -m uvicorn app.main:app --reload
```

Wait for: `INFO: Application startup complete.`

### Step 2: Open the Web Interface
Open this file in your browser:
```
D:\projects\math-agent-system\frontend\index.html
```

Or simply double-click `frontend/index.html`

### Step 3: Try it!
- Click any example button or type your own question
- Click "🚀 Solve Problem"
- Watch it route to either Knowledge Base or Web Search
- See step-by-step solutions

---

## Option 2: Python Test Script 🐍

### Step 1: Start the Backend (if not running)
```powershell
cd D:\projects\math-agent-system\backend
python -m uvicorn app.main:app --reload
```

### Step 2: Run the Test Script (in a new terminal)
```powershell
cd D:\projects\math-agent-system\backend
python test_agent.py
```

This will automatically test 6 questions:
- 3 from Knowledge Base (quadratic, derivative, circle area)
- 3 requiring Web Search (research topics)

---

## Option 3: Interactive API Documentation 📚

### Step 1: Start the Backend
```powershell
cd D:\projects\math-agent-system\backend
python -m uvicorn app.main:app --reload
```

### Step 2: Open Swagger UI
Visit: http://localhost:8000/api/docs

### Step 3: Test the API
1. Expand `POST /api/v1/question`
2. Click "Try it out"
3. Enter your question:
```json
{
  "question": "Solve x² + 5x + 6 = 0"
}
```
4. Click "Execute"
5. See the JSON response with routing decision, solution, and steps

---

## 🧪 Example Questions to Test

### Knowledge Base Questions (Fast, High Confidence):
```
Solve the quadratic equation: x² + 5x + 6 = 0
Find the derivative of f(x) = x³ + 2x² - 5x + 1
Calculate the area of a circle with radius 7 cm
What is the integral of x²?
Factor x² - 9
```

### Web Search Questions (Novel/Research Topics):
```
Explain the Collatz conjecture and current research
What are the latest approaches to the Riemann Hypothesis?
How is algebraic topology used in machine learning?
What is the ABC conjecture?
Explain topological data analysis
```

---

## 🎯 What You'll See

### For Knowledge Base Questions:
- ⚡ Fast response (~1-2 seconds)
- 📚 Source: "Knowledge Base"
- 🎯 High confidence (>90%)
- ✅ Step-by-step solutions

### For Web Search Questions:
- 🌐 Source: "Web Search"
- ⏱️ Slower response (~5-7 seconds)
- 📖 More contextual explanations
- 🔗 Citations from web sources

---

## 🛠️ Troubleshooting

### "Cannot connect to backend"
Make sure the server is running:
```powershell
cd D:\projects\math-agent-system\backend
python -m uvicorn app.main:app --reload
```

### "ModuleNotFoundError"
Activate your virtual environment:
```powershell
cd D:\projects\math-agent-system\backend
.\venv\Scripts\Activate.ps1
```

### "Port 8000 already in use"
Kill the existing process or use a different port:
```powershell
python -m uvicorn app.main:app --reload --port 8001
```
(Then update the port in the HTML file)

---

## 📊 Understanding the Output

Each response includes:

| Field | Description |
|-------|-------------|
| **source** | `knowledge_base` or `web_search` |
| **confidence** | 0-1 score (higher = more confident) |
| **processing_time** | Seconds taken to solve |
| **solution** | Final answer |
| **steps** | Array of step-by-step explanations |
| **citations** | Web sources used (if web search) |

---

## 🎓 Next Steps

1. **Test with your own questions** - Try different math topics
2. **Submit feedback** - Use the API to improve the agent
3. **Check the routing logic** - See how it decides KB vs Web
4. **Review the code** - Check `app/agents/math_routing_agent.py`

---

## 📹 Creating Your Demo

Record a screen capture showing:

1. Starting the backend server
2. Opening the web interface
3. Testing a KB question (e.g., quadratic equation)
4. Testing a web search question (e.g., Collatz conjecture)
5. Showing the routing decision and confidence scores
6. Demonstrating step-by-step solutions

---

## 🔥 Quick One-Liner Test

Start server and test in one go:
```powershell
# Terminal 1: Start server
cd D:\projects\math-agent-system\backend; python -m uvicorn app.main:app --reload

# Terminal 2: Run tests (after server starts)
cd D:\projects\math-agent-system\backend; python test_agent.py
```

---

## 💡 Pro Tips

- The HTML interface works offline (no npm install needed!)
- Use Swagger UI for detailed JSON responses
- Check `backend/app/main.py` logs to see routing decisions
- Submit feedback via API to trigger DSPy optimization
- Add your own questions to test_agent.py

**Enjoy your AI Math Professor! 🎓✨**
