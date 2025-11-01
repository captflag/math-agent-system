"""
LangGraph-based Math Routing Agent
Implements state machine for intelligent routing between KB and web search
"""
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.knowledge.vector_store import vector_store
from app.mcp_client.search_client import search_client
from app.core.config import settings
import logging
import operator

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State for the routing agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    question: str
    sanitized_question: str
    retrieval_attempted: bool
    kb_results: list
    kb_confidence: float
    web_results: list
    source: str
    solution: dict
    error: str

class MathRoutingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.DEFAULT_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_question", self.route_question)
        workflow.add_node("search_knowledge_base", self.search_knowledge_base)
        workflow.add_node("search_web", self.search_web)
        workflow.add_node("generate_solution", self.generate_solution)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define edges
        workflow.set_entry_point("route_question")
        
        # Conditional routing from route_question
        workflow.add_conditional_edges(
            "route_question",
            self.should_search_kb,
            {
                "knowledge_base": "search_knowledge_base",
                "web_search": "search_web",
                "error": "handle_error"
            }
        )
        
        # From KB search, check if results are good enough
        workflow.add_conditional_edges(
            "search_knowledge_base",
            self.kb_results_sufficient,
            {
                "sufficient": "generate_solution",
                "insufficient": "search_web"
            }
        )
        
        # From web search, always generate solution
        workflow.add_edge("search_web", "generate_solution")
        
        # End points
        workflow.add_edge("generate_solution", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def route_question(self, state: AgentState) -> AgentState:
        """Analyze question and determine routing strategy"""
        logger.info("Routing question...")
        
        question = state["sanitized_question"]
        
        # Use LLM to analyze question characteristics
        routing_prompt = f"""Analyze this mathematical question and determine if it likely exists in a knowledge base or requires web search.

Question: {question}

Consider:
1. Is this a standard mathematical problem (algebra, calculus, geometry)?
2. Or is it about recent research, current events, or niche topics?

Respond with ONLY one word: "KNOWLEDGE_BASE" or "WEB_SEARCH"
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=routing_prompt)])
            decision = response.content.strip().upper()
            
            state["messages"] = state.get("messages", []) + [
                HumanMessage(content=routing_prompt),
                AIMessage(content=decision)
            ]
            
            logger.info(f"Routing decision: {decision}")
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            state["error"] = str(e)
        
        return state
    
    async def search_knowledge_base(self, state: AgentState) -> AgentState:
        """Search vector database for relevant problems"""
        logger.info("Searching knowledge base...")
        
        try:
            results = await vector_store.search(
                query=state["sanitized_question"],
                top_k=settings.TOP_K_RESULTS
            )
            
            state["kb_results"] = results
            state["kb_confidence"] = results[0]["score"] if results else 0.0
            state["retrieval_attempted"] = True
            state["source"] = "knowledge_base"
            
            logger.info(f"KB search found {len(results)} results, confidence: {state['kb_confidence']}")
            
        except Exception as e:
            logger.error(f"Error searching KB: {e}")
            state["error"] = str(e)
            state["kb_results"] = []
            state["kb_confidence"] = 0.0
        
        return state
    
    async def search_web(self, state: AgentState) -> AgentState:
        """Search web using MCP server"""
        logger.info("Searching web...")
        
        try:
            results = await search_client.search(state["sanitized_question"])
            
            state["web_results"] = results
            state["source"] = "web_search" if not state.get("kb_results") else "hybrid"
            
            logger.info(f"Web search found {len(results)} results")
            
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            state["error"] = str(e)
            state["web_results"] = []
        
        return state
    
    async def generate_solution(self, state: AgentState) -> AgentState:
        """Generate step-by-step solution using retrieved context"""
        logger.info("Generating solution...")
        
        # Combine context from KB and web
        context = ""
        if state.get("kb_results"):
            context += "From Knowledge Base:\n"
            for r in state["kb_results"][:2]:
                context += f"- {r.get('content', '')}\n"
        
        if state.get("web_results"):
            context += "\nFrom Web Search:\n"
            for r in state["web_results"][:2]:
                context += f"- {r.get('content', '')}\n"
        
        solution_prompt = f"""You are a mathematical professor. Solve this problem with clear, step-by-step explanations suitable for students.

Question: {state['sanitized_question']}

Context:
{context}

Provide your solution in this exact JSON format:
{{
    "steps": [
        {{
            "step_number": 1,
            "description": "Brief description",
            "formula": "Mathematical formula (if applicable)",
            "explanation": "Detailed explanation"
        }}
    ],
    "final_answer": "Clear final answer",
    "references": ["source1", "source2"]
}}

Be precise, educational, and ensure mathematical accuracy.
"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=solution_prompt)])
            
            # Parse response (simplified - would use JSON parsing in production)
            state["solution"] = {
                "steps": [],  # Parse from response
                "final_answer": "Solution generated",
                "source": state["source"],
                "references": []
            }
            
            logger.info("Solution generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating solution: {e}")
            state["error"] = str(e)
        
        return state
    
    async def handle_error(self, state: AgentState) -> AgentState:
        """Handle errors gracefully"""
        logger.error(f"Error handler: {state.get('error')}")
        state["solution"] = {
            "steps": [],
            "final_answer": "Unable to generate solution due to error",
            "source": "error",
            "error": state.get("error")
        }
        return state
    
    def should_search_kb(self, state: AgentState) -> str:
        """Decide whether to search KB first"""
        if state.get("error"):
            return "error"
        
        # Check routing decision from messages
        if state.get("messages"):
            last_message = state["messages"][-1]
            if "KNOWLEDGE_BASE" in last_message.content:
                return "knowledge_base"
            elif "WEB_SEARCH" in last_message.content:
                return "web_search"
        
        # Default to KB search
        return "knowledge_base"
    
    def kb_results_sufficient(self, state: AgentState) -> str:
        """Check if KB results are good enough"""
        confidence = state.get("kb_confidence", 0.0)
        threshold = settings.SIMILARITY_THRESHOLD
        
        if confidence >= threshold:
            logger.info(f"KB results sufficient (confidence: {confidence})")
            return "sufficient"
        else:
            logger.info(f"KB results insufficient (confidence: {confidence}), falling back to web")
            return "insufficient"
    
    async def solve(self, question: str, sanitized_question: str) -> dict:
        """Main entry point for solving a question"""
        initial_state = {
            "messages": [],
            "question": question,
            "sanitized_question": sanitized_question,
            "retrieval_attempted": False,
            "kb_results": [],
            "kb_confidence": 0.0,
            "web_results": [],
            "source": "",
            "solution": {},
            "error": ""
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        return final_state["solution"]

# Global instance
math_agent = MathRoutingAgent()
