"""
Output Guardrails for validating LLM-generated responses
Ensures accuracy, quality, and safety of solutions
"""
from typing import Tuple, List, Dict
from app.core.config import settings
import logging
import re

logger = logging.getLogger(__name__)

class OutputGuardrails:
    def __init__(self):
        self.min_step_count = 1
        self.max_step_count = 20
        self.min_step_length = 10
        
        # Hallucination indicators
        self.hallucination_patterns = [
            r"I don't know", r"I'm not sure", r"cannot be solved",
            r"impossible to determine", r"insufficient information"
        ]
        
        # Quality indicators
        self.quality_keywords = {
            "step", "therefore", "thus", "hence", "because", "given",
            "calculate", "substitute", "simplify", "solve", "find"
        }
    
    def validate(self, solution: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate output solution through multiple guardrails
        
        Args:
            solution: Dict containing steps, final_answer, and other fields
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not settings.ENABLE_OUTPUT_GUARDRAILS:
            return True, None
        
        # 1. Structure validation
        if not self._validate_structure(solution):
            return False, "Invalid solution structure"
        
        # 2. Step quality validation
        if not self._validate_steps(solution.get("steps", [])):
            return False, "Solution steps do not meet quality standards"
        
        # 3. Mathematical notation validation
        if not self._validate_math_notation(solution):
            return False, "Invalid mathematical notation detected"
        
        # 4. Hallucination detection
        if self._detect_hallucination(solution):
            return False, "Potential hallucination detected in solution"
        
        # 5. Citation/reference validation
        if solution.get("source") == "web_search" and not solution.get("references"):
            logger.warning("Web search solution missing references")
        
        # 6. Final answer validation
        if not self._validate_final_answer(solution.get("final_answer", "")):
            return False, "Final answer is missing or invalid"
        
        logger.info("Output validation passed")
        return True, None
    
    def _validate_structure(self, solution: Dict) -> bool:
        """Validate solution has required fields"""
        required_fields = ["steps", "final_answer", "source"]
        return all(field in solution for field in required_fields)
    
    def _validate_steps(self, steps: List[Dict]) -> bool:
        """Validate quality of solution steps"""
        if len(steps) < self.min_step_count:
            logger.warning(f"Too few steps: {len(steps)}")
            return False
        
        if len(steps) > self.max_step_count:
            logger.warning(f"Too many steps: {len(steps)}")
            return False
        
        # Check each step has required fields and minimum length
        for step in steps:
            if not all(k in step for k in ["step_number", "description", "explanation"]):
                return False
            
            if len(step.get("explanation", "")) < self.min_step_length:
                logger.warning(f"Step {step.get('step_number')} explanation too short")
                return False
        
        # Check step numbering is sequential
        for i, step in enumerate(steps, 1):
            if step.get("step_number") != i:
                logger.warning(f"Non-sequential step numbering at position {i}")
                return False
        
        return True
    
    def _validate_math_notation(self, solution: Dict) -> bool:
        """Validate mathematical notation is properly formatted"""
        # Check for common notation errors
        text = str(solution)
        
        # Check for unmatched parentheses
        if text.count('(') != text.count(')'):
            logger.warning("Unmatched parentheses in solution")
            return False
        
        if text.count('[') != text.count(']'):
            logger.warning("Unmatched brackets in solution")
            return False
        
        # Check for incomplete fractions
        if re.search(r'/\s*[^0-9\w\(]', text):
            logger.warning("Incomplete fraction notation")
        
        return True
    
    def _detect_hallucination(self, solution: Dict) -> bool:
        """Detect potential hallucinations in the solution"""
        text = str(solution).lower()
        
        # Check for hallucination indicators
        for pattern in self.hallucination_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Hallucination pattern detected: {pattern}")
                return True
        
        # Check for contradictions (simple heuristic)
        if "correct" in text and "incorrect" in text:
            logger.warning("Potential contradiction detected")
        
        return False
    
    def _validate_final_answer(self, final_answer: str) -> bool:
        """Validate final answer exists and is properly formatted"""
        if not final_answer or len(final_answer.strip()) < 1:
            return False
        
        # Should not be too vague
        vague_answers = ["it depends", "varies", "multiple answers", "unclear"]
        if any(vague in final_answer.lower() for vague in vague_answers):
            logger.warning("Final answer is too vague")
        
        return True
    
    def calculate_quality_score(self, solution: Dict) -> float:
        """
        Calculate quality score for the solution (0-1)
        Used for confidence scoring
        """
        score = 0.0
        max_score = 5.0
        
        # 1. Step completeness (0-1)
        steps = solution.get("steps", [])
        if 3 <= len(steps) <= 10:
            score += 1.0
        elif len(steps) > 0:
            score += 0.5
        
        # 2. Quality keywords present (0-1)
        text = str(solution).lower()
        keyword_count = sum(1 for kw in self.quality_keywords if kw in text)
        score += min(keyword_count / 5.0, 1.0)
        
        # 3. Has references (0-1)
        if solution.get("references"):
            score += 1.0
        elif solution.get("source") == "knowledge_base":
            score += 0.5
        
        # 4. Mathematical notation (0-1)
        if self._validate_math_notation(solution):
            score += 1.0
        
        # 5. Final answer present (0-1)
        if self._validate_final_answer(solution.get("final_answer", "")):
            score += 1.0
        
        return score / max_score

# Global instance
output_guardrails = OutputGuardrails()
