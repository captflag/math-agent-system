"""
Input Guardrails for validating and sanitizing incoming questions
Ensures questions are mathematics-related and safe
"""
from typing import Tuple, Optional, List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from app.core.config import settings
import logging
import re

logger = logging.getLogger(__name__)

class InputGuardrails:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.allowed_topics = settings.ALLOWED_TOPICS.split(",")
        self.max_length = settings.MAX_QUESTION_LENGTH
        
        # Mathematics-related keywords for topic classification
        self.math_keywords = {
            "algebra", "equation", "solve", "calculate", "derivative", "integral",
            "function", "polynomial", "matrix", "vector", "geometry", "triangle",
            "circle", "angle", "theorem", "proof", "trigonometry", "sin", "cos",
            "tan", "logarithm", "exponential", "statistics", "probability",
            "mean", "median", "variance", "calculus", "limit", "series", "sequence",
            "permutation", "combination", "graph", "plot", "number", "prime",
            "factor", "quadratic", "linear", "differential", "integration"
        }
        
        # Prohibited content patterns
        self.prohibited_patterns = [
            r"hack", r"crack", r"exploit", r"malware", r"virus",
            r"sexual", r"porn", r"xxx", r"violence", r"kill"
        ]
    
    def validate(self, question: str) -> Tuple[bool, Optional[str], str]:
        """
        Validate input question through multiple guardrails
        
        Returns:
            Tuple[bool, Optional[str], str]: (is_valid, sanitized_question, error_message)
        """
        if not settings.ENABLE_INPUT_GUARDRAILS:
            return True, question, ""
        
        # 1. Length validation
        if len(question) > self.max_length:
            return False, None, f"Question exceeds maximum length of {self.max_length} characters"
        
        if len(question.strip()) < 5:
            return False, None, "Question is too short. Please provide more details."
        
        # 2. Prohibited content check
        for pattern in self.prohibited_patterns:
            if re.search(pattern, question.lower()):
                logger.warning(f"Prohibited content detected: {pattern}")
                return False, None, "Question contains prohibited content"
        
        # 3. Topic validation - ensure it's mathematics-related
        is_math_related = self._is_mathematics_topic(question)
        if not is_math_related:
            return False, None, "Question must be related to mathematics. This system is designed for mathematical education only."
        
        # 4. PII detection and anonymization
        sanitized_question = self._detect_and_remove_pii(question)
        
        # 5. SQL injection and XSS prevention
        if self._contains_injection_patterns(sanitized_question):
            return False, None, "Question contains potentially unsafe patterns"
        
        logger.info(f"Input validation passed for question: {sanitized_question[:50]}...")
        return True, sanitized_question, ""
    
    def _is_mathematics_topic(self, question: str) -> bool:
        """Check if question is related to mathematics"""
        question_lower = question.lower()
        
        # Check for math keywords
        keyword_count = sum(1 for keyword in self.math_keywords if keyword in question_lower)
        
        # Check for mathematical symbols
        math_symbols = ['+', '-', '*', '/', '=', '<', '>', '∫', '∑', '√', '^', '²', '³']
        symbol_count = sum(1 for symbol in math_symbols if symbol in question)
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d', question))
        
        # Heuristic: Consider it math-related if:
        # - Has at least 1 math keyword OR
        # - Has math symbols and numbers
        is_math = keyword_count >= 1 or (symbol_count >= 1 and has_numbers)
        
        return is_math
    
    def _detect_and_remove_pii(self, text: str) -> str:
        """Detect and anonymize PII using Presidio"""
        try:
            # Analyze for PII
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN", "PERSON", "LOCATION"]
            )
            
            # Anonymize if PII detected
            if results:
                logger.info(f"PII detected and anonymized: {len(results)} entities")
                anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized.text
            
            return text
        except Exception as e:
            logger.error(f"Error in PII detection: {e}")
            return text
    
    def _contains_injection_patterns(self, text: str) -> bool:
        """Check for SQL injection and XSS patterns"""
        injection_patterns = [
            r"<script", r"</script>", r"javascript:", r"onerror=",
            r"DROP TABLE", r"DELETE FROM", r"INSERT INTO", r"UPDATE.*SET",
            r"' OR '1'='1", r"'; --", r"UNION SELECT"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

# Global instance
input_guardrails = InputGuardrails()
