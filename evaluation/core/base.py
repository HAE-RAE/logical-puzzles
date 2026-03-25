import logging
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    puzzle_id: str
    difficulty: str
    correct: bool
    partial_score: float
    expected: Any
    predicted: Any
    raw_response: str
    latency_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "difficulty": self.difficulty,
            "correct": self.correct,
            "partial_score": self.partial_score,
            "expected": str(self.expected),
            "predicted": str(self.predicted) if self.predicted is not None else None,
            "raw_response": self.raw_response,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class TaskEvaluator(Protocol):
    """
    Minimum interface that each task must implement
    
    Uses Protocol to support duck typing.
    """
    
    def evaluate(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        use_async: bool = False,
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        ...


class BaseEvaluator(ABC):
    """
    Base class providing common logic for all Evaluators
    
    Each Evaluator should inherit from this class and implement only:
    - SYSTEM_PROMPT: Define as a class variable
    - _parse_answer(): Parse LLM response
    - _check_answer(): Check answer correctness
    """
    
    SYSTEM_PROMPT: str
    
    def evaluate(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        use_async: bool = False,
        max_concurrent: int = 10,
        task_name: Optional[str] = None
    ) -> List[EvaluationResult]:
        """
        Execute evaluation
        
        Args:
            puzzles: List of puzzles to evaluate
            llm_client: LLM client
            verbose: Whether to output progress
            use_async: Whether to use async mode
            max_concurrent: Maximum number of concurrent executions in async mode
            task_name: Task name (for logging display)
            
        Returns:
            List of evaluation results
        """
        self._task_name = task_name
        if use_async:
            if verbose:
                logger.info("Starting async evaluation...")
            try:
                # Check if there's already a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    logger.warning("Event loop already running, falling back to synchronous mode")
                    use_async = False
                except RuntimeError:
                    # No running loop, can use asyncio.run() normally
                    pass
                
                if use_async:
                    try:
                        # Create and run event loop directly
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            results = loop.run_until_complete(
                                self._evaluate_async(puzzles, llm_client, verbose, max_concurrent)
                            )
                        finally:
                            # Clean up event loop
                            try:
                                loop.close()
                            except Exception as e:
                                if verbose:
                                    logger.warning(f"Error closing loop: {e}")
                            finally:
                                asyncio.set_event_loop(None)
                        
                        return results
                    except Exception as e:
                        logger.error(f"Error in async evaluation: {e}")
                        import traceback
                        if verbose:
                            logger.debug(traceback.format_exc())
                        raise
                else:
                    # Fallback to synchronous mode
                    results = []
                    for i, puzzle in enumerate(puzzles):
                        result = self._evaluate_single(puzzle, llm_client)
                        results.append(result)
                        
                        if verbose:
                            status = "CORRECT" if result.correct else "INCORRECT"
                            logger.info(f"[{i+1}/{len(puzzles)}] {puzzle['id']}... {status}")
                    
                    return results
            except Exception as e:
                logger.error(f"Error in async evaluation: {e}")
                import traceback
                if verbose:
                    logger.debug(traceback.format_exc())
                raise
        else:
            results = []
            for i, puzzle in enumerate(puzzles):
                result = self._evaluate_single(puzzle, llm_client)
                results.append(result)
                
                if verbose:
                    status = "CORRECT" if result.correct else "INCORRECT"
                    logger.info(f"[{i+1}/{len(puzzles)}] {puzzle['id']}... {status}")
            
            return results
    
    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient"
    ) -> EvaluationResult:
        """
        Evaluate a single puzzle
        
        Args:
            puzzle: Puzzle data
            llm_client: LLM client
            
        Returns:
            Evaluation result
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": puzzle["question"]}
        ]
        
        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, "", latency, {"error": str(e)})
    
    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10
    ) -> List[EvaluationResult]:
        """
        Execute async evaluation
        
        Args:
            puzzles: List of puzzles to evaluate
            llm_client: LLM client
            verbose: Whether to output progress
            max_concurrent: Maximum number of concurrent executions
            
        Returns:
            List of evaluation results
        """
        # Prepare all messages
        messages_list = []
        for puzzle in puzzles:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": puzzle["question"]}
            ]
            messages_list.append(messages)
        
        total_puzzles = len(puzzles)
        task_name = getattr(self, '_task_name', None)
        task_prefix = f"[{task_name}] " if task_name else ""
        
        if verbose:
            logger.info(f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, max_concurrent={max_concurrent}")
        
        # Async batch generation
        start_time = time.time()
        
        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)")
        
        responses = await llm_client.async_batch_generate(
            messages_list, 
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None
        )
        total_latency = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in {total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)")
        
        # Process results
        results = []
        correct_count = 0
        error_count = 0
        
        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)
            
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            
            results.append(result)
        
        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            logger.info(f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, {error_count} errors")
        
        return results
    
    def _process_response(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Common response processing logic (shared by sync/async)
        
        Args:
            puzzle: Puzzle data
            response: LLM response text
            latency_ms: Response latency (milliseconds)
            usage: LLM usage information (may include error information)
            
        Returns:
            EvaluationResult object
        """
        usage = usage or {}
        
        # Check for LLM call failure (if error is in usage)
        if "error" in usage:
            return self._create_error_result(
                puzzle, 
                response if response else "", 
                latency_ms, 
                usage["error"]
            )
        
        try:
            # Parse answer (call abstract method)
            predicted = self._parse_answer(response, puzzle)
            
            # Check answer (call abstract method)
            correct, partial_score = self._check_answer(
                puzzle["answer"],
                predicted
            )
            
            return EvaluationResult(
                puzzle_id=puzzle["id"],
                difficulty=puzzle.get("difficulty", "Unknown"),
                correct=correct,
                partial_score=partial_score,
                expected=puzzle["answer"],
                predicted=predicted,
                raw_response=response,
                latency_ms=latency_ms
            )
        except Exception as e:
            # Error during response parsing/processing
            return self._create_error_result(puzzle, response, latency_ms, str(e))
    
    def _create_error_result(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        error: str
    ) -> EvaluationResult:
        """
        Create error result
        
        Args:
            puzzle: Puzzle data
            response: LLM response text (can be empty string on error)
            latency_ms: Response latency (milliseconds)
            error: Error message
            
        Returns:
            EvaluationResult object (error state)
        """
        return EvaluationResult(
            puzzle_id=puzzle["id"],
            difficulty=puzzle.get("difficulty", "Unknown"),
            correct=False,
            partial_score=0.0,
            expected=puzzle["answer"],
            predicted=None,
            raw_response=response if response else "",
            latency_ms=latency_ms,
            error=error
        )
    
    @abstractmethod
    def _parse_answer(self, response: str, puzzle: Dict[str, Any]) -> Optional[Any]:
        """
        Extract answer from LLM response (implemented by each evaluator)
        
        Args:
            response: LLM response text
            puzzle: Puzzle data (use if needed)
            
        Returns:
            Parsed answer (can be None)
        """
        pass
    
    @abstractmethod
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Any]
    ) -> Tuple[bool, float]:
        """
        Check answer (implemented by each evaluator)
        
        Args:
            expected: Correct answer
            predicted: Predicted answer
            
        Returns:
            (is_correct, partial_score) tuple
        """
        pass
