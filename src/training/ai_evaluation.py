"""
AI-based evaluation utilities using LangChain with Claude API.
Provides quality scoring and detailed feedback on model responses.
"""

import os
import json
from typing import Dict, Any, List, Optional
from src.utils.logging import get_logger
from src.utils.error_handling import EvaluationError, handle_errors

logger = get_logger(__name__)

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not installed. Install with: pip install langchain langchain-anthropic langchain-openai langchain-google-genai")


def _get_llm_for_provider(
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    Get LangChain LLM instance for specified provider.
    
    Args:
        provider: Provider name ("anthropic", "openai", "gemini")
        model: Model name (optional, uses defaults if not provided)
        api_key: API key (optional, uses env vars if not provided)
        
    Returns:
        LangChain LLM instance
    """
    if provider.lower() == "anthropic":
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EvaluationError("ANTHROPIC_API_KEY not found. Set it as environment variable.")
        model = model or "claude-3-5-sonnet-20241022"
        return ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0.0,
            max_tokens=500,
        )
    elif provider.lower() == "openai":
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EvaluationError("OPENAI_API_KEY not found. Set it as environment variable.")
        model = model or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0.0,
            max_tokens=500,
        )
    elif provider.lower() == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EvaluationError("GEMINI_API_KEY not found. Set it as environment variable.")
        model = model or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=500,
        )
    else:
        raise EvaluationError(f"Unknown provider: {provider}. Supported: anthropic, openai, gemini")


@handle_errors(error_type=EvaluationError)
def evaluate_with_ai(
    prompts: List[str],
    generated_responses: List[str],
    expected_responses: Optional[List[str]] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model responses using LangChain with various AI providers for quality assessment.
    
    Args:
        prompts: List of input prompts/scenarios
        generated_responses: List of model-generated responses
        expected_responses: Optional list of expected/reference responses
        provider: AI provider to use ("anthropic", "openai", "gemini")
        model: Model name (optional, uses provider defaults)
        api_key: API key (optional, uses env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY)
        max_samples: Maximum number of samples to evaluate (for cost control)
        
    Returns:
        Dictionary with AI evaluation metrics and feedback
    """
    if not LANGCHAIN_AVAILABLE:
        raise EvaluationError(
            "LangChain not installed. Install with: pip install langchain langchain-anthropic langchain-openai langchain-google-genai"
        )
    
    # Get LLM instance for provider
    llm = _get_llm_for_provider(provider=provider, model=model, api_key=api_key)
    
    # Limit samples if specified
    if max_samples:
        prompts = prompts[:max_samples]
        generated_responses = generated_responses[:max_samples]
        if expected_responses:
            expected_responses = expected_responses[:max_samples]
    
    provider_name = provider.upper()
    model_name = model or "default"
    logger.info(f"Evaluating {len(prompts)} samples with {provider_name} via LangChain (model: {model_name})...")
    
    ai_metrics = {
        "total_samples": len(prompts),
        "evaluated_samples": 0,
        "average_score": 0.0,
        "scores": [],
        "feedback": [],
        "errors": [],
    }
    
    evaluation_prompt_template = """You are evaluating a crisis response AI model. Your task is to assess the quality of the model's response to a crisis scenario.

Evaluate the response based on:
1. **Completeness**: Does it cover all key aspects (FACTS, UNCERTAINTIES, ANALYSIS, GUIDANCE)?
2. **Accuracy**: Are the facts correctly identified and uncertainties properly acknowledged?
3. **Actionability**: Is the guidance clear, specific, and actionable?
4. **Structure**: Is the response well-organized and easy to follow?
5. **Crisis Appropriateness**: Is the response appropriate for the severity and type of crisis?

Provide:
1. A quality score from 0-100 (where 100 is excellent)
2. Brief feedback (2-3 sentences) on strengths and weaknesses
3. A JSON object with scores for each criterion (completeness, accuracy, actionability, structure, appropriateness) each 0-100

Format your response as JSON:
{{
  "score": <overall_score_0-100>,
  "criterion_scores": {{
    "completeness": <0-100>,
    "accuracy": <0-100>,
    "actionability": <0-100>,
    "structure": <0-100>,
    "appropriateness": <0-100>
  }},
  "feedback": "<brief_feedback_text>"
}}

Crisis Scenario:
{prompt}

Model Response:
{response}
"""
    
    total_score = 0.0
    successful_evaluations = 0
    
    for idx, (prompt, response) in enumerate(zip(prompts, generated_responses)):
        try:
            # Prepare evaluation prompt
            eval_prompt = evaluation_prompt_template.format(
                prompt=prompt[:2000],  # Limit prompt length
                response=response[:3000]  # Limit response length
            )
            
            # Call LLM via LangChain
            message = HumanMessage(content=eval_prompt)
            ai_response_obj = llm.invoke([message])
            # Handle different response formats
            if hasattr(ai_response_obj, 'content'):
                ai_response = ai_response_obj.content
            elif isinstance(ai_response_obj, str):
                ai_response = ai_response_obj
            else:
                ai_response = str(ai_response_obj)
            
            # Parse JSON response
            try:
                # Try to extract JSON from response
                json_start = ai_response.find("{")
                json_end = ai_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    evaluation = json.loads(json_str)
                    
                    score = evaluation.get("score", 0)
                    criterion_scores = evaluation.get("criterion_scores", {})
                    feedback = evaluation.get("feedback", "")
                    
                    ai_metrics["scores"].append({
                        "sample_idx": idx,
                        "overall_score": score,
                        "criterion_scores": criterion_scores,
                        "feedback": feedback
                    })
                    
                    total_score += score
                    successful_evaluations += 1
                    
                else:
                    # Fallback: try to extract score from text
                    logger.warning(f"Could not parse JSON from {provider_name} response for sample {idx}")
                    ai_metrics["errors"].append({
                        "sample_idx": idx,
                        "error": f"Could not parse {provider_name} response as JSON",
                        "response": ai_response[:200]
                    })
                    
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error for sample {idx}: {str(e)}")
                ai_metrics["errors"].append({
                    "sample_idx": idx,
                    "error": f"JSON decode error: {str(e)}",
                    "response": ai_response[:200]
                })
                
        except Exception as e:
            logger.error(f"Error evaluating sample {idx} with {provider_name}: {str(e)}")
            ai_metrics["errors"].append({
                "sample_idx": idx,
                "error": f"{provider_name} API error: {str(e)}"
            })
    
    # Calculate average score
    if successful_evaluations > 0:
        ai_metrics["average_score"] = total_score / successful_evaluations
        ai_metrics["evaluated_samples"] = successful_evaluations
    else:
        ai_metrics["average_score"] = 0.0
        logger.warning("No successful AI evaluations completed")
    
    logger.info(f"AI evaluation complete: {successful_evaluations}/{len(prompts)} samples evaluated")
    logger.info(f"Average quality score: {ai_metrics['average_score']:.1f}/100")
    
    return ai_metrics


def add_ai_evaluation_to_metrics(
    base_metrics: Dict[str, Any],
    ai_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge AI evaluation results into base evaluation metrics.
    
    Args:
        base_metrics: Standard evaluation metrics
        ai_metrics: AI evaluation metrics from Claude
        
    Returns:
        Combined metrics dictionary
    """
    combined_metrics = base_metrics.copy()
    
    # Add AI evaluation section
    combined_metrics["ai_evaluation"] = {
        "enabled": True,
        "average_score": ai_metrics.get("average_score", 0.0),
        "evaluated_samples": ai_metrics.get("evaluated_samples", 0),
        "total_samples": ai_metrics.get("total_samples", 0),
        "scores": ai_metrics.get("scores", []),
        "errors": ai_metrics.get("errors", [])
    }
    
    # Add summary statistics
    if ai_metrics.get("scores"):
        criterion_scores = {
            "completeness": [],
            "accuracy": [],
            "actionability": [],
            "structure": [],
            "appropriateness": []
        }
        
        for score_entry in ai_metrics["scores"]:
            criterion = score_entry.get("criterion_scores", {})
            for key in criterion_scores.keys():
                if key in criterion:
                    criterion_scores[key].append(criterion[key])
        
        # Calculate averages
        combined_metrics["ai_evaluation"]["criterion_averages"] = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in criterion_scores.items()
        }
    
    return combined_metrics
