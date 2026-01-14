"""
Model Performance Statistics Tracker

Tracks prediction performance metrics per model including:
- Success/failure counts
- Response times
- Token usage (if available)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


class ModelStatsTracker:
    """Track model performance statistics across predictions."""
    
    def __init__(self):
        """Initialize the stats tracker."""
        self.stats = defaultdict(lambda: {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "total_response_time": 0.0,
            "total_tokens": 0,
            "response_times": []  # Keep list for calculating median/percentiles
        })
    
    def _calculate_median(self, values: list) -> float:
        """Calculate median of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 0:
            return (sorted_values[mid - 1] + sorted_values[mid]) / 2
        else:
            return sorted_values[mid]
    
    def record_prediction(
        self,
        provider: str,
        model: str,
        success: bool,
        response_time: Optional[float] = None,
        tokens: Optional[int] = None
    ):
        """
        Record a prediction result for a model.
        
        Args:
            provider: Provider name (e.g., "openai", "gemini")
            model: Model name (e.g., "gpt-4o", "gemini-2.0-flash")
            success: Whether the prediction was successful
            response_time: Response time in seconds (optional)
            tokens: Number of tokens used (optional)
        """
        model_key = f"{provider}/{model}"
        stats = self.stats[model_key]
        
        stats["total_predictions"] += 1
        
        if success:
            stats["successful_predictions"] += 1
        else:
            stats["failed_predictions"] += 1
        
        if response_time is not None:
            stats["total_response_time"] += response_time
            stats["response_times"].append(response_time)
        
        if tokens is not None:
            stats["total_tokens"] += tokens
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all models.
        
        Returns:
            Dictionary with statistics per model
        """
        summary = {}
        
        for model_key, stats in self.stats.items():
            total = stats["total_predictions"]
            successful = stats["successful_predictions"]
            failed = stats["failed_predictions"]
            
            # Calculate success rate
            success_rate = (successful / total * 100) if total > 0 else 0.0
            
            # Calculate average response time
            avg_response_time = (
                stats["total_response_time"] / successful 
                if successful > 0 else 0.0
            )
            
            # Calculate median response time
            median_response_time = self._calculate_median(stats["response_times"])
            
            summary[model_key] = {
                "total_predictions": total,
                "successful_predictions": successful,
                "failed_predictions": failed,
                "success_rate": round(success_rate, 2),
                "avg_response_time": round(avg_response_time, 3),
                "median_response_time": round(median_response_time, 3),
                "total_tokens": stats["total_tokens"]
            }
        
        return summary
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics across all models.
        
        Returns:
            Dictionary with aggregated statistics
        """
        total_predictions = 0
        total_successful = 0
        total_failed = 0
        all_response_times = []
        total_tokens = 0
        
        for stats in self.stats.values():
            total_predictions += stats["total_predictions"]
            total_successful += stats["successful_predictions"]
            total_failed += stats["failed_predictions"]
            all_response_times.extend(stats["response_times"])
            total_tokens += stats["total_tokens"]
        
        success_rate = (total_successful / total_predictions * 100) if total_predictions > 0 else 0.0
        avg_response_time = (sum(all_response_times) / len(all_response_times)) if all_response_times else 0.0
        
        # Calculate median
        median_response_time = self._calculate_median(all_response_times)
        
        return {
            "total_predictions": total_predictions,
            "successful_predictions": total_successful,
            "failed_predictions": total_failed,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "median_response_time": round(median_response_time, 3),
            "total_tokens": total_tokens
        }
    
    def save_stats(self, output_path: Path, filename: Optional[str] = None):
        """
        Save statistics to JSON file.
        
        Args:
            output_path: Directory to save stats file
            filename: Optional custom filename (without extension)
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_performance_{timestamp}"
        
        stats_file = output_path / f"{filename}.json"
        
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "overall_stats": self.get_overall_stats(),
            "per_model_stats": self.get_stats_summary()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return stats_file
    
    def merge(self, other: 'ModelStatsTracker'):
        """
        Merge statistics from another tracker into this one.
        
        Args:
            other: Another ModelStatsTracker instance
        """
        for model_key, other_stats in other.stats.items():
            self.stats[model_key]["total_predictions"] += other_stats["total_predictions"]
            self.stats[model_key]["successful_predictions"] += other_stats["successful_predictions"]
            self.stats[model_key]["failed_predictions"] += other_stats["failed_predictions"]
            self.stats[model_key]["total_response_time"] += other_stats["total_response_time"]
            self.stats[model_key]["total_tokens"] += other_stats["total_tokens"]
            self.stats[model_key]["response_times"].extend(other_stats["response_times"])
    
    def print_summary(self):
        """Print a formatted summary of statistics to console."""
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE STATISTICS")
        print("=" * 80)
        
        overall = self.get_overall_stats()
        print(f"\nOverall Statistics:")
        print(f"  Total Predictions: {overall['total_predictions']}")
        print(f"  Successful: {overall['successful_predictions']}")
        print(f"  Failed: {overall['failed_predictions']}")
        print(f"  Success Rate: {overall['success_rate']}%")
        print(f"  Avg Response Time: {overall['avg_response_time']}s")
        print(f"  Median Response Time: {overall['median_response_time']}s")
        if overall['total_tokens'] > 0:
            print(f"  Total Tokens: {overall['total_tokens']}")
        
        print(f"\nPer-Model Statistics:")
        summary = self.get_stats_summary()
        for model_key in sorted(summary.keys()):
            model_stats = summary[model_key]
            print(f"\n  {model_key}:")
            print(f"    Predictions: {model_stats['total_predictions']}")
            print(f"    Success Rate: {model_stats['success_rate']}%")
            print(f"    Avg Response Time: {model_stats['avg_response_time']}s")
            print(f"    Median Response Time: {model_stats['median_response_time']}s")
            if model_stats['total_tokens'] > 0:
                print(f"    Total Tokens: {model_stats['total_tokens']}")
        
        print("=" * 80)


def is_abstention(response: str) -> bool:
    """
    Check if a response contains an abstention (with or without brackets).
    
    Args:
        response: Response text to check
        
    Returns:
        True if response contains abstention pattern, False otherwise
    """
    if not response:
        return False
    response_str = str(response)
    return "[لا أعلم]" in response_str or "لا أعلم" in response_str


def count_abstentions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Count abstentions (responses containing '[لا أعلم]' or 'لا أعلم') total and per model.
    
    Args:
        data: Dataset dictionary with metadata and questions
        
    Returns:
        Dictionary with 'total' and 'per_model' counts
        Example: {"total": 10, "per_model": {"openai/gpt-4o": 3, "gemini/gemini-2.0-flash": 7}}
    """
    total = 0
    per_model = {}
    
    questions = data.get("questions", [])
    for question in questions:
        if question is None:
            continue
        
        predictions = question.get("predictions", [])
        for pred in predictions:
            raw_response = pred.get("raw_response", "")
            if is_abstention(raw_response):
                total += 1
                model = pred.get("model", "unknown")
                per_model[model] = per_model.get(model, 0) + 1
    
    return {"total": total, "per_model": per_model}


def calculate_evaluation_statistics(
    evaluation_data: Dict[str, Any],
    prediction_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for evaluation results.
    
    This function generates:
    - Summary statistics (total questions, predictions, evaluations, abstentions)
    - Prediction model statistics (performance metrics + average scores from judges)
    - Judge model statistics (performance metrics + score distributions)
    
    Args:
        evaluation_data: Dictionary with evaluation results including 'questions' array
        prediction_metadata: Optional prediction metadata with abstention_count and models_used
        
    Returns:
        Dictionary with comprehensive statistics structure:
        {
            "summary": {...},
            "prediction_model_stats": {...},
            "judge_model_stats": {...}
        }
    """
    questions = evaluation_data.get("questions", [])
    
    # Initialize counters
    total_questions = 0
    total_predictions = 0
    total_evaluations = 0
    
    # Track prediction model metrics
    pred_model_data = defaultdict(lambda: {
        "total_predictions": 0,
        "valid_predictions": 0,
        "abstained": 0,
        "failed": 0,
        "response_times": [],
        "tokens": [],
        "scores": []  # Collect all scores from all judges for this model
    })
    
    # Track judge model metrics
    judge_model_data = defaultdict(lambda: {
        "total_evaluations": 0,
        "valid_evaluations": 0,
        "failed": 0,
        "response_times": [],
        "tokens": [],
        "score_distributions": defaultdict(lambda: defaultdict(int))  # {pred_model: {score: count}}
    })
    
    # Process questions
    for question in questions:
        if question is None:
            continue
        
        total_questions += 1
        
        # Handle both structures: old (predictions array) and new (prediction dict)
        predictions = question.get("predictions", [])
        if isinstance(question.get("prediction"), dict):
            # New structure: single prediction with evaluations
            predictions = [question.get("prediction")]
        
        for prediction in predictions:
            if not prediction:
                continue
            
            total_predictions += 1
            pred_model = prediction.get("model", "unknown")
            pred_data = pred_model_data[pred_model]
            
            pred_data["total_predictions"] += 1
            
            # Track prediction metrics
            # A prediction is VALID only if:
            # 1. success = True
            # 2. raw_response is not None and not empty
            success = prediction.get("success", False)
            raw_response = prediction.get("raw_response")
            is_valid = (
                success == True and
                raw_response is not None and
                str(raw_response).strip() != ''
            )
            
            if is_valid:
                pred_data["valid_predictions"] += 1
                
                # Track response time (only for valid predictions)
                if prediction.get("response_time"):
                    pred_data["response_times"].append(prediction["response_time"])
                
                # Track tokens (only for valid predictions)
                if prediction.get("completion_tokens"):
                    pred_data["tokens"].append(prediction["completion_tokens"])
                
                # Check for abstention
                if is_abstention(raw_response):
                    pred_data["abstained"] += 1
            elif success:
                # success=True but raw_response is None or empty - count as failed
                pred_data["failed"] += 1
            else:
                pred_data["failed"] += 1
            
            # Process evaluations for this prediction
            evaluations = prediction.get("evaluations", [])
            for evaluation in evaluations:
                if not evaluation:
                    continue
                
                total_evaluations += 1
                judge_model = evaluation.get("judge_model", "unknown")
                judge_data = judge_model_data[judge_model]
                
                judge_data["total_evaluations"] += 1
                
                # Track judge metrics
                # Default to False (treat missing success field as failed) for consistency with merge logic
                if evaluation.get("success", False):
                    judge_data["valid_evaluations"] += 1
                    
                    # Track response time
                    if evaluation.get("response_time"):
                        judge_data["response_times"].append(evaluation["response_time"])
                    
                    # Track tokens
                    if evaluation.get("completion_tokens"):
                        judge_data["tokens"].append(evaluation["completion_tokens"])
                    
                    # Extract score (v1 or v2 format)
                    score = None
                    if "score" in evaluation:
                        # v1 format: single score (0.0-1.0)
                        score = evaluation["score"]
                    elif all(k in evaluation for k in ["correctness_score", "relevance_score", "completeness_score"]):
                        # v2 format: average of three scores (0-10 each, convert to 0-1)
                        # Order: 1. Correctness, 2. Relevance, 3. Completeness
                        score = (
                            evaluation["correctness_score"] +
                            evaluation["relevance_score"] +
                            evaluation["completeness_score"]
                        ) / 30.0  # Normalize to 0-1 scale
                    
                    if score is not None:
                        # Add to prediction model's score collection
                        pred_data["scores"].append(score)
                        
                        # Add to judge's score distribution (round to nearest 0.1)
                        score_key = round(score, 1)
                        judge_data["score_distributions"][pred_model][score_key] += 1
                else:
                    judge_data["failed"] += 1
    
    # Build summary statistics
    summary = {
        "total_questions": total_questions,
        "total_predictions": total_predictions,
        "total_evaluations": total_evaluations,
        "total_abstentions": sum(data["abstained"] for data in pred_model_data.values())
    }
    
    # Build prediction model statistics
    prediction_model_stats = {}
    for model, data in pred_model_data.items():
        total = data["total_predictions"]
        valid = data["valid_predictions"]
        failed = data["failed"]
        abstained = data["abstained"]
        
        # Calculate success rate
        success_rate = (valid / total * 100) if total > 0 else 0.0
        
        # Calculate average score (from all judges)
        avg_score = (sum(data["scores"]) / len(data["scores"])) if data["scores"] else 0.0
        
        # Calculate average response time
        # Formula: Sum of all VALID response_time values / Number of VALID predictions
        avg_response_time = (sum(data["response_times"]) / valid) if valid > 0 and data["response_times"] else 0.0
        
        # Calculate total tokens (sum of all VALID completion_tokens)
        total_tokens = sum(data["tokens"])
        
        prediction_model_stats[model] = {
            "total_predictions": total,
            "valid_predictions": valid,
            "abstained": abstained,
            "failed": failed,
            "success_rate": round(success_rate, 2),
            "avg_score": round(avg_score, 3),
            "avg_response_time": round(avg_response_time, 3),
            "total_tokens": total_tokens
        }
    
    # Build judge model statistics
    judge_model_stats = {}
    for judge, data in judge_model_data.items():
        total = data["total_evaluations"]
        valid = data["valid_evaluations"]
        failed = data["failed"]
        
        # Calculate success rate
        success_rate = (valid / total * 100) if total > 0 else 0.0
        
        # Calculate average response time
        avg_response_time = (sum(data["response_times"]) / len(data["response_times"])) if data["response_times"] else 0.0
        
        # Calculate total tokens
        total_tokens = sum(data["tokens"])
        
        # Convert score distributions to sorted dict
        score_dist = {}
        for pred_model, scores_dict in data["score_distributions"].items():
            score_dist[pred_model] = dict(sorted(scores_dict.items()))
        
        judge_model_stats[judge] = {
            "total_evaluations": total,
            "valid_evaluations": valid,
            "failed": failed,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "total_tokens": total_tokens,
            "score_distribution_by_predicted_model": score_dist
        }
    
    return {
        "summary": summary,
        "prediction_model_stats": prediction_model_stats,
        "judge_model_stats": judge_model_stats
    }