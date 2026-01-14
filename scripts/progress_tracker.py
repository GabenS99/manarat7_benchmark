import time
from typing import Dict, Any

class ProgressTracker:
    """Track and display progress for the prediction pipeline."""
    
    def __init__(self):
        self.start_time = time.time()
        self.file_start_time = None
        self.current_file_idx = 0
        self.total_files = 0
        self.current_question_idx = 0
        self.total_questions = 0
        self.completed_questions = 0
        self.completed_models = 0
        self.total_models = 0
        
    def start_file_processing(self, file_idx: int, total_files: int, filename: str):
        """Start tracking a new file."""
        self.current_file_idx = file_idx
        self.total_files = total_files
        self.file_start_time = time.time()
        self.completed_questions = 0
        print(f"\n{'='*80}")
        print(f"[TRACK] FILE {file_idx}/{total_files}: {filename}")
        print(f"{'='*80}")
        
    def start_question_processing(self, total_questions: int, total_models: int, mode: str = "prediction"):
        """Start tracking questions/tasks in a file.
        
        Args:
            total_questions: Total number of questions (prediction mode) or tasks (evaluation mode)
            total_models: Total number of models or judges
            mode: "prediction" or "evaluation" to customize messaging
        """
        self.total_questions = total_questions
        self.total_models = total_models
        self.current_question_idx = 0
        self.completed_questions = 0
        if total_questions > 0:
            if mode == "evaluation":
                print(f"  [TRACK] Processing {total_questions} evaluation tasks with {total_models} judges")
            else:
                print(f"  [TRACK] Processing {total_questions} questions with {total_models} models")
            if total_models > 0:
                print(f"  [TRACK]  Mode: {'Parallel' if total_models > 1 else 'Sequential'}")
        
    def update_question_progress(self, question_idx: int, total_questions: int = None, question_id: str = None, model_name: str = None, pred_idx: int = None, total_predictions_for_question: int = None):
        """Update progress for current question/task.
        
        Args:
            question_idx: Current question index (1-based)
            total_questions: Total number of questions
            question_id: Question ID for display
            model_name: Prediction model name being evaluated
            pred_idx: Prediction index within current question (1-based)
            total_predictions_for_question: Total predictions for current question
        """
        self.current_question_idx = question_idx
        # Use instance variable if not provided
        if total_questions is None:
            total_questions = self.total_questions
        self.completed_questions = question_idx - 1
        progress_pct = (question_idx / total_questions) * 100 if total_questions > 0 else 0
        
        # Calculate ETA
        elapsed = self._get_elapsed_time()
        if question_idx > 1 and elapsed > 0:
            avg_time_per_question = elapsed / (question_idx - 1)
            remaining_questions = total_questions - question_idx + 1
            eta_seconds = avg_time_per_question * remaining_questions
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        elapsed_str = self._format_time(elapsed)
        
        # Build progress message with optional context
        if question_id and model_name:
            # Evaluation mode: show Question X/Y, Model Z/W (model_name)
            question_part = f"Question {question_idx}/{total_questions}"
            if pred_idx is not None and total_predictions_for_question is not None:
                model_part = f"Model {pred_idx}/{total_predictions_for_question} ({model_name})"
            else:
                model_part = f"Model: {model_name}"
            
            print(f"\n  [TRACK] {question_part} ({progress_pct:.1f}%) | {model_part} | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")
        else:
            # Prediction mode: show question number only
            print(f"\n  [TRACK] Question {question_idx}/{total_questions} ({progress_pct:.1f}%) | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")
        
    def update_model_progress(self, completed: int, total: int, model_name: str, success: bool):
        """Update progress for model completion."""
        self.completed_models = completed
        status = "[DONE]" if success else "[FAILED]"
        print(f"    {status} [{model_name}] ({completed}/{total} models)")
        
    def finish_question(self, question_idx: int):
        """Mark a question as completed."""
        self.completed_questions = question_idx
        if self.total_questions > 0:
            progress_pct = (self.completed_questions / self.total_questions) * 100
            elapsed_str = self._format_time(self._get_elapsed_time())
            print(f"  [DONE] Completed question {question_idx}/{self.total_questions} ({progress_pct:.1f}%) | {elapsed_str}")
        
    def finish_batch(self, batch_num: int, total_batches: int, items_so_far: int, item_type: str = "questions"):
        """Mark a batch as completed.
        
        Args:
            batch_num: Current batch number
            total_batches: Total number of batches
            items_so_far: Number of items (questions/tasks/evaluations) processed so far
            item_type: Type of item for display ("questions" or "evaluations")
        """
        progress_pct = (batch_num / total_batches) * 100 if total_batches > 0 else 0
        elapsed_str = self._format_time(self._get_elapsed_time())
        print(f"  [TRACK] Batch {batch_num}/{total_batches} saved ({progress_pct:.1f}%) | "
              f"{items_so_far} {item_type} | {elapsed_str}")
        
    def finish_file(self, success: bool, questions_processed: int):
        """Mark a file as completed."""
        elapsed_str = self._format_time(self._get_elapsed_time())
        status = "[DONE]" if success else "[FAILED]"
        print(f"\n  {status} File completed: {questions_processed} questions | {elapsed_str}")
        
    def get_summary(self) -> Dict[str, Any]:
        """Get overall progress summary."""
        total_elapsed = time.time() - self.start_time
        return {
            "files_completed": self.current_file_idx,
            "total_files": self.total_files,
            "total_elapsed": total_elapsed,
            "elapsed_str": self._format_time(total_elapsed)
        }
        
    def _get_elapsed_time(self) -> float:
        """Get elapsed time since file start."""
        return time.time() - self.file_start_time if self.file_start_time else 0
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
