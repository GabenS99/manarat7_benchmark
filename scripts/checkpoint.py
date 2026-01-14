"""
Checkpoint Manager for Prediction Pipeline

Manages checkpointing and resumption of prediction processing.
Allows recovery from interruptions by tracking completed batches.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


def load_saved_questions(
    saved_file: Path,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    batch_num: Optional[int] = None,
    completed_batches: Optional[List[int]] = None
) -> List[Optional[Dict[str, Any]]]:
    """
    Load questions from saved JSON file.
    
    Can load either:
    - All questions (if start_idx and end_idx are None) - for initial resume
    - A specific range (if start_idx and end_idx are provided) - for batch resume
    
    Used when resuming from checkpoint to load already-processed batches
    instead of overwriting them with None placeholders.
    
    Edge cases handled:
    - File doesn't exist: returns [] (signals to reprocess batch)
    - File exists but questions field is None: returns [] (signals to reprocess)
    - File exists but questions list is empty: returns [] (signals to reprocess)
    - File exists but has fewer questions than expected: returns partial data with None padding
    - File exists with None questions: returns them as-is (None may be from input file)
    
    Args:
        saved_file: Path to saved JSON file (typically output_path / "json" / "filename.json")
        start_idx: Start index (inclusive) for question range, or None to load all
        end_idx: End index (exclusive) for question range, or None to load all
        batch_num: Optional batch number for logging (if None, no batch-specific logging)
        completed_batches: Optional list of completed batches (checked when loading all)
        
    Returns:
        List of question dictionaries:
        - All questions if start_idx/end_idx are None (or empty list if not resuming/invalid)
        - Specific range if start_idx/end_idx provided (or empty list if file invalid, signals reprocess)
        - Partial data with None padding if file has fewer questions than expected
    """
    # Load all questions (initial resume)
    if start_idx is None and end_idx is None:
        if completed_batches is None or not completed_batches or not saved_file.exists():
            return []
        
        try:
            with open(saved_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            saved_questions = saved_data.get("questions", [])
            
            # Edge case: questions field is None (not a list)
            if saved_questions is None:
                print("  [WARNING] Saved file has None questions field, starting fresh")
                return []
            
            # Edge case: questions list is empty
            if not saved_questions:
                print("  [WARNING] Saved file has no questions, starting fresh")
                return []
            
            # Note: None questions are preserved as-is (they may be from input file)
            # We don't reprocess them unless we verify the issue is from processing, not input
            print(f"  [RESUME] Loaded {len(saved_questions)} items from existing file")
            return saved_questions
            
        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"  [WARNING] Could not load existing file: {e}, starting fresh")
            return []
        except Exception as e:
            print(f"  [WARNING] Unexpected error loading file: {type(e).__name__}: {e}, starting fresh")
            return []
    
    # Load specific range (batch resume)
    if start_idx is None or end_idx is None:
        raise ValueError("Both start_idx and end_idx must be provided for range loading")
    
    # Edge case: File doesn't exist (checkpoint may be stale)
    if not saved_file.exists():
        if batch_num is not None:
            print(f"  [WARNING] No saved file found for batch {batch_num}, checkpoint may be stale")
            print(f"  [INFO] Will reprocess batch {batch_num} instead of skipping")
        # Return empty list to signal "reprocess this batch" instead of None placeholders
        return []
    
    try:
        with open(saved_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        saved_questions = saved_data.get("questions", [])
        
        # Edge case: questions field is None (not a list)
        if saved_questions is None:
            if batch_num is not None:
                print(f"  [WARNING] Saved file has None questions field for batch {batch_num}, will reprocess")
            return []
        
        # Edge case: questions list is empty
        if not saved_questions:
            if batch_num is not None:
                print(f"  [WARNING] Saved file has no questions for batch {batch_num}, will reprocess")
            return []
        
        # Edge case: File has fewer questions than expected (partial data)
        if len(saved_questions) < end_idx:
            if batch_num is not None:
                print(f"  [WARNING] Saved file has {len(saved_questions)} questions, expected at least {end_idx} for batch {batch_num}")
            
            # If we have some data for this range, return it with None padding
            if len(saved_questions) > start_idx:
                available_count = len(saved_questions) - start_idx
                needed_count = end_idx - start_idx
                result = saved_questions[start_idx:]
                # Pad with None if we don't have enough
                if len(result) < needed_count:
                    result.extend([None] * (needed_count - len(result)))
                if batch_num is not None:
                    print(f"  [RESUME] Loaded {available_count} questions from saved file for batch {batch_num} (partial data)")
                return result
            else:
                # No data available for this range
                if batch_num is not None:
                    print(f"  [WARNING] No data available for range [{start_idx}:{end_idx}], will reprocess batch {batch_num}")
                return []
        
        # Success: return the requested range
        if batch_num is not None:
            print(f"  [RESUME] Loaded {end_idx - start_idx} questions from saved file for batch {batch_num}")
        return saved_questions[start_idx:end_idx]
        
    except (IOError, OSError, json.JSONDecodeError) as e:
        if batch_num is not None:
            print(f"  [WARNING] Could not load saved data for batch {batch_num}: {e}, will reprocess")
        return []
    except Exception as e:
        if batch_num is not None:
            print(f"  [WARNING] Unexpected error loading batch {batch_num}: {type(e).__name__}: {e}, will reprocess")
        return []


def load_or_skip_batch(
    batch_num: int,
    completed_batches: List[int],
    saved_file: Path,
    start_idx: int,
    end_idx: int
) -> Tuple[Optional[List], bool]:
    """
    Load batch from file if completed, or return None to process.
    
    Args:
        batch_num: Current batch number
        completed_batches: List of completed batch numbers (will be modified if batch needs reprocessing)
        saved_file: Path to saved evaluations file
        start_idx: Starting index for batch
        end_idx: Ending index for batch
    
    Returns:
        Tuple of (loaded_data, should_skip):
            - loaded_data: List of loaded questions or None
            - should_skip: True if batch should be skipped (already loaded), False if needs processing
    """
    if batch_num not in completed_batches:
        return None, False
    
    # Try to load from saved file
    loaded = load_saved_questions(saved_file, start_idx, end_idx, batch_num)
    
    # Edge case: marked complete but data unavailable
    if not loaded:
        print(f"  [INFO] Batch {batch_num} marked as completed but data unavailable, reprocessing...")
        completed_batches.remove(batch_num)
        return None, False
    
    return loaded, True


def atomic_write_json(file_path: Path, data: Dict[str, Any], backup: bool = True) -> bool:
    """
    Atomically write JSON data to file to prevent data loss on interruption.
    
    Uses write-then-rename pattern for atomicity:
    1. Write to temporary file (.tmp)
    2. Flush and sync to disk (os.fsync)
    3. Rename temp file to final file (atomic on most filesystems)
    4. Verify final file exists and is valid JSON
    5. Optionally keep backup of previous file
    
    Args:
        file_path: Path to final JSON file
        data: Dictionary to write as JSON
        backup: If True, keep backup of previous file (.bak)
    
    Returns:
        True if write succeeded and was verified, False otherwise
    """
    file_path = Path(file_path)
    temp_file = file_path.with_suffix(file_path.suffix + '.tmp')
    backup_file = file_path.with_suffix(file_path.suffix + '.bak')
    
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup of existing file if it exists and backup is requested
        if backup and file_path.exists():
            try:
                shutil.copy2(file_path, backup_file)
            except Exception as e:
                print(f"[WARNING] Failed to create backup: {e}")
        
        # Write to temporary file
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()  # Flush Python buffer
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename (works on most filesystems)
        temp_file.replace(file_path)
        
        # Verify the write succeeded
        if not file_path.exists():
            print(f"[ERROR] Atomic write failed: final file does not exist")
            return False
        
        # Verify file is valid JSON by reading it back
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Atomic write verification failed: file is not valid JSON: {e}")
            # Restore from backup if available
            if backup_file.exists():
                try:
                    shutil.copy2(backup_file, file_path)
                    print(f"[INFO] Restored from backup")
                except Exception as restore_error:
                    print(f"[ERROR] Failed to restore from backup: {restore_error}")
            return False
        
        # Clean up temp file if it still exists (shouldn't after rename, but just in case)
        # This handles edge cases where rename might not have removed it
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass  # Ignore cleanup errors (temp file may have been removed by rename)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Atomic write failed: {e}")
        # Try to restore from backup if available
        if backup and backup_file.exists() and not file_path.exists():
            try:
                shutil.copy2(backup_file, file_path)
                print(f"[INFO] Restored from backup after error")
            except Exception as restore_error:
                print(f"[ERROR] Failed to restore from backup: {restore_error}")
        # Clean up temp file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        return False


def replace_or_extend_list(
    target_list: List[Any],
    new_items: List[Any],
    start_idx: int
) -> None:
    """
    Replace or extend items in target_list with new_items starting at start_idx.
    
    This helper ensures proper handling when resuming from checkpoints:
    - If target_list already has items at start_idx, they are replaced
    - If target_list is shorter, items are appended
    
    Args:
        target_list: List to modify (modified in-place)
        new_items: New items to insert/replace
        start_idx: Starting index for replacement/extension
    """
    # Replace existing entries or extend as needed
    for i, item in enumerate(new_items):
        idx = start_idx + i
        if idx < len(target_list):
            target_list[idx] = item
        else:
            target_list.append(item)


class CheckpointManager:
    """Manage checkpointing for prediction pipeline."""
    
    def __init__(self, checkpoint_dir: Path, file_identifier: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            file_identifier: Unique identifier for the file being processed
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.file_identifier = file_identifier
        self.checkpoint_file = self.checkpoint_dir / f".checkpoint_{file_identifier}.json"
        self.checkpoint_data = None
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load existing checkpoint if available.
        
        Returns:
            Checkpoint data dictionary or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                self.checkpoint_data = json.load(f)
            return self.checkpoint_data
        except Exception as e:
            print(f"[WARNING]: Failed to load checkpoint: {e}")
            return None
    
    def save_checkpoint(
        self,
        completed_batches: List[int],
        total_batches: int,
        stats: Optional[Dict[str, Any]] = None,
        output_filename: Optional[str] = None
    ):
        """
        Save checkpoint with current progress.
        
        Args:
            completed_batches: List of batch numbers that have been completed
            total_batches: Total number of batches in the file
            stats: Optional statistics to save with checkpoint
            output_filename: Output filename with timestamp (to preserve across resumes)
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "file_identifier": self.file_identifier,
            "completed_batches": completed_batches,
            "total_batches": total_batches,
            "last_updated": datetime.now().isoformat(),
            "stats": stats or {}
        }
        
        # Store output filename if provided (preserves timestamp across resumes)
        if output_filename:
            checkpoint_data["output_filename"] = output_filename
        
        # Use atomic write to prevent data loss on interruption
        success = atomic_write_json(self.checkpoint_file, checkpoint_data, backup=True)
        if success:
            self.checkpoint_data = checkpoint_data
        else:
            print(f"[ERROR] Failed to save checkpoint atomically")
    
    def get_output_filename(self) -> Optional[str]:
        """
        Get stored output filename from checkpoint.
        
        Returns:
            Output filename with timestamp, or None if not stored
        """
        if self.checkpoint_data is None:
            return None
        return self.checkpoint_data.get("output_filename")
    
    def clear_checkpoint(self):
        """Remove checkpoint file when processing is complete."""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                self.checkpoint_data = None
            except Exception as e:
                print(f"[WARNING]: Failed to delete checkpoint: {e}")
    
    def should_skip_batch(self, batch_num: int) -> bool:
        """
        Check if a batch should be skipped (already completed).
        
        Args:
            batch_num: Batch number to check
            
        Returns:
            True if batch is already completed, False otherwise
        """
        if self.checkpoint_data is None:
            return False
        
        completed_batches = self.checkpoint_data.get("completed_batches", [])
        return batch_num in completed_batches
    
    def get_completed_batches(self) -> List[int]:
        """
        Get list of completed batch numbers.
        
        Returns:
            List of completed batch numbers (empty if no checkpoint)
        """
        if self.checkpoint_data is None:
            return []
        return self.checkpoint_data.get("completed_batches", [])
    
    def get_progress_percentage(self) -> float:
        """
        Get progress as a percentage.
        
        Returns:
            Progress percentage (0.0 - 100.0)
        """
        if self.checkpoint_data is None:
            return 0.0
        
        total = self.checkpoint_data.get("total_batches", 0)
        completed = len(self.checkpoint_data.get("completed_batches", []))
        
        if total == 0:
            return 0.0
        
        return (completed / total) * 100.0
    
    def print_resumption_info(self):
        """Print information about resuming from checkpoint."""
        if self.checkpoint_data is None:
            return
        
        total = self.checkpoint_data.get("total_batches", 0)
        completed = len(self.checkpoint_data.get("completed_batches", []))
        remaining = total - completed
        progress = self.get_progress_percentage()
        last_updated = self.checkpoint_data.get("last_updated", "Unknown")
        
        print("\n  [TRACK] Resuming from checkpoint:")
        print(f"     Last updated: {last_updated}")
        print(f"     Progress: {completed}/{total} batches ({progress:.1f}%)")
        print(f"     Remaining: {remaining} batches")


