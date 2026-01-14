"""
Main script for prediction pipeline.

Processes all data_to_predict files with all prediction_models configured.
"""
import traceback
from typing import Dict, Any, List, Tuple, Callable, Optional
from dataclasses import asdict
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from constants import QuestionType
from config_loader import load_config
from predictors import (
    get_prediction_gemini, get_prediction_groq, get_prediction_mistral,
    get_prediction_chatgpt, get_prediction_deepseek, get_prediction_fanar,
    get_prediction_local_jais, get_prediction_local_allam, get_prediction_local_jais2
)
from response_parser import apply_mcq_post_processing
from output_saver import save_predictions, build_question_structure, save_batch_predictions
from checkpoint import CheckpointManager, load_saved_questions, replace_or_extend_list, load_or_skip_batch, atomic_write_json
from stats_tracker import ModelStatsTracker
from progress_tracker import ProgressTracker


def initialize_pipeline(config_path: str = "config.yaml") -> Tuple[Any, bool]:
    """
    This function initializes and validates the prediction pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple of (config_loader_instance, True if initialization and validation were successful)
    """
    print("=" * 80)
    print("PHASE 1: INITIALIZATION & CONFIGURATION")
    print("=" * 80)
    
    # Load configuration
    try:
        config = load_config(config_path)
        print("[OK] Configuration loaded successfully")
    except FileNotFoundError as e:
        print(f"[ERROR] Configuration file not found: {e}")
        return None, False
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        #print(f"🔍 DEBUG: Exception type: {type(e).__name__}")
        #print(f"🔍 DEBUG: Exception details: {str(e)}")
        traceback.print_exc()
        return None, False
    
    # Display configuration summary
    print("\n" + config.summary())
    
    # Ensure output directories exist
    try:
        #print(f"🔍 DEBUG: Creating/verifying output directories...")
        config.ensure_output_dirs()
        print("[OK] Output directories created/verified")
        #print(f"🔍 DEBUG: Base output paths: {list(config.paths.output.keys())}")
    except Exception as e:
        print(f"[WARNING] Could not create output directories: {e}")
        #print(f"🔍 DEBUG: Directory creation exception: {type(e).__name__}")
        # Continue anyway - directories might already exist
    
    # Validate input paths exist (questions for prediction pipeline)
    #print(f"🔍 DEBUG: Validating input paths...")
    all_exist, missing_paths = config.validate_paths_exist(
        check_input=True, 
        check_output=False,
        input_data_type="questions"  # Explicit: prediction pipeline uses questions as input
    )
    #print(f"🔍 DEBUG: All paths exist: {all_exist}, Missing count: {len(missing_paths)}")
    if not all_exist:
        print(f"\n[WARNING] {len(missing_paths)} input path(s) missing:")
        for path in missing_paths[:5]:  # Show first 5
            print(f"  - {path}")
        if len(missing_paths) > 5:
            print(f"  ... and {len(missing_paths) - 5} more")
        print("  Processing will continue, but missing files will be skipped.")
    else:
        print("\n[OK] All input paths validated")
    
    print("=" * 80)
    #print(f"🔍 DEBUG: Initialization complete, returning (config, True)")
    return config, True


def get_predictor_functions(model_dict: Dict[str, List[str]]) -> List[Tuple[str, str, Callable]]:
    """
    Get predictor functions for each model in the prediction_models dictionary.
    
    Args:
        model_dict: Dictionary mapping provider names to lists of model names
        
    Returns:
        List of tuples: (provider, model_name, prediction_function)
    """
    #print(f"🔍 DEBUG: get_predictor_functions() called with {len(model_dict)} providers")
    #print(f"🔍 DEBUG: Providers in model_dict: {list(model_dict.keys())}")
    
    # Map provider names to prediction functions
    PROVIDER_FUNCTIONS = {
        "openai": get_prediction_chatgpt,
        "gemini": get_prediction_gemini,
        "deepseek": get_prediction_deepseek,
        "fanar": get_prediction_fanar,
        "groq": get_prediction_groq,
        "mistral": get_prediction_mistral,
        "local_jais": get_prediction_local_jais,
        "local_jais2": get_prediction_local_jais2,
        "local_allam": get_prediction_local_allam,
    }
    
    predictor_functions = []
    for provider, model_list in model_dict.items():
        #print(f"🔍 DEBUG: Processing provider '{provider}' with {len(model_list)} models: {model_list}")
        
        if provider not in PROVIDER_FUNCTIONS:
            #print(f"🔍 DEBUG: ERROR - Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDER_FUNCTIONS.keys())}")
        
        prediction_func = PROVIDER_FUNCTIONS[provider]
        #print(f"🔍 DEBUG: Mapped provider '{provider}' to function: {prediction_func.__name__}")
        
        for model_name in model_list:
            predictor_functions.append((provider, model_name, prediction_func))
            #print(f"🔍 DEBUG: Added predictor: ({provider}, {model_name}, {prediction_func.__name__})")
    
    #print(f"🔍 DEBUG: Total predictor functions created: {len(predictor_functions)}")
    return predictor_functions


def call_with_retry(
    prediction_func: Callable,
    kwargs: Dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 5.0,
    retry_backoff: float = 2.0,
    provider: Optional[str] = None,
    model_version: str = "unknown") -> Tuple[str, float, bool, int]:
    """
    Call prediction function with retry logic and exponential backoff.
    
    Args:
        prediction_func: Function to call
        kwargs: Arguments to pass to function
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        retry_backoff: Multiplier for exponential backoff
        provider: Provider name for logging
        model_version: Model name for logging
    
    Returns:
        Tuple of (prediction_text, response_time, success, attempt_number)
    """
    provider_info = f"{provider}/" if provider else ""
    
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        start_time = time.time()
        
        try:
            prediction = prediction_func(**kwargs)
            response_time = time.time() - start_time
            
            # Check if prediction is valid (not None or empty)
            if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
                if attempt < max_retries:
                    delay = retry_delay * (retry_backoff ** attempt)
                    print(f"      [WARNING] Attempt {attempt + 1}/{max_retries + 1} returned None/empty for [{provider_info}{model_version}]")
                    print(f"      [RETRY] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"      [ERROR] All {max_retries + 1} attempts failed for [{provider_info}{model_version}]: returned None/empty")
                    return "", response_time, False, attempt + 1
            
            # Success!
            if attempt > 0:
                print(f"      [OK] Succeeded on attempt {attempt + 1}/{max_retries + 1} for [{provider_info}{model_version}]")
            return prediction, response_time, True, attempt + 1
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e).lower()
            error_str = str(e)
            
            # Check if it's quota exhaustion (permanent) vs rate limiting (temporary)
            is_quota_exhausted = any(keyword in error_msg for keyword in [
                'quota exceeded', 'resource_exhausted', 'limit: 0', 
                'exceeded your current quota'
            ])
            is_rate_limit = any(keyword in error_msg for keyword in [
                'rate limit', 'too many requests', '429'
            ]) and not is_quota_exhausted
            
            # Check if it's authentication error (won't be fixed by retrying)
            is_auth_error = any(keyword in error_msg for keyword in [
                'gated repo', '401', 'authentication', 'access denied',
                'cannot access gated repo', 'restricted', 'must be authenticated'
            ])
            
            # Don't retry if quota is exhausted - it won't work until quota resets
            if is_quota_exhausted:
                print(f"      [ERROR] Quota exhausted for [{provider_info}{model_version}]: {type(e).__name__}")
                print(f"      [SKIP] Skipping retries - quota will not reset until next billing period")
                return "", response_time, False, attempt + 1
            
            # Don't retry authentication errors - they require user action
            if is_auth_error:
                print(f"      [ERROR] Authentication error for [{provider_info}{model_version}]: {type(e).__name__}")
                print(f"      [SKIP] Skipping retries - authentication requires user action")
                print(f"      [INFO] For gated repositories, you need:")
                print(f"        1. Hugging Face token in .env file (HUGGINGFACE_TOKEN)")
                print(f"        2. Access granted to the model repository")
                return "", response_time, False, attempt + 1
            
            if attempt < max_retries:
                delay = retry_delay * (retry_backoff ** attempt)
                if is_rate_limit:
                    delay *= 2  # Double delay for rate limits
                
                print(f"      [WARNING] Attempt {attempt + 1}/{max_retries + 1} failed for [{provider_info}{model_version}]: {type(e).__name__}: {error_str[:200]}")
                print(f"      [RETRY] Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"      [ERROR] All {max_retries + 1} attempts failed for [{provider_info}{model_version}]: {type(e).__name__}: {error_str[:200]}")
                return "", response_time, False, attempt + 1
    
    # Should never reach here
    return "", 0.0, False, max_retries + 1


def process_question(
    prediction_func: Callable,
    model_version: str,
    question_data: Dict[str, Any],
    question_type: QuestionType,
    provider: Optional[str] = None,
    **prediction_config) -> Dict[str, Any]:
    """
    Process a single question with a prediction model, including retry logic.
    
    Args:
        prediction_func: Prediction function to call (signature: func(question, model_version, 
                        question_type, temperature, max_tokens, **kwargs))
        model_version: Model version/name
        question_data: Question data dictionary
        question_type: Type of question (MCQ, COMP, KNOW)
        provider: Provider name (optional)
        **prediction_config: Prediction parameters including:
            - Predictor params: few_shots, show_cot, discipline, verbose_instructions, 
              abstention, verbalized_elicitation, temperature, max_tokens, word_limit,
              choice1-4 (MCQ), text (COMP)
            - Retry params: max_retries, retry_delay, retry_backoff
    
    Returns:
        Dictionary with prediction results (model, raw_response, completion_tokens, 
        response_time, success, generated_at)
    """
    # Extract retry parameters (these are for call_with_retry, not predictors)
    max_retries = prediction_config.get("max_retries", 3)
    retry_delay = prediction_config.get("retry_delay", 5.0)
    retry_backoff = prediction_config.get("retry_backoff", 2.0)
    
    # Extract question fields
    question = question_data.get("question", "")
    
    # Build predictor kwargs for the refactored predictor signatures:
    # Predictors now have: func(question, model_version, question_type, temperature, 
    #                           max_tokens, **kwargs)
    # Filter out retry params (max_retries, retry_delay, retry_backoff) as they're handled 
    # by call_with_retry, not predictors
    predictor_kwargs = {
        "question": question,
        "model_version": model_version,
        "question_type": question_type,
        # Model parameters
        "temperature": prediction_config.get("temperature"),
        "max_tokens": prediction_config.get("max_tokens"),
        # Prompt configuration parameters (pass through to generate_prompt)
        "few_shots": prediction_config.get("few_shots", False),
        "show_cot": prediction_config.get("show_cot", False),
        "discipline": prediction_config.get("discipline"),
        "verbose_instructions": prediction_config.get("verbose_instructions", False),
        "abstention": prediction_config.get("abstention", False),
        "verbalized_elicitation": prediction_config.get("verbalized_elicitation", False),
        "word_limit": prediction_config.get("word_limit"),
    }
    
    # Add type-specific fields
    if question_type == QuestionType.MCQ:
        choices = question_data.get("choices", {})
        predictor_kwargs.update({
            "choice1": choices.get("A"),
            "choice2": choices.get("B"),
            "choice3": choices.get("C"),
            "choice4": choices.get("D"),
        })
    elif question_type == QuestionType.COMP:
        predictor_kwargs["text"] = question_data.get("text", "")
    
    # Get prediction with retry logic
    prediction_result, response_time, success, attempts = call_with_retry(
        prediction_func,
        predictor_kwargs,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        provider=provider,
        model_version=model_version
    )
    
    #print(f"🔍 DEBUG: call_with_retry() returned - success={success}, attempts={attempts}, response_time={response_time:.2f}s")
    
    # Extract text and tokens from prediction result
    # Predictor functions return either a dict {"text": ..., "tokens": ...} or a string
    if isinstance(prediction_result, dict):
        raw_text = prediction_result.get("text", prediction_result)
        token_data = prediction_result.get("tokens")
        # Extract only completion_tokens (output tokens)
        output_tokens = token_data.get("completion_tokens") if token_data else None
    else:
        raw_text = prediction_result if prediction_result else ""
        output_tokens = None
    
    return {
        "model": model_version,
        "raw_response": raw_text,
        "completion_tokens": output_tokens,
        "response_time": response_time,
        "success": success,
        "generated_at": datetime.now().isoformat()
    }


def process_question_with_all_models(
    predictor_functions: List[Tuple[str, str, Callable]],
    question_data: Dict[str, Any],
    question_type: QuestionType,
    question_idx: int,
    total_questions: int,
    enable_parallel_processing: bool,
    max_parallel_workers: int,
    stats_tracker: ModelStatsTracker,
    progress_tracker: Optional[ProgressTracker] = None,
    **prediction_config) -> List[Dict[str, Any]]:
    """
    Process a single question with all models (parallel or sequential).
    
    This is a DRY helper function to avoid code duplication between batch and non-batch modes.
    
    Args:
        predictor_functions: List of (provider, model_name, prediction_func) tuples
        question_data: Question data dictionary
        question_type: Question type enum
        question_idx: Current question index (for logging)
        total_questions: Total number of questions (for logging)
        enable_parallel_processing: Enable parallel execution
        max_parallel_workers: Number of parallel workers
        stats_tracker: Stats tracker instance
        progress_tracker: Progress tracker instance (optional)
        **prediction_config: Prediction parameters (discipline, few_shots, show_cot,
                            verbose_instructions, abstention, verbalized_elicitation,
                            temperature, max_tokens, word_limit, max_retries, 
                            retry_delay, retry_backoff, parallel_timeout)
    
    Returns:
        List of prediction dictionaries (one per model)
    """
    q_predictions = []
    
    # Update progress tracker
    if progress_tracker:
        progress_tracker.update_question_progress(question_idx, total_questions)
    
    if enable_parallel_processing:
        # PARALLEL MODE: Process models concurrently
        #print(f"🔍 DEBUG: Starting parallel execution with {max_parallel_workers} workers for {len(predictor_functions)} models")
        
        # Create a list to store futures with their metadata
        future_to_model = {}
        
        with ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
            # Submit all prediction tasks
            for predictor_idx, (provider, model_name, prediction_func) in enumerate(predictor_functions):
                #print(f"🔍 DEBUG: Submitting task for [{provider}/{model_name}]...")
                future = executor.submit(
                    process_question,
                    prediction_func,
                    model_name,
                    question_data,
                    question_type,
                    provider,
                    **prediction_config
                )
                # Store future with metadata to preserve order
                future_to_model[future] = (predictor_idx, provider, model_name)
            
            # Collect results in order (important!)
            results = {}
            completed_count = 0
            parallel_timeout = prediction_config.get("parallel_timeout", 120)
            for future in as_completed(future_to_model):
                predictor_idx, provider, model_name = future_to_model[future]
                try:
                    pred = future.result(timeout=parallel_timeout)
                    completed_count += 1
                    if progress_tracker:
                        progress_tracker.update_model_progress(
                            completed_count, len(predictor_functions), 
                            f"{provider}/{model_name}", pred.get('success', False)
                        )
                    results[predictor_idx] = (pred, provider, model_name)
                except Exception as e:
                    print(f"[WARNING] Exception in thread for [{provider}/{model_name}]: {type(e).__name__}: {str(e)}")
                    # Create a failed prediction entry
                    results[predictor_idx] = ({
                        "model": model_name,
                        "raw_response": "",
                        "completion_tokens": None,
                        "response_time": 0.0,
                        "success": False,
                        "generated_at": datetime.now().isoformat()
                    }, provider, model_name)
        
        # Process results in original order
        for predictor_idx in sorted(results.keys()):
            pred, provider, model_name = results[predictor_idx]
            
            # Record stats (done sequentially to avoid race conditions)
            stats_tracker.record_prediction(
                provider=provider,
                model=model_name,
                success=pred.get("success", False),
                response_time=pred.get("response_time"),
                tokens=pred.get("completion_tokens")
            )
            
            # Post-process and map MCQ responses
            apply_mcq_post_processing(pred, question_type)
            
            q_predictions.append(pred)
        
        #print(f"🔍 DEBUG: All models completed for question {question_idx}, total predictions: {len(q_predictions)}")
    
    else:
        # SEQUENTIAL MODE: Process models one at a time
        #print(f"🔍 DEBUG: Processing models sequentially (parallel processing disabled)")
        
        for model_idx, (provider, model_name, prediction_func) in enumerate(predictor_functions, 1):
            #print(f"🔍 DEBUG: Calling predictor for [{provider}/{model_name}]...")
            pred = process_question(
                prediction_func,
                model_name,
                question_data,
                question_type,
                provider,
                **prediction_config
            )
            
            # Update progress tracker
            if progress_tracker:
                progress_tracker.update_model_progress(
                    model_idx, len(predictor_functions),
                    f"{provider}/{model_name}", pred.get('success', False)
                )
            
            # Record stats
            stats_tracker.record_prediction(
                provider=provider,
                model=model_name,
                success=pred.get("success", False),
                response_time=pred.get("response_time"),
                tokens=pred.get("completion_tokens")
            )
            
            # Post-process and map MCQ responses
            apply_mcq_post_processing(pred, question_type)
            
            q_predictions.append(pred)
    
    # Mark question as completed
    if progress_tracker:
        progress_tracker.finish_question(question_idx)
    
    return q_predictions


def run_prediction_pipeline(
    predictor_functions,
    file_path,
    question_type: QuestionType,
    output_path,
    model_dict: Dict[str, List[str]],
    batch_save_size: Optional[int] = None,
    checkpoint_enabled: bool = True,
    enable_parallel_processing: bool = True,
    max_parallel_workers: int = 3,
    progress_tracker: Optional[ProgressTracker] = None,
    **prediction_config) -> Dict[str, Any]:
    """
    Run prediction pipeline with batch processing, checkpointing, and stats tracking.
    
    Args:
        predictor_functions: List of (provider, model, function) tuples
        file_path: Path to input JSON file
        question_type: Question type enum
        output_path: Directory to save outputs
        model_dict: Dictionary of models by provider
        batch_save_size: Save every N questions (None = no batching)
        checkpoint_enabled: Enable checkpointing
        enable_parallel_processing: Enable parallel execution of models
        max_parallel_workers: Number of parallel workers (1=sequential, 2-3=balanced, 7+=max)
        progress_tracker: Progress tracker instance (optional)
        **prediction_config: Prediction parameters (few_shots, show_cot, 
                            verbose_instructions, abstention, verbalized_elicitation,
                            temperature, max_tokens, word_limit, max_retries, 
                            retry_delay, retry_backoff, discipline)
    
    Returns:
        Dictionary with metadata, questions, and stats
    """
    #print(f"🔍 DEBUG: run_prediction_pipeline() started")
    #print(f"🔍 DEBUG: File: {file_path.name}")
    #print(f"🔍 DEBUG: Question type: {question_type.value}")
    #print(f"🔍 DEBUG: Predictors: {len(predictor_functions)}")
    #print(f"🔍 DEBUG: Batch size: {batch_save_size}, Checkpoint: {checkpoint_enabled}")
    
    # Load data
    try:
        #print(f"🔍 DEBUG: Loading JSON file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        #print(f"🔍 DEBUG: JSON loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load file: {e}")
        #print(f"🔍 DEBUG: Exception: {type(e).__name__}: {str(e)}")
        return None
    
    # Handle different JSON structures - direct array or dict with metadata/questions
    if isinstance(data, list):
        # Direct array of questions (current format)
        metadata = {}
        questions = data
    else:
        # Dictionary with metadata and questions
        metadata = data.get("metadata", {})
        questions = data.get("questions", [])
    
    # Extract discipline from metadata and add to prediction_config
    discipline = metadata.get("discipline", "العلوم الشرعية")
    prediction_config = {**prediction_config, "discipline": discipline}
    
    #print(f"🔍 DEBUG: Metadata: {metadata}")
    #print(f"🔍 DEBUG: Discipline: {discipline}")
    #print(f"🔍 DEBUG: Number of questions: {len(questions)}")
    
    if not questions:
        print(f"  [WARNING] Warning: No questions found in file {file_path}")
        #print(f"🔍 DEBUG: Returning empty result")
        return {"metadata": metadata, "questions": [], "stats": None}
    
    # Initialize stats tracker
    #print(f"🔍 DEBUG: Initializing ModelStatsTracker...")
    stats_tracker = ModelStatsTracker()
    
    # Initialize checkpoint manager if enabled
    checkpoint_manager = None
    completed_batches = []
    if checkpoint_enabled and batch_save_size:
        #print(f"🔍 DEBUG: Checkpointing enabled, initializing CheckpointManager...")
        file_identifier = file_path.stem
        #print(f"🔍 DEBUG: File identifier: {file_identifier}")
        checkpoint_manager = CheckpointManager(output_path, file_identifier)
        checkpoint_data = checkpoint_manager.load_checkpoint()
        #print(f"🔍 DEBUG: Checkpoint data loaded: {checkpoint_data is not None}")
        
        if checkpoint_data:
            checkpoint_manager.print_resumption_info()
            completed_batches = checkpoint_manager.get_completed_batches()
            #print(f"🔍 DEBUG: Completed batches from checkpoint: {completed_batches}")
    
    # Process questions (with or without batching)
    total_questions = len(questions)
    total_models = len(predictor_functions)
    
    # Initialize progress tracking for this file
    if progress_tracker:
        progress_tracker.start_question_processing(total_questions, total_models)
    
    if batch_save_size and batch_save_size > 0:
        # Batch processing mode
        num_batches = (total_questions + batch_save_size - 1) // batch_save_size
        if not progress_tracker:  # Only print if no progress tracker (to avoid duplication)
            print(f"  [BATCH] Processing {total_questions} questions in {num_batches} batch(es) of {batch_save_size}")
        #print(f"🔍 DEBUG: Batch processing mode - {num_batches} batches")
        
        # Load existing saved file if resuming (to preserve already-processed batches)
        saved_file = output_path / "json" / f"{file_path.stem}.json"
        all_predictions = load_saved_questions(saved_file, completed_batches=completed_batches)
        
        for batch_num in range(1, num_batches + 1):
            #print(f"🔍 DEBUG: ========== BATCH {batch_num}/{num_batches} ==========")
            
            # Calculate batch indices
            start_idx = (batch_num - 1) * batch_save_size
            end_idx = min(start_idx + batch_save_size, total_questions)
            
            # Check if batch should be skipped (already completed)
            loaded, should_skip = load_or_skip_batch(
                batch_num, completed_batches, saved_file, start_idx, end_idx
            )
            
            if should_skip:
                print(f"  [SKIP] Skipping batch {batch_num}/{num_batches} (already completed)")
                replace_or_extend_list(all_predictions, loaded, start_idx)
                continue
            
            batch_questions = questions[start_idx:end_idx]
            
            print(f"  [BATCH] Processing batch {batch_num}/{num_batches} (questions {start_idx + 1}-{end_idx})")
            #print(f"🔍 DEBUG: Batch range: [{start_idx}:{end_idx}], Questions in batch: {len(batch_questions)}")
            
            batch_predictions = []
            for idx, question_data in enumerate(batch_questions, start=start_idx + 1):
                # Use helper function to process question with all models
                q_predictions = process_question_with_all_models(
                    predictor_functions,
                    question_data,
                    question_type,
                    idx,
                    total_questions,
                    enable_parallel_processing,
                    max_parallel_workers,
                    stats_tracker,
                    progress_tracker,
                    **prediction_config
                )
                
                question_structure = build_question_structure(question_data, question_type, q_predictions)
                batch_predictions.append(question_structure)
            
            # Replace or extend the predictions list (handles resumption properly)
            replace_or_extend_list(all_predictions, batch_predictions, start_idx)
            
            # Save batch (saves JSON only during batch processing, Excel saved at end)
            batch_data = {"metadata": metadata, "questions": all_predictions}  # All questions processed so far
            # Filter out retry params (not needed for saving metadata)
            save_config = {k: v for k, v in prediction_config.items() 
                          if k not in ["max_retries", "retry_delay", "retry_backoff"]}
            save_batch_predictions(
                batch_data, 
                output_path, 
                file_path.stem, 
                batch_num, 
                model_dict,
                question_type.value,
                save_excel=False,  # Disabled during batch processing for performance
                **save_config
            )
            # Update progress tracker
            if progress_tracker:
                progress_tracker.finish_batch(batch_num, num_batches, len(all_predictions), item_type="questions")
            else:
                print(f"  [OK] Batch {batch_num}/{num_batches} saved (cumulative: {len(all_predictions)} questions)")
            
            # Update checkpoint
            if checkpoint_manager:
                # Only append if not already in list (avoid duplicates)
                if batch_num not in completed_batches:
                    completed_batches.append(batch_num)
                checkpoint_manager.save_checkpoint(completed_batches, num_batches)
    else:
        # Non-batch processing mode
        if not progress_tracker:  # Only print if no progress tracker
            print(f"  Processing {total_questions} questions...")
        for idx, question_data in enumerate(questions, 1):
            # Use helper function to process question with all models
            q_predictions = process_question_with_all_models(
                predictor_functions,
                question_data,
                question_type,
                idx,
                total_questions,
                enable_parallel_processing,
                max_parallel_workers,
                stats_tracker,
                progress_tracker,
                **prediction_config
            )
            
            question_structure = build_question_structure(question_data, question_type, q_predictions)
            all_predictions.append(question_structure)
    
    # Clear checkpoint after completion
    if checkpoint_manager:
        checkpoint_manager.clear_checkpoint()
        print("  [CLEAR] Checkpoint cleared")
    
    return {
        "metadata": metadata,
        "questions": all_predictions,
        "stats": stats_tracker
    }


def main():
    """Main prediction pipeline entry point."""

    start_time = time.time()
    
    # Phase 1: Initialization & Configuration
    config, initialized = initialize_pipeline("config.yaml")
    
    if not config or not initialized:
        print("\n[ERROR] Pipeline initialization failed. Exiting.")
        #print(f"🔍 DEBUG: Exiting main() due to initialization failure")
        return

    # Phase 2: Discovery
    print("\n[DEBUG] ========== PHASE 2: DISCOVERY ==========")
    #print(f"🔍 DEBUG: Getting all JSON files from config...")
    all_files = config.get_all_json_files()
    #print(f"🔍 DEBUG: Found {len(all_files)} JSON files to process")
    
    #print(f"🔍 DEBUG: Getting prediction models dictionary...")
    model_dict = config.models.prediction_models
    #print(f"🔍 DEBUG: Model dictionary has {len(model_dict)} providers")
    # for provider, models in model_dict.items():
    #     #print(f"🔍 DEBUG:   - {provider}: {models}")
    #     pass
    
    #print(f"🔍 DEBUG: Calling get_predictor_functions()...")
    predictor_functions = get_predictor_functions(model_dict)
    #print(f"🔍 DEBUG: Got {len(predictor_functions)} predictor functions")
    
    # Get prediction parameters from config
    #print(f"🔍 DEBUG: Getting prediction parameters from config...")
    pred_params = config.prediction_config
    #print(f"🔍 DEBUG: Prediction params type: {type(pred_params)}")
    
    print("\n" + "=" * 80)
    print("PHASE 2: PREDICTION PARAMETERS")
    print("=" * 80)
    print(f"Few-shot examples: {pred_params.few_shots}")
    print(f"Chain-of-thought: {pred_params.show_cot}")
    print(f"Verbose instructions: {pred_params.verbose_instructions}")
    print(f"Abstention: {pred_params.abstention}")
    print(f"Temperature: {pred_params.temperature}")
    print(f"Max tokens: {pred_params.max_tokens}")
    print(f"Word limit: {pred_params.word_limit}")
    print(f"Batch save size: {pred_params.batch_save_size}")
    print(f"Checkpointing: {pred_params.checkpoint_enabled}")
    print(f"Max retries: {pred_params.max_retries}")
    print(f"Retry delay: {pred_params.retry_delay}s")
    print(f"Retry backoff: {pred_params.retry_backoff}x")
    print(f"Parallel processing: {pred_params.enable_parallel_processing}")
    print(f"Max parallel workers: {pred_params.max_parallel_workers}")
    print("=" * 80)
    #print(f"🔍 DEBUG: Prediction parameters displayed")

    # Phase 3: Processing
    print("\n" + "=" * 80)
    print("PHASE 3: PROCESSING")
    print("=" * 80)
    #print(f"🔍 DEBUG: Initializing processing phase...")
    
    stats = {"success": 0, "failed": 0, "total_questions": 0}
    #print(f"🔍 DEBUG: Stats tracker initialized: {stats}")
    
    global_stats_tracker = ModelStatsTracker()
    #print(f"🔍 DEBUG: Global ModelStatsTracker created")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    
    # Load global checkpoint to skip already completed files
    global_checkpoint_file = config.paths.get_output_path("predictions") / ".global_checkpoint.json"
    completed_files = set()
    if global_checkpoint_file.exists():
        try:
            with open(global_checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                completed_files = set(checkpoint_data.get("completed_files", []))
                if completed_files:
                    print(f"\n  [RESUME] Resuming from global checkpoint: {len(completed_files)} files already completed")
        except Exception as e:
            print(f"[WARNING] Warning: Failed to load global checkpoint: {e}")
    
    total_files = len(all_files)
    #print(f"🔍 DEBUG: Total files to process: {total_files}")
    
    for file_idx, (question_type, difficulty, file_path) in enumerate(all_files, 1):
        # Check if file is already completed
        file_identifier = f"{question_type.value}_{difficulty.value}_{file_path.stem}"
        if file_identifier in completed_files:
            print(f"\n[SKIP] Skipping file {file_idx}/{total_files} (already completed): {file_path.name}")
            stats["success"] += 1  # Count as success since it was completed before
            continue
        
        # Start tracking this file
        progress_tracker.start_file_processing(file_idx, total_files, file_path.name)
        print(f"   Type: {question_type.value}, Difficulty: {difficulty.value}")
        
        try:
            #print(f"🔍 DEBUG: Getting output path for predictions...")
            output_path = config.paths.get_output_path("predictions", question_type, difficulty)
            #print(f"🔍 DEBUG: Output path: {output_path}")
            
            #print(f"🔍 DEBUG: Calling run_prediction_pipeline()...")
            #print(f"🔍 DEBUG: Parameters - {len(predictor_functions)} predictors, batch_size={pred_params.batch_save_size}")
            
            # Build prediction config dict once (includes all prediction parameters)
            prediction_config = asdict(pred_params)
            # Remove fields that are passed as explicit keyword arguments
            # These are pipeline control parameters, not prediction parameters
            for key in ["batch_save_size", "checkpoint_enabled", 
                        "enable_parallel_processing", "max_parallel_workers"]:
                prediction_config.pop(key, None)
            
            predictions = run_prediction_pipeline(
                predictor_functions,
                file_path,
                question_type,
                output_path,
                model_dict,
                batch_save_size=pred_params.batch_save_size,
                checkpoint_enabled=pred_params.checkpoint_enabled,
                enable_parallel_processing=pred_params.enable_parallel_processing,
                max_parallel_workers=pred_params.max_parallel_workers,
                progress_tracker=progress_tracker,
                **prediction_config
            )
            
            #print(f"🔍 DEBUG: run_prediction_pipeline() returned: {predictions is not None}")
            if predictions:
                #print(f"🔍 DEBUG: Predictions has {len(predictions.get('questions', []))} questions")
                # Save final predictions (filter out retry params - not needed for saving metadata)
                save_config = {k: v for k, v in prediction_config.items() 
                              if k not in ["max_retries", "retry_delay", "retry_backoff"]}
                save_predictions(
                    predictions,
                    output_path,
                    question_type,
                    model_dict,
                    save_json=True,
                    save_excel=True,
                    filename=file_path.stem,
                    **save_config
                )
                
                # Merge stats
                file_stats_tracker = predictions.get("stats")
                if file_stats_tracker:
                    global_stats_tracker.merge(file_stats_tracker)
                
                stats["success"] += 1
                questions_processed = len(predictions.get("questions", []))
                stats["total_questions"] += questions_processed
                progress_tracker.finish_file(True, questions_processed)
                
                # Mark file as completed in global checkpoint
                completed_files.add(file_identifier)
                # Use atomic write to prevent data loss on interruption
                global_checkpoint_data = {
                    "completed_files": list(completed_files),
                    "last_updated": datetime.now().isoformat()
                }
                success = atomic_write_json(global_checkpoint_file, global_checkpoint_data, backup=True)
                if not success:
                    print(f"[ERROR] Failed to save global checkpoint atomically - data may be lost!")
            else:
                stats["failed"] += 1
                progress_tracker.finish_file(False, 0)
        except Exception as e:
            stats["failed"] += 1
            progress_tracker.finish_file(False, 0)
            print(f"   [ERROR] Error processing file: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Phase 4: Summary
    print("\n[DEBUG] ========== PHASE 4: SUMMARY ==========")
    total_time = time.time() - start_time
    #print(f"🔍 DEBUG: Total execution time: {total_time:.2f}s")
    
    # Get progress summary
    progress_summary = progress_tracker.get_summary()
    
    print("\n" + "=" * 80)
    print("PHASE 4: SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Failed to process: {stats['failed']}")
    print(f"Total questions processed: {stats['total_questions']}")
    print(f"Total processing time: {progress_summary['elapsed_str']} ({total_time / 60:.2f} minutes)")
    print("=" * 80)
    #print(f"🔍 DEBUG: Stats breakdown: {stats}")
    
    # Save and print model performance statistics
    if stats['total_questions'] > 0:
        #print(f"🔍 DEBUG: Saving global stats tracker...")
        stats_output_path = config.paths.get_output_path("statistics")
        #print(f"🔍 DEBUG: Stats output path: {stats_output_path}")
        stats_file = global_stats_tracker.save_stats(stats_output_path)
        print(f"\n[STATS] Model performance statistics saved to: {stats_file}")
        #print(f"🔍 DEBUG: Stats file created: {stats_file}")
        global_stats_tracker.print_summary()
    else:
        #print(f"🔍 DEBUG: No questions processed, skipping stats save")
        pass
    
    print("\n[OK] Pipeline completed successfully!")
    print("=" * 80)
    #print(f"🔍 DEBUG: ========== MAIN() FUNCTION COMPLETED ==========")



if __name__ == "__main__":
    main()