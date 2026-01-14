"""
Configuration Loader for Benchmarking Pipeline

This module provides a comprehensive configuration management system for the benchmarking
pipeline. It handles:
- Loading and validating YAML configuration files
- Managing input/output path configurations with enum integration
- Creating and validating directory structures
- Extracting metadata from file paths
- Supporting both prediction and evaluation pipelines with context-aware operations

Pipeline Contexts:
    The module supports two distinct pipeline contexts:
    
    1. Prediction Pipeline (main.py):
       - Input: Questions from data/question_type/difficulty_dir/json/
       - Output: Predictions to results/predictions/question_type/difficulty_dir/json|excel/
       - Validates: Question input paths exist
       - Creates: Prediction output directories
    
    2. Evaluation Pipeline (LLM_as_a_judge.py):
       - Input: Predictions from results/predictions/question_type/difficulty_dir/json/
       - Output: Evaluations to results/evaluations/question_type/difficulty_dir/json|excel/
       - Validates: Prediction input paths exist
       - Creates: Evaluation output directories

Directory Structure:
    Input:  base/data/question_type/difficulty_dir/json/
    Output: base/results/predictions|evaluations/question_type/difficulty_dir/json|excel/

Key Classes:
    - ConfigPaths: Manages path operations and validations with unified get_path() interface
    - DataConfig: Container for question type and difficulty combinations
    - ModelConfig: Container for prediction and evaluation model configurations
    - PredictionConfig: Container for prediction pipeline parameters
    - BatchProcessingConfig: Container for batch processing parameters
    - ConfigLoader: Main configuration loader with validation and path management

Method Integration:
    Methods are designed to build on each other seamlessly:
    - get_questions_path() → get_path() → validates existence
    - get_output_path() → get_path() → may call ensure_output_dirs()
    - get_json_files() → get_path() → filters valid, non-empty files
    - get_all_json_files() → get_json_files() → collects all combinations
    - validate_paths_exist() → get_path() → context-aware validation
    - extract_*_from_path() → _find_output_key_index() → path parsing

Usage Examples:
    # Basic usage
    config = load_config("config.yaml")
    
    # Prediction pipeline
    files = config.get_all_json_files(data_type="questions")
    output_path = config.paths.get_path("predictions", QuestionType.COMP, Difficulty.BEGINNER)
    config.validate_paths_exist(check_input=True, input_data_type="questions")
    
    # Evaluation pipeline
    pred_files = config.get_all_json_files(data_type="predictions")
    eval_path = config.paths.get_path("evaluations", QuestionType.COMP, Difficulty.BEGINNER)
    config.validate_paths_exist(check_input=True, input_data_type="predictions")
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from constants import QuestionType, Difficulty, DIFFICULTY_DIRS


@dataclass
class ConfigPaths:
    """
    Container for all path configurations with enum integration.
    
    Manages input and output paths for the benchmarking pipeline, providing unified
    path operations that work seamlessly with QuestionType and Difficulty enums.
    
    Attributes:
        input: Dictionary mapping QuestionType values to base input paths
        output: Dictionary mapping output type names to base output paths
        base_dir: Base directory for all paths (resolved absolute path)
    
    Path Structure:
        - Input:  base_dir / input[question_type] / difficulty_dir / json/
        - Output: base_dir / output[type] / question_type / difficulty_dir / json|excel/
    """
    input: Dict[str, Path]  # Maps QuestionType to base input path
    output: Dict[str, Path]  # Maps output type to output path
    base_dir: Path
    
    def get_questions_path(self, question_type: QuestionType, difficulty: Difficulty) -> Path:
        """
        Get and validate the path to input questions data.
        
        This is a convenience method that wraps get_path for questions data type.
        It ensures the path exists before returning it.
        
        Args:
            question_type: Type of question (MCQ, COMP, KNOW)
            difficulty: Difficulty level (BEGINNER, INTERMEDIATE, ADVANCED)
        
        Returns:
            Path to the questions directory (without /json suffix)
        
        Raises:
            ValueError: If question_type is not configured
            FileNotFoundError: If the path does not exist
        
        Example:
            path = config_paths.get_questions_path(QuestionType.COMP, Difficulty.BEGINNER)
            # Returns: base_dir/data/COMP/A_beginner
        """
        # Use get_path for consistency - it handles validation
        return self.get_path("questions", question_type, difficulty)

    def ensure_output_dirs(self, data_config: Any, output_type: str = "predictions"):
        """
        Create output directories for all combinations in data_config.
        
        Creates the directory structure for predictions or evaluations:
        base_dir/output[output_type]/question_type/difficulty_dir/{json,excel}/
        
        For structured output types (predictions, evaluations), creates subdirectories
        for each question_type/difficulty combination. For other types, only creates
        the base directory.
        
        Args:
            data_config: DataConfig instance with get_combinations() method
            output_type: Type of output ('predictions', 'evaluations', etc.)
        
        Raises:
            TypeError: If data_config doesn't have get_combinations() method
            ValueError: If output_type is not configured
            RuntimeError: If directory creation fails
        
        Example:
            config_paths.ensure_output_dirs(prediction_data, "predictions")
            # Creates: results/predictions/COMP/A_beginner/{json,excel}/
        """
        if not hasattr(data_config, 'get_combinations'):
            raise TypeError("data_config must have get_combinations() method")
        
        # Get base output path (without question_type/difficulty for base level)
        if output_type not in self.output:
            raise ValueError(
                f"Unknown output type: {output_type}. "
                f"Available: {list(self.output.keys())}"
            )
        
        base_output = self.base_dir / self.output[output_type]
        base_output.mkdir(parents=True, exist_ok=True)
        
        # Validate base directory was created/exists
        if not base_output.exists():
            raise RuntimeError(f"Failed to create base output directory: {base_output}")
            
        # For predictions and evaluations, create subdirectories
        if output_type in ('predictions', 'evaluations'):
            for combo in data_config.get_combinations():
                combo_dir = base_output / combo['question_type'].value / combo['difficulty_dir']
                combo_dir.mkdir(parents=True, exist_ok=True)
                (combo_dir / "json").mkdir(parents=True, exist_ok=True)
                (combo_dir / "excel").mkdir(parents=True, exist_ok=True)

                # Validate all directories were created
                if not combo_dir.exists():
                    raise RuntimeError(f"Failed to create combination directory: {combo_dir}")
                if not (combo_dir / "json").exists():
                    raise RuntimeError(f"Failed to create JSON directory: {combo_dir / 'json'}")
                if not (combo_dir / "excel").exists():
                    raise RuntimeError(f"Failed to create Excel directory: {combo_dir / 'excel'}")

    def get_output_path(self, output_type: str = "predictions", question_type: Optional[QuestionType] = None, difficulty: Optional[Difficulty] = None, data_config: Optional[Any] = None) -> Path:
        """
        Get output path for a specific output type, optionally with question_type and difficulty.
        This is a convenience wrapper around get_path for backward compatibility.
        
        Args:
            output_type: Type of output ('predictions', 'evaluations', etc.)
            question_type: Optional question type for building subdirectory paths
            difficulty: Optional difficulty level for building subdirectory paths
            data_config: Optional config for creating directories if they don't exist
        """
        return self.get_path(output_type, question_type, difficulty, data_config)
    
    def get_path(self, data_type: str = "predictions", question_type: Optional[QuestionType] = None, difficulty: Optional[Difficulty] = None, data_config: Optional[Any] = None) -> Path:
        """
        Unified path getter that handles all data types with optional subdirectory building.
        
        This is the main path resolution method. It:
        - Validates paths exist (for questions and specific output subdirectories)
        - Creates directories if data_config is provided and path doesn't exist
        - Builds paths incrementally based on provided parameters
        
        Path Building Logic:
        - Questions: base_dir/input[question_type]/[difficulty_dir/]
        - Output types: base_dir/output[type]/[question_type/][difficulty_dir/]
        
        Args:
            data_type: Type of data ('questions', 'predictions', 'evaluations', 'statistics', etc.)
            question_type: Optional question type for building subdirectory paths
            difficulty: Optional difficulty level for building subdirectory paths
            data_config: Optional DataConfig for creating directories if they don't exist
        
        Returns:
            Path object (validated to exist for questions and specific output paths)
        
        Raises:
            ValueError: If data_type is unknown or required parameters are missing
            FileNotFoundError: If path doesn't exist and validation is required
        
        Examples:
            # Get base predictions path
            path = get_path("predictions")
            
            # Get specific prediction path (creates if data_config provided)
            path = get_path("predictions", QuestionType.COMP, Difficulty.BEGINNER, data_config)
            
            # Get questions path (requires question_type, difficulty optional)
            path = get_path("questions", QuestionType.COMP, Difficulty.BEGINNER)
        """
        if data_type == "questions":
            if question_type is None:
                raise ValueError("question_type is required for data_type='questions'")
            
            # Validate question_type is configured
            if question_type.value not in self.input:
                raise ValueError(
                    f"Unknown question type: {question_type.value}. "
                    f"Available: {list(self.input.keys())}"
                )
            
            base_input = self.base_dir / self.input[question_type.value]
            
            # Build path incrementally based on what's provided
            if difficulty:
                difficulty_dir = DIFFICULTY_DIRS[difficulty]
                path = base_input / difficulty_dir
                
                # Validate full path exists if both are provided
                if not path.exists():
                    raise FileNotFoundError(
                        f"Questions path does not exist: {path} "
                        f"(question_type: {question_type.value}, difficulty: {difficulty.value})"
                    )
                return path
            else:
                # Return base path for question type (without difficulty)
                if not base_input.exists():
                    raise FileNotFoundError(
                        f"Questions base path does not exist: {base_input} "
                        f"(question_type: {question_type.value})"
                    )
                return base_input
        elif data_type in self.output:
            base_output = self.base_dir / self.output[data_type]
            
            # Build full path if question_type and/or difficulty are provided
            if question_type:
                base_output = base_output / question_type.value
            if difficulty:
                difficulty_dir = DIFFICULTY_DIRS[difficulty]
                base_output = base_output / difficulty_dir
            
            # Create directories if data_config provided and path doesn't exist
            # This allows lazy directory creation when needed
            if data_config is not None and not base_output.exists():
                if data_type in ('predictions', 'evaluations'):
                    # Use ensure_output_dirs for structured output types (creates all subdirs)
                    self.ensure_output_dirs(data_config, data_type)
                else:
                    # For other output types (statistics, visualizations), just create base directory
                    base_output.mkdir(parents=True, exist_ok=True)
            
            # Validate existence only if both question_type and difficulty are provided
            # This indicates a specific subdirectory that should already exist
            if question_type and difficulty and not base_output.exists():
                raise FileNotFoundError(
                    f"Output path does not exist: {base_output} "
                    f"(data_type: {data_type}, question_type: {question_type.value}, "
                    f"difficulty: {difficulty.value})"
                )
            
            return base_output
        else:
            raise ValueError(f"Unknown data type: {data_type}. Available: 'questions', {list(self.output.keys())}")

    def get_json_files(self, question_type: QuestionType, difficulty: Difficulty, data_type: str = "questions", data_config: Optional[Any] = None) -> List[Path]:
        """
        Get all valid, non-empty JSON files from the specified directory.
        
        Uses get_path to resolve the directory, then filters JSON files to ensure
        they exist, are regular files, and are not empty. Returns sorted list.
        
        Args:
            question_type: Question type for the data
            difficulty: Difficulty level for the data
            data_type: Type of data ('questions', 'predictions', 'evaluations')
            data_config: Optional DataConfig for creating directories if they don't exist
        
        Returns:
            Sorted list of Path objects to valid, non-empty JSON files
        
        Raises:
            ValueError: If data_type is invalid
            FileNotFoundError: If JSON directory doesn't exist
            NotADirectoryError: If path is not a directory
        """
        if data_type not in ("questions", "predictions", "evaluations"):
            raise ValueError(
                f"Invalid data type: {data_type}. "
                f"Must be one of: 'questions', 'predictions', 'evaluations'"
            )

        # Use get_path for consistency - it handles both questions and output types
        # Note: get_path for questions returns path without /json, so we add it
        base_path = self.get_path(data_type, question_type, difficulty, data_config)
        json_dir = base_path / "json"
        if not json_dir.exists():
            raise FileNotFoundError(
                f"JSON directory not found: {data_type} for {question_type.value} {difficulty.value} -> {json_dir}"
            )
        if not json_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {json_dir}")
        
        # Get all JSON files and validate they exist and are not empty
        # Filtering ensures we only return valid, processable files
        json_files = []
        for file_path in sorted(json_dir.glob("*.json")):
            # Defensive checks (glob should only return existing files, but be safe)
            if not file_path.exists():
                continue  # Skip if file doesn't exist
            
            if not file_path.is_file():
                continue  # Skip if not a regular file (e.g., symlinks, directories)
            
            # Skip empty files (they're not useful for processing)
            if file_path.stat().st_size == 0:
                continue
            
            json_files.append(file_path)
        
        return json_files
    
    def get_output_path_with_task_folder(
        self, 
        output_type: str, 
        task_folder: str, 
        difficulty: Difficulty
    ) -> Path:
        """
        Get output path using explicit task folder name (mirrors input structure).
        
        This is useful when you want to mirror the exact folder structure from predictions
        to evaluations, including suffixes like "COMP_single_shot_with_abstention".
        
        Args:
            output_type: "predictions" or "evaluations"
            task_folder: Task folder name (e.g., "COMP_single_shot_with_abstention")
            difficulty: Difficulty enum
            
        Returns:
            Path to output directory
            
        Example:
            >>> config.paths.get_output_path_with_task_folder(
            ...     "evaluations", 
            ...     "COMP_single_shot_with_abstention", 
            ...     Difficulty.BEGINNER
            ... )
            Path('results/evaluations/COMP_single_shot_with_abstention/A_beginner')
        """
        # Get output path from output dict (which contains Path objects)
        if output_type not in self.output:
            raise ValueError(
                f"Unknown output type: {output_type}. "
                f"Available: {list(self.output.keys())}"
            )
        
        output_path = self.output[output_type]
        diff_dir = DIFFICULTY_DIRS.get(difficulty, difficulty.value)
        
        return self.base_dir / output_path / task_folder / diff_dir


@dataclass
class DataConfig:
    """
    Container for data processing configuration with enum integration.
    
    Defines which question types and difficulty levels to process, and provides
    methods to generate all combinations for batch processing.
    
    Attributes:
        question_types: List of QuestionType enums to process
        difficulty_levels: List of Difficulty enums to process
    """
    question_types: List[QuestionType]
    difficulty_levels: List[Difficulty]
    
    def get_combinations(self) -> List[Dict[str, Any]]:
        """
        Generate all combinations of question_type and difficulty.
        
        Creates a cartesian product of question_types × difficulty_levels,
        returning dictionaries with both enum objects and string values for convenience.
        
        Returns:
            List of dictionaries, each containing:
            - 'question_type': QuestionType enum
            - 'difficulty': Difficulty enum
            - 'question_type_str': String value
            - 'difficulty_str': String value
            - 'difficulty_dir': Directory name (from DIFFICULTY_DIRS mapping)
        """
        return [
            {
                'question_type': qt,
                'difficulty': dl,
                'question_type_str': qt.value,
                'difficulty_str': dl.value,
                'difficulty_dir': DIFFICULTY_DIRS[dl]
            }
            for qt in self.question_types
            for dl in self.difficulty_levels
        ]


@dataclass
class ModelConfig:
    """
    Container for model configurations.
    
    Stores model names organized by provider for both prediction and evaluation pipelines.
    
    Attributes:
        prediction_models: Dict mapping provider names to lists of model names
        evaluation_models: Dict mapping provider names to lists of judge model names
    """
    prediction_models: Dict[str, List[str]]
    evaluation_models: Dict[str, List[str]]
    

    def get_prediction_models_by_provider(self, provider: str) -> List[str]:
        """
        Get prediction models for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
        
        Returns:
            List of model names for the provider, or empty list if not found
        """
        return self.prediction_models.get(provider, [])
    
    def get_evaluation_models_by_provider(self, provider: str) -> List[str]:
        """
        Get evaluation (judge) models for a specific provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
        
        Returns:
            List of judge model names for the provider, or empty list if not found
        """
        return self.evaluation_models.get(provider, [])


@dataclass
class PredictionConfig:
    """
    Container for prediction configuration parameters.
    
    All parameters for controlling the prediction pipeline behavior, including
    prompt engineering, model parameters, retry logic, and parallel processing.
    
    Attributes:
        few_shots: Whether to include few-shot examples in prompts
        show_cot: Whether to request chain-of-thought reasoning
        verbose_instructions: Whether to include detailed instructions
        abstention: Whether to allow models to abstain from answering
        verbalized_elicitation: Whether to request confidence scores
        temperature: Model temperature (None = provider default)
        max_tokens: Maximum tokens for response (None = provider default)
        word_limit: Maximum words for response (None = no limit)
        batch_save_size: Save checkpoint every N questions
        checkpoint_enabled: Whether to use checkpointing for resumability
        max_retries: Number of retry attempts on failure
        retry_delay: Initial retry delay in seconds
        retry_backoff: Exponential backoff multiplier
        enable_parallel_processing: Whether to process models in parallel
        max_parallel_workers: Number of parallel workers (1=sequential, 2-3=balanced, 7+=max)
        parallel_timeout: Timeout in seconds for parallel worker futures (default: 120)
        evaluation_mode: Evaluation mode for LLM-as-a-judge (None, "standard", or "granular")
    """
    few_shots: bool = True
    show_cot: bool = False
    verbose_instructions: bool = True
    abstention: bool = False
    verbalized_elicitation: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    word_limit: Optional[int] = None
    batch_save_size: int = 10  # Save every N questions (batch-only mode, always enabled)
    checkpoint_enabled: bool = True
    max_retries: int = 3  # Number of retry attempts
    retry_delay: float = 5.0  # Initial retry delay in seconds
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    enable_parallel_processing: bool = True  # Enable parallel execution of models
    max_parallel_workers: int = 3  # Number of parallel workers (1=sequential, 2-3=balanced, 7+=max)
    parallel_timeout: int = 120  # Timeout in seconds for parallel worker futures
    evaluation_mode: Optional[str] = None  # Evaluation mode: None (default), "standard" (v1), or "granular" (v2)


@dataclass
class BatchProcessingConfig:
    """
    Container for file batch processing configuration.
    
    Controls parallel file processing and checkpointing for batch operations.
    Currently used by run_all_targeted_predictions.py.
    
    Attributes:
        max_file_workers: Number of files to process simultaneously
        checkpoint_save_interval: Save checkpoint every N files processed
    """
    max_file_workers: int = 4  # Process N files simultaneously
    checkpoint_save_interval: int = 3  # Save checkpoint every N files


class ConfigLoader:
    """
    Main configuration loader for the benchmarking pipeline.
    
    Loads and validates YAML configuration files, providing typed access to all
    configuration sections. Integrates with constants.py to use QuestionType and
    Difficulty enums throughout.
    
    Pipeline Contexts:
        - Prediction Pipeline (main.py):
          * Input: questions from data/question_type/difficulty_dir/json/
          * Output: predictions to results/predictions/question_type/difficulty_dir/json|excel/
        
        - Evaluation Pipeline (LLM_as_a_judge.py):
          * Input: predictions from results/predictions/question_type/difficulty_dir/json/
          * Output: evaluations to results/evaluations/question_type/difficulty_dir/json|excel/
    
    Directory Structure Handled:
        Input:  base_dir/data/question_type/difficulty_dir/json/
        Output: base_dir/results/predictions|evaluations/question_type/difficulty_dir/json|excel/
    
    Key Features:
        - Lazy loading: Configuration is parsed only when load() is called
        - Type safety: All paths and enums are validated
        - Path management: Unified path operations via ConfigPaths
        - Pipeline support: Separate configs for prediction and evaluation pipelines
        - Context-aware: Methods adapt to prediction vs evaluation pipeline contexts
    
    Usage:
        # Load configuration
        config = ConfigLoader("config.yaml").load()
        
        # Prediction pipeline
        files = config.get_all_json_files(data_type="questions")
        config.ensure_output_dirs()  # Create prediction directories
        config.validate_paths_exist(check_input=True, input_data_type="questions")
        
        # Evaluation pipeline
        pred_files = config.get_all_json_files(data_type="predictions")
        config.ensure_evaluation_dirs()  # Create evaluation directories
        config.validate_paths_exist(check_input=True, input_data_type="predictions")
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._raw_config: Optional[Dict[str, Any]] = None
        self._paths: Optional[ConfigPaths] = None
        self._prediction_data: Optional[DataConfig] = None
        self._evaluation_data: Optional[DataConfig] = None
        self._models: Optional[ModelConfig] = None
        self._prediction_config: Optional[PredictionConfig] = None
        self._batch_processing_config: Optional[BatchProcessingConfig] = None
    
    def load(self) -> 'ConfigLoader':
        """
        Load and validate the configuration file.
        
        Reads the YAML file, validates its structure, and parses all sections
        into typed objects. Must be called before accessing any configuration properties.
        
        Returns:
            Self for method chaining
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is empty or invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
        
        if not self._raw_config:
            raise ValueError("Config file is empty or invalid")
        
        # Validate and parse all sections into typed objects
        self._validate_and_parse()
        
        return self
    
    def _validate_and_parse(self):
        """
        Validate config structure and parse into typed objects.
        
        This method performs comprehensive validation and converts raw YAML data
        into strongly-typed Python objects with enum integration.
        """
        # Parse base directory (resolve to absolute path for consistency)
        base_dir_str = self._raw_config.get('base_dir', '.')
        base_dir = Path(base_dir_str).resolve()
        
        # Parse and validate paths section
        paths_config = self._raw_config.get('paths', {})
        if not isinstance(paths_config, dict):
            raise ValueError("'paths' must be a dictionary.")
        
        input_paths_raw = paths_config.get('input', {})
        output_paths_raw = paths_config.get('output', {})
        
        if not input_paths_raw or not output_paths_raw:
            raise ValueError("'paths.input' and 'paths.output' must be defined")
        
        # Convert input paths to Path objects and validate question types
        # Input paths must match QuestionType enum values
        input_paths = {}
        for qtype_str, path_str in input_paths_raw.items():
            try:
                # Validate it's a valid QuestionType enum value
                QuestionType(qtype_str)
                input_paths[qtype_str] = Path(path_str)
            except ValueError:
                raise ValueError(
                    f"Invalid question type '{qtype_str}'. "
                    f"Must be one of: {[qt.value for qt in QuestionType]}"
                )
        
        # Convert output paths to Path objects (no enum validation needed)
        output_paths = {}
        for otype, path_str in output_paths_raw.items():
            output_paths[otype] = Path(path_str)
        
        self._paths = ConfigPaths(
            input=input_paths,
            output=output_paths,
            base_dir=base_dir
        )
        
        # Parse data configurations with enum conversion
        pred_data = self._raw_config.get('data_to_predict', {})
        eval_data = self._raw_config.get('data_to_evaluate', {})
        
        # Convert string lists to enum lists
        pred_qtypes = self._parse_question_types(pred_data.get('question_types', []))
        pred_diffs = self._parse_difficulties(pred_data.get('difficulty_levels', []))
        
        eval_qtypes = self._parse_question_types(eval_data.get('question_types', []))
        eval_diffs = self._parse_difficulties(eval_data.get('difficulty_levels', []))
        
        self._prediction_data = DataConfig(
            question_types=pred_qtypes,
            difficulty_levels=pred_diffs
        )
        
        self._evaluation_data = DataConfig(
            question_types=eval_qtypes,
            difficulty_levels=eval_diffs
        )
        
        # Parse model configurations
        pred_models = self._raw_config.get('prediction_models', {})
        eval_models = self._raw_config.get('evaluation_models', {})
        
        if not pred_models and not eval_models:
            raise ValueError("At least one of 'prediction_models' or 'evaluation_models' must be defined")
        
        self._models = ModelConfig(
            prediction_models=pred_models,
            evaluation_models=eval_models
        )
        
        # Parse prediction parameters
        pred_params = self._raw_config.get('prediction_parameters', {})
        self._prediction_config = PredictionConfig(
            few_shots=pred_params.get('few_shots', True),
            show_cot=pred_params.get('show_cot', False),
            verbose_instructions=pred_params.get('verbose_instructions', True),
            abstention=pred_params.get('abstention', False),
            verbalized_elicitation=pred_params.get('verbalized_elicitation', True),
            temperature=pred_params.get('temperature'),
            max_tokens=pred_params.get('max_tokens'),
            word_limit=pred_params.get('word_limit'),
            batch_save_size=pred_params.get('batch_save_size', 10),
            checkpoint_enabled=pred_params.get('checkpoint_enabled', True),
            max_retries=pred_params.get('max_retries', 3),
            retry_delay=pred_params.get('retry_delay', 5.0),
            retry_backoff=pred_params.get('retry_backoff', 2.0),
            enable_parallel_processing=pred_params.get('enable_parallel_processing', True),
            max_parallel_workers=pred_params.get('max_parallel_workers', 3),
            parallel_timeout=pred_params.get('parallel_timeout', 120),
            evaluation_mode=pred_params.get('evaluation_mode', None)
        )
        
        # Parse batch processing parameters
        batch_params = self._raw_config.get('batch_processing', {})
        self._batch_processing_config = BatchProcessingConfig(
            max_file_workers=batch_params.get('max_file_workers', 4),
            checkpoint_save_interval=batch_params.get('checkpoint_save_interval', 3)
        )
    
    def _parse_question_types(self, qtypes: List[str]) -> List[QuestionType]:
        """Convert string list to QuestionType enum list."""
        result = []
        for qtype_str in qtypes:
            try:
                result.append(QuestionType(qtype_str))
            except ValueError:
                raise ValueError(
                    f"Invalid question type '{qtype_str}'. "
                    f"Must be one of: {[qt.value for qt in QuestionType]}"
                )
        return result
    
    def _parse_difficulties(self, diffs: List[str]) -> List[Difficulty]:
        """Convert string list to Difficulty enum list."""
        result = []
        for diff_str in diffs:
            try:
                result.append(Difficulty(diff_str))
            except ValueError:
                raise ValueError(
                    f"Invalid difficulty level '{diff_str}'. "
                    f"Must be one of: {[d.value for d in Difficulty]}"
                )
        return result
    
    @property
    def paths(self) -> ConfigPaths:
        """
        Get paths configuration.
        
        Provides access to ConfigPaths instance for path operations.
        All path-related operations should go through this property.
        """
        if self._paths is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._paths
    
    @property
    def prediction_data(self) -> DataConfig:
        """
        Get prediction data configuration.
        
        Returns DataConfig containing question types and difficulty levels
        configured for the prediction pipeline.
        """
        if self._prediction_data is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._prediction_data
    
    @property
    def evaluation_data(self) -> DataConfig:
        """
        Get evaluation data configuration.
        
        Returns DataConfig containing question types and difficulty levels
        configured for the evaluation pipeline.
        """
        if self._evaluation_data is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._evaluation_data
    
    @property
    def models(self) -> ModelConfig:
        """
        Get models configuration.
        
        Returns ModelConfig containing prediction and evaluation model
        configurations organized by provider.
        """
        if self._models is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._models
    
    @property
    def prediction_config(self) -> PredictionConfig:
        """
        Get prediction configuration parameters.
        
        Returns PredictionConfig containing all parameters for controlling
        the prediction pipeline behavior (prompt engineering, retry logic, etc.).
        """
        if self._prediction_config is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._prediction_config
    
    @property
    def batch_processing_config(self) -> BatchProcessingConfig:
        """
        Get batch processing configuration parameters.
        
        Returns BatchProcessingConfig containing parameters for parallel file
        processing and checkpointing in batch operations.
        """
        if self._batch_processing_config is None:
            raise RuntimeError("Config not loaded. Call load() first.")
        return self._batch_processing_config
    
    def ensure_output_dirs(self):
        """
        Create all output directories for configured prediction data combinations.
        
        Convenience method for prediction pipeline. Creates directory structure:
        results/predictions/question_type/difficulty_dir/{json,excel}/
        
        For evaluation directories, use ensure_evaluation_dirs() instead.
        """
        self.paths.ensure_output_dirs(self.prediction_data)
    
    def ensure_evaluation_dirs(self, target_types: Optional[set] = None):
        """
        Create evaluation output directories for configured evaluation data combinations.
        
        This is a convenience method that wraps ensure_output_dirs with filtering.
        If you don't need filtering, use: config.paths.ensure_output_dirs(config.evaluation_data, "evaluations")
        
        Args:
            target_types: Set of QuestionType enums to filter by (default: all types)
        """
        # Filter combinations if target_types specified
        if target_types:
            filtered_data = DataConfig(
                question_types=[qt for qt in self.evaluation_data.question_types if qt in target_types],
                difficulty_levels=self.evaluation_data.difficulty_levels
            )
            self.paths.ensure_output_dirs(filtered_data, "evaluations")
        else:
            self.paths.ensure_output_dirs(self.evaluation_data, "evaluations")
    
    def validate_paths_exist(self, check_input: bool = True, check_output: bool = False, input_data_type: str = "questions") -> Tuple[bool, List[str]]:
        """
        Validate that configured paths exist.
        
        Context-aware validation that adapts to different pipeline contexts:
        - Prediction pipeline (main.py): 
          * Input = questions (from data/)
          * Output = predictions (to results/predictions/)
          * Use: validate_paths_exist(check_input=True, input_data_type="questions")
        
        - Evaluation pipeline (LLM_as_a_judge.py):
          * Input = predictions (from results/predictions/)
          * Output = evaluations (to results/evaluations/)
          * Use: validate_paths_exist(check_input=True, input_data_type="predictions")
        
        Checks existence of input paths and/or output paths based on flags.
        For input paths, validates JSON directories for the specified data type.
        For output paths, validates base output directories (not subdirectories).
        
        Args:
            check_input: Whether to validate input paths (default: True)
            check_output: Whether to validate output paths (default: False)
            input_data_type: Type of input data to validate ('questions' or 'predictions')
                           - 'questions': For prediction pipeline (input = questions from data/)
                           - 'predictions': For evaluation pipeline (input = predictions from results/predictions/)
        
        Returns:
            Tuple of (all_exist: bool, missing_paths: List[str])
            - all_exist: True if all checked paths exist
            - missing_paths: List of descriptive error messages for missing paths
        
        Examples:
            # Prediction pipeline: validate question input paths
            all_exist, missing = config.validate_paths_exist(check_input=True, input_data_type="questions")
            
            # Evaluation pipeline: validate prediction input paths
            all_exist, missing = config.validate_paths_exist(check_input=True, input_data_type="predictions")
        """
        missing = []
        
        if check_input:
            # Select appropriate data config based on input_data_type
            if input_data_type == "questions":
                # Prediction pipeline: check question input paths
                data_config = self.prediction_data
                data_type = "questions"
                label_prefix = "Input (questions)"
            elif input_data_type == "predictions":
                # Evaluation pipeline: check prediction input paths
                data_config = self.prediction_data  # Predictions use prediction_data config
                data_type = "predictions"
                label_prefix = "Input (predictions)"
            else:
                raise ValueError(
                    f"Invalid input_data_type: {input_data_type}. "
                    f"Must be 'questions' or 'predictions'"
                )
            
            # Validate paths for all combinations in the selected data config
            for combo in data_config.get_combinations():
                # Use get_path for consistency - it validates existence
                input_path = self.paths.get_path(
                    data_type,
                    combo['question_type'],
                    combo['difficulty']
                )
                json_dir = input_path / "json"
                if not json_dir.exists():
                    missing.append(
                        f"{label_prefix} JSON dir: {combo['question_type'].value}/"
                        f"{combo['difficulty_dir']}/json/ -> {json_dir}"
                    )
        
        if check_output:
            for output_type in self.paths.output.keys():
                # Check base output path (without data_config to avoid creating)
                output_path = self.paths.get_path(output_type)
                if not output_path.exists():
                    missing.append(f"Output path for {output_type}: {output_path}")
        
        return (len(missing) == 0, missing)
    
    def get_all_json_files(self, data_type: str = "questions", data_config: Optional[DataConfig] = None) -> List[Tuple[QuestionType, Difficulty, Path]]:
        """
        Get all JSON files for all configured question_type/difficulty combinations.
        
        Iterates through all combinations in data_config and collects JSON files from
        each combination's directory. Uses get_json_files internally for consistency.
        
        Args:
            data_type: Type of data ('questions', 'predictions', 'evaluations')
            data_config: DataConfig to use. If None, automatically selects:
                        - evaluation_data for 'evaluations'
                        - prediction_data for 'questions' or 'predictions'
        
        Returns:
            List of tuples: (question_type, difficulty, file_path)
            Files are sorted by path and filtered to only include valid, non-empty JSON files.
        
        Example:
            # Get all question files
            files = config.get_all_json_files(data_type="questions")
            
            # Get all prediction files
            files = config.get_all_json_files(data_type="predictions")
        """
        # Auto-select appropriate data config if not provided
        # Evaluations use evaluation_data, everything else uses prediction_data
        if data_config is None:
            data_config = self.evaluation_data if data_type == "evaluations" else self.prediction_data
        
        # Collect files from all combinations
        all_files = []
        for combo in data_config.get_combinations():
            # Pass data_config only for output types (predictions/evaluations) to enable lazy creation
            files = self.paths.get_json_files(
                combo['question_type'],
                combo['difficulty'],
                data_type,
                data_config if data_type in ('predictions', 'evaluations') else None
            )
            for file_path in files:
                all_files.append((
                    combo['question_type'],
                    combo['difficulty'],
                    file_path
                ))
        return all_files
    
    def _find_output_key_index(self, file_path: Path) -> Optional[int]:
        """
        Find the index of 'predictions' or 'evaluations' in path parts.
        
        Helper method for path extraction. Finds where the output type folder appears
        in the path structure. Files are organized in folders, not by filename suffixes.
        
        Path structure: base/results/predictions|evaluations/question_type/difficulty_dir/json/file.json
        
        Args:
            file_path: Path to search in
        
        Returns:
            Index of 'predictions' or 'evaluations' in path.parts, or None if not found
        """
        parts = file_path.parts
        for key in ('predictions', 'evaluations'):
            if key in parts:
                return parts.index(key)
        return None
    
    def extract_question_type_from_path(self, file_path: Path) -> Optional[QuestionType]:
        """
        Extract question type from a prediction/evaluation file path.
        
        Path structure: base/results/predictions|evaluations/question_type/difficulty_dir/json/file.json
        Files are in folders named 'predictions' or 'evaluations', not in filenames.
        
        Handles paths like: results/predictions/COMP/A_beginner/json/file.json
        Also handles suffixed directory names like: COMP_single_shot → COMP
        
        Args:
            file_path: Path to prediction/evaluation file
            
        Returns:
            QuestionType enum or None if not found
        """
        idx = self._find_output_key_index(file_path)
        if idx is None:
            return None
        
        parts = file_path.parts
        # Question type is immediately after 'predictions' or 'evaluations'
        if idx + 1 < len(parts):
            q_type_str = parts[idx + 1]
            
            # Handle suffixes in directory names (e.g., COMP_single_shot → COMP)
            for qt in QuestionType:
                if q_type_str == qt.value or q_type_str.startswith(f"{qt.value}_"):
                    return qt
        
        return None
    
    def extract_difficulty_from_path(self, file_path: Path) -> Optional[Difficulty]:
        """
        Extract difficulty from a prediction/evaluation file path.
        
        Path structure: base/results/predictions|evaluations/question_type/difficulty_dir/json/file.json
        Difficulty directory is 2 levels after 'predictions' or 'evaluations'.
        
        Handles paths like: results/predictions/COMP/A_beginner/json/file.json
        
        Args:
            file_path: Path to prediction/evaluation file
            
        Returns:
            Difficulty enum or None if not found
        """
        idx = self._find_output_key_index(file_path)
        if idx is None:
            return None
        
        parts = file_path.parts
        # Difficulty directory is 2 levels after predictions/evaluations
        # Structure: .../predictions/question_type/difficulty_dir/json/file.json
        if idx + 2 < len(parts):
            difficulty_str = parts[idx + 2]
            
            # Match against DIFFICULTY_DIRS mapping
            for diff_enum, diff_dir in DIFFICULTY_DIRS.items():
                if diff_dir == difficulty_str:
                    return diff_enum
        
        return None
    
    def extract_task_folder_from_path(self, file_path: Path) -> Optional[str]:
        """
        Extract task folder name from prediction/evaluation file path.
        
        Path structure: base/results/predictions|evaluations/task_folder/difficulty_dir/json/file.json
        Returns the task folder name (e.g., "COMP" or "COMP_single_shot_with_abstention").
        
        This preserves the exact folder structure, including suffixes like "_single_shot_with_abstention",
        which is important for mirroring prediction folder structure in evaluations.
        
        Args:
            file_path: Path to prediction/evaluation file
            
        Returns:
            Task folder name or None if not found
            
        Example:
            >>> config.paths.extract_task_folder_from_path(
            ...     Path("results/predictions/COMP_single_shot_with_abstention/A_beginner/json/file.json")
            ... )
            "COMP_single_shot_with_abstention"
        """
        idx = self._find_output_key_index(file_path)
        if idx is None:
            return None
        
        parts = file_path.parts
        # Task folder is immediately after 'predictions' or 'evaluations'
        if idx + 1 < len(parts):
            return parts[idx + 1]
        
        return None
    
    def summary(self) -> str:
        """
        Get a human-readable summary of the entire configuration.
        
        Returns a formatted string with all configuration details including:
        - Base directory and paths
        - Data combinations for prediction and evaluation
        - Model counts by provider
        
        Returns:
            Multi-line formatted string summarizing the configuration
        """
        lines = [
            "=" * 60,
            "Configuration Summary",
            "=" * 60,
            f"Base Directory: {self.paths.base_dir}",
            "",
            "Input Paths:",
            *[f"  {qtype}: {path}" for qtype, path in self.paths.input.items()],
            "",
            "Output Paths:",
            *[f"  {otype}: {path}" for otype, path in self.paths.output.items()],
            "",
            f"Prediction Data: {len(self.prediction_data.get_combinations())} combinations",
            f"  Question Types: {[qt.value for qt in self.prediction_data.question_types]}",
            f"  Difficulty Levels: {[d.value for d in self.prediction_data.difficulty_levels]}",
            "",
            f"Evaluation Data: {len(self.evaluation_data.get_combinations())} combinations",
            f"  Question Types: {[qt.value for qt in self.evaluation_data.question_types]}",
            f"  Difficulty Levels: {[d.value for d in self.evaluation_data.difficulty_levels]}",
            "",
            f"Prediction Models: {sum(len(models) for models in self.models.prediction_models.values())} models",
            *[f"  {provider}: {len(models)} models" 
              for provider, models in self.models.prediction_models.items()],
            "",
            f"Evaluation Models: {sum(len(models) for models in self.models.evaluation_models.values())} models",
            *[f"  {provider}: {len(models)} models" 
              for provider, models in self.models.evaluation_models.items()],
            "=" * 60
        ]
        return "\n".join(lines)


# Convenience function for simple usage
def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Convenience function to load configuration file and return a ConfigLoader instance.
    
    This is the recommended entry point for loading configuration. It creates a
    ConfigLoader, loads the YAML file, validates it, and returns the ready-to-use instance.
    
    Args:
        config_path: Path to YAML configuration file (default: "config.yaml")
        
    Returns:
        ConfigLoader instance with configuration loaded and validated
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or missing required sections
    
    Example:
        config = load_config("config.yaml")
        files = config.get_all_json_files()
    """
    return ConfigLoader(config_path).load()


if __name__ == "__main__":
    # Simple test/demo of config loader
    config = load_config("config.yaml")
    print(config.summary())
    
    # Quick validation
    all_exist, missing = config.validate_paths_exist(check_input=True, check_output=False)
    if not all_exist:
        print(f"\n[WARNING] Missing input paths: {len(missing)}")
        for path in missing[:5]:
            print(f"  - {path}")
    else:
        print("All input paths exist")
    
    # ========================================================================
    # 1. ALL FETCHED INPUT DATA FILES
    # ========================================================================
    print("\n" + "=" * 80)
    print("1. ALL FETCHED INPUT DATA FILES")
    print("=" * 80)
    all_files = config.get_all_json_files()
    print(f"Total JSON files found: {len(all_files)}")
    
    # Process each file one by one
    for question_type, difficulty, file_path in all_files:
        print(f"    [FILE] {file_path.resolve()}")

    # if all_files:
    #     # Group by question_type and difficulty
    #     grouped = defaultdict(list)
    #     for qtype, difficulty, file_path in all_files:
    #         key = f"{qtype.value}/{difficulty.value}"
    #         grouped[key].append(file_path)
        
    #     for key in sorted(grouped.keys()):
    #         qtype_str, diff_str = key.split('/')
    #         print(f"\n  {qtype_str} - {diff_str}:")
    #         for file_path in sorted(grouped[key]):
    #             print(f"    📄 {file_path}")
    # else:
    #     print("  ⚠️  No JSON files found")
    
    # ========================================================================
    # 2. ALL OUTPUT PATHS (FULL/DETAILED)
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. ALL OUTPUT PATHS (FULL/DETAILED)")
    print("=" * 80)
    
    # Predictions output paths
    print("\n  Predictions:")
    for combo in config.prediction_data.get_combinations():
        output_path = config.paths.get_path(
            "predictions",
            combo['question_type'],
            combo['difficulty'],
            config.prediction_data
        )
        full_path = output_path.resolve()
        print(f"    [DIR] {full_path}")
    
    # Evaluations output paths
    print("\n  Evaluations:")
    for combo in config.evaluation_data.get_combinations():
        output_path = config.paths.get_path(
            "evaluations",
            combo['question_type'],
            combo['difficulty'],
            config.evaluation_data
        )
        full_path = output_path.resolve()
        print(f"    [DIR] {full_path}")
    
    # Statistics and visualizations (no subdirectories)
    print("\n  Statistics:")
    stat_path = config.paths.get_path("statistics")
    print(f"    [DIR] {stat_path.resolve()}")
    
    print("\n  Visualizations:")
    vis_path = config.paths.get_path("visualizations")
    print(f"    [DIR] {vis_path.resolve()}")
    
    # ========================================================================
    # 3. PREDICTION MODELS BY PROVIDER
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. PREDICTION MODELS BY PROVIDER")
    print("=" * 80)
    
    if config.models.prediction_models:
        print(config.models.prediction_models)
    #     for provider in sorted(config.models.prediction_models.keys()):
    #         models = config.models.get_prediction_models_by_provider(provider)
    #         print(f"\n  {provider}:")
    #         if models:
    #             for model in models:
    #                 print(f"    🤖 {model}")
    #         else:
    #             print("    (no models configured)")
    # else:
    #     print("  ⚠️  No prediction models configured")
    
    # ========================================================================
    # 4. EVALUATION MODELS BY PROVIDER
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. EVALUATION MODELS BY PROVIDER")
    print("=" * 80)
    
    if config.models.evaluation_models:
        print(config.models.evaluation_models)
    #     for provider in sorted(config.models.evaluation_models.keys()):
    #         models = config.models.get_evaluation_models_by_provider(provider)
    #         print(f"\n  {provider}:")
    #         if models:
    #             for model in models:
    #                 print(f"    🤖 {model}")
    #         else:
    #             print("    (no models configured)")
    # else:
    #     print("  ⚠️  No evaluation models configured")
    
    print("\n" + "=" * 80)