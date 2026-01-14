from enum import Enum


# ============================================================================
# DATASET CONSTANTS
# ============================================================================

# QUESTION TYPE ENUMS
class QuestionType(str, Enum):
    """Question type enumerations."""
    MCQ = "MCQ"              # Multiple Choice Questions
    KNOW = "KNOW"            # Knowledge/Open-ended questions
    COMP = "COMP"            # Reading Comprehension questions
    # Note: MIRATH and RES may be added in future

# DIFFICULTY LEVEL ENUMS
class Difficulty(str, Enum):
    """Difficulty level enumerations."""
    BEGINNER = "Beginner"          # A_beginner / مبتدئ
    INTERMEDIATE = "Intermediate"  # B_intermediate / متوسط
    ADVANCED = "Advanced"          # C_advanced / متقدم
    
# DISCIPLINE ENUMS
class Discipline(str, Enum):
    """Islamic studies discipline enumerations."""
    # Core disciplines based on dataset organization
    DISCIPLINE_QURAN = "القرآن وعلومه"
    DISCIPLINE_HADITH = "الحديث وعلومه"
    DISCIPLINE_AQIDAH = "العقيدة الإسلامية"
    DISCIPLINE_TASAWWUF = "التصوف"
    DISCIPLINE_USUL = "أصول الفقه"
    DISCIPLINE_FIQH = "الفقه"
    DISCIPLINE_SIRAH = "السيرة النبوية"
    
    # Additional disciplines that may appear in sources later
    MIRATH = "الميراث"                    # Inheritance Law (if separate)

# data files structure
METADATA_FIELDS = ["id_source", "source_text", "difficulty_level", "total_questions", "discipline"] #COMP has an extra field "total_text_chunks"
MCQ_FIELDS = ["id_question", "question", "choices", "correct_answer"]
COMP_FIELDS = ["id_text_chunk", "text", "id_question", "question", "correct_answer", "has_quranic_verse", "has_hadith"]
KNOW_FIELDS = ["id_question", "question", "correct_answer", "has_quranic_verse", "has_hadith", "no_requested_items"]

# MCQ CHOICE MAP (English -> Arabic)
MCQ_choice_map_ar = {
    "A": "أ",
    "B": "ب", 
    "C": "ج", 
    "D": "د", 
    "E": "هـ", 
    "F": "و"
}

# MCQ CHOICE REVERSE MAP (Arabic -> English)
MCQ_choice_map_en = {v: k for k, v in MCQ_choice_map_ar.items()}

# ============================================================================
# CORRECT ANSWER WORD COUNT RANGES BY DIFFICULTY LEVEL
# ============================================================================
# Format: (min_words, max_words)
# Based on analysis of COMP and KNOW datasets

# COMP word count ranges
COMP_BEGINNER_WORD_COUNT_RANGE = (1, 52)
COMP_INTERMEDIATE_WORD_COUNT_RANGE = (1, 58)
COMP_ADVANCED_WORD_COUNT_RANGE = (20, 202)

# KNOW word count ranges
KNOW_BEGINNER_WORD_COUNT_RANGE = (1, 102)
KNOW_INTERMEDIATE_WORD_COUNT_RANGE = (1, 49)
KNOW_ADVANCED_WORD_COUNT_RANGE = (1, 99)

# ============================================================================
# FILE PATHS AND NAMING CONVENTIONS
# ============================================================================

# Default data directory structure
DATA_DIR_BASE = "data"
DATA_DIR_MCQ = f"{DATA_DIR_BASE}/MCQ"
DATA_DIR_COMP = f"{DATA_DIR_BASE}/COMP"
DATA_DIR_KNOW = f"{DATA_DIR_BASE}/KNOW"

# Difficulty subdirectories
DIFFICULTY_DIRS = {
    Difficulty.BEGINNER: "A_beginner",
    Difficulty.INTERMEDIATE: "B_intermediate",
    Difficulty.ADVANCED: "C_advanced",
}

# Output directories
OUTPUT_DIR_PREDICTIONS = "results/predictions"
OUTPUT_DIR_PREDICTIONS_MCQ = f"{OUTPUT_DIR_PREDICTIONS}/MCQ"
OUTPUT_DIR_PREDICTIONS_COMP = f"{OUTPUT_DIR_PREDICTIONS}/COMP"
OUTPUT_DIR_PREDICTIONS_KNOW = f"{OUTPUT_DIR_PREDICTIONS}/KNOW"

OUTPUT_DIR_EVALUATIONS = "results/evaluations"
OUTPUT_DIR_EVALUATIONS_MCQ = f"{OUTPUT_DIR_EVALUATIONS}/MCQ"
OUTPUT_DIR_EVALUATIONS_COMP = f"{OUTPUT_DIR_EVALUATIONS}/COMP"
OUTPUT_DIR_EVALUATIONS_KNOW = f"{OUTPUT_DIR_EVALUATIONS}/KNOW"

OUTPUT_DIR_STATISTICS = "results/statistics"
OUTPUT_DIR_VISUALIZATIONS = "results/visualizations"


# Prediction file naming: {source_id}_{source_text}_{timestamp}_prediction.json
PREDICTION_FILE_PATTERN = "{source_id}_{source_text}_prediction_{timestamp}{ext}"

# Evaluation file naming: {source_id}_{source_text}_{timestamp}_evaluation.json
EVALUATION_FILE_PATTERN = "{source_id}_{source_text}_evaluation_{timestamp}{ext}"

# ============================================================================
# EVALUATION SCHEMAS
# ============================================================================

# JSON Schema for LLM-as-a-Judge evaluation output (v1)
# Used with structured output API to ensure consistent, parseable responses
# v1 format: Single score (0-1) with explanation and chain-of-thought
EVALUATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "multipleOf": 0.1,
            "description": "Overall evaluation score between 0 and 1 (in 0.1 increments). 0 = complete failure, 1 = perfect answer. Based on correctness, relevance, and completeness."
        },
        "explanation": {
            "type": "string",
            "description": "Brief explanation in Arabic (ملاحظات) clarifying why the score was given. Should be concise and focused on the key reasons for the score."
        }
    },
    "required": ["score", "explanation"],
    "additionalProperties": False
}

# JSON Schema for LLM-as-a-Judge evaluation output (v2)
# Used with structured output API to ensure consistent, parseable responses
# v2 format: Separate scores (0-10) for correctness, relevance, and completeness (in that order)
EVALUATION_JSON_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "chain_of_thought": {
            "type": "string",
            "description": "Detailed Chain-of-Thought analysis in Arabic justifying each score. Should explain why each criterion received its score."
        },
        "correctness_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Score for correctness (الصحة): Are the facts in the student's answer correct and free from errors? 0 = complete failure, 10 = perfect."
        },
        "relevance_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Score for relevance (الصلة): Does the answer address the specific question without filler or off-topic content? Note: Do not penalize based on length alone. 0 = complete failure, 10 = perfect."
        },
        "completeness_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            "description": "Score for completeness (الشمول): Does the answer cover all aspects of the question and all key points from the correct answer? 0 = complete failure, 10 = perfect."
        }
    },
    "required": ["chain_of_thought", "correctness_score", "relevance_score", "completeness_score"],
    "additionalProperties": False
}

# ============================================================================
# MODEL NAME NORMALIZATION UTILITIES
# ============================================================================

def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name by removing provider prefix.
    
    Strips provider prefixes (openai/, gemini/, deepseek/, groq/, fanar/, local_allam/) to get
    the pure model name for API calls and consistent storage. Preserves model path prefixes
    that are part of the model identifier (qwen/, ALLaM-AI/, etc.).
    
    Args:
        model_name: Model name with or without provider prefix
        
    Returns:
        Normalized model name (unprefixed, except model path prefixes are preserved)
        
    Examples:
        normalize_model_name("openai/gpt-5.2") -> "gpt-5.2"
        normalize_model_name("groq/allam-2-7b") -> "allam-2-7b"
        normalize_model_name("groq/qwen/qwen3-32b") -> "qwen/qwen3-32b"
        normalize_model_name("qwen/qwen3-32b") -> "qwen/qwen3-32b"
        normalize_model_name("local_allam/ALLaM-AI/ALLaM-7B-Instruct-preview") -> "ALLaM-AI/ALLaM-7B-Instruct-preview"
        normalize_model_name("ALLaM-AI/ALLaM-7B-Instruct-preview") -> "ALLaM-AI/ALLaM-7B-Instruct-preview"
        normalize_model_name("gpt-5.2") -> "gpt-5.2"
    """
    if not model_name:
        return model_name
    
    # Known provider prefixes to strip
    provider_prefixes = [
        "openai/", "gemini/", "deepseek/", "groq/", "fanar/", 
        "mistral/", "local_allam/", "local_jais/"
    ]
    
    # Preserve model path prefixes (part of model identifier, not provider)
    model_path_prefixes = ["qwen/", "ALLaM-AI/"]
    
    # Check if it starts with a model path prefix (preserve it)
    for prefix in model_path_prefixes:
        if model_name.startswith(prefix):
            return model_name
    
    # Check if it starts with a provider prefix (strip it)
    for prefix in provider_prefixes:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    
    # If it contains "/" but doesn't match known prefixes, check if it's a model path
    # (e.g., "ALLaM-AI/ALLaM-7B-Instruct-preview" without provider prefix)
    if "/" in model_name:
        first_part = model_name.split("/", 1)[0]
        # If first part matches a model path prefix pattern, preserve the full path
        if first_part in ["ALLaM-AI", "qwen"]:
            return model_name
        # Otherwise, it might be an unknown provider prefix, strip it
        return model_name.split("/", 1)[1]
    
    return model_name


def normalize_question_type(question_type) -> str:
    """
    Normalize question_type to string value for consistent comparisons.
    
    Handles both QuestionType enum and string inputs.
    
    Args:
        question_type: QuestionType enum or string ("MCQ", "COMP", "KNOW")
        
    Returns:
        String value of question type
        
    Examples:
        normalize_question_type(QuestionType.MCQ) -> "MCQ"
        normalize_question_type("MCQ") -> "MCQ"
        normalize_question_type("mcq") -> "mcq"  # Preserves case if string
    """
    if isinstance(question_type, QuestionType):
        return question_type.value
    return str(question_type)


def extract_clean_base_name(filename: str) -> str:
    """
    Extract clean base name by removing all known prefixes, suffixes, and timestamps.
    
    This utility function ensures consistent file naming across the codebase by removing:
    - File extensions (.json, .yaml)
    - Prefixes (merged_, index_)
    - Suffixes (_targeted_evaluations, _evaluations, _predictions, _backup)
    - Timestamp patterns (_YYYYMMDD_HHMMSS)
    
    Used for generating consistent index file names and merged file names regardless
    of input file naming conventions.
    
    Args:
        filename: Input filename (with or without extension)
        
    Returns:
        Clean base name without prefixes, suffixes, or timestamps
        
    Examples:
        extract_clean_base_name("merged_ARBP-Beginner-COMP_20250102_123456.json") -> "ARBP-Beginner-COMP"
        extract_clean_base_name("index_ARBP-Beginner-COMP_evaluations.yaml") -> "ARBP-Beginner-COMP"
        extract_clean_base_name("ARBP-Beginner-COMP_targeted_evaluations.json") -> "ARBP-Beginner-COMP"
        extract_clean_base_name("ARBP-Beginner-COMP_targeted_evaluations_20250102_123456.json") -> "ARBP-Beginner-COMP"
    """
    import re
    
    name = filename
    
    # Remove extension if present
    if name.endswith('.json') or name.endswith('.yaml'):
        name = name.rsplit('.', 1)[0]
    
    # Remove prefixes (order matters - remove longest first)
    prefixes_to_remove = ['merged_', 'index_']
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break  # Only remove one prefix
    
    # Remove timestamp patterns first (before suffix removal)
    # This handles cases like {base}_targeted_evaluations_{timestamp}
    name = re.sub(r'_\d{8}_\d{6}$', '', name)
    
    # Remove suffixes (order matters - remove longest first)
    suffixes_to_remove = ['_targeted_evaluations', '_evaluations', '_predictions', '_backup']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break  # Only remove one suffix
    
    return name


def get_excluded_question_ids(question_type: QuestionType = None) -> list:
    """
    Get list of question IDs that should be excluded from benchmarking.
    
    These questions are used as few-shot examples in prompts, so including
    them in benchmarking would create data leakage.
    
    Args:
        question_type: Optional QuestionType filter. If provided, only returns
                      excluded IDs for that question type. If None, returns all.
    
    Returns:
        List of question ID strings to exclude from benchmarking.
    """
    excluded = []
    
    if question_type is None or question_type == QuestionType.MCQ:
        excluded.extend(MCQ_FEW_SHOT_EXAMPLE_IDS)
    
    if question_type is None or question_type == QuestionType.KNOW:
        excluded.extend(KNOW_FEW_SHOT_EXAMPLE_IDS)
    
    if question_type is None or question_type == QuestionType.COMP:
        excluded.extend(COMP_FEW_SHOT_EXAMPLE_IDS)
    
    return excluded