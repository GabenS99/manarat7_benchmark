"""
Post-processing module for cleaning responses from various models in prediction pipeline.

This module handles model-specific response patterns for predictions:
- qwen/qwen3-32b: Removes chain-of-thought reasoning (fallback if reasoning_effort fails)
- Islamic-RAG & Fanar: Extracts answer before confidence level (الثقة)
- allam-2-7b (Groq): Extracts answer before explanatory text (removes content after parenthesis)
- ALLaM-AI/ALLaM-7B-Instruct-preview (local): Extracts first Arabic letter before newlines/explanation
- Jais-2 (inceptionai/Jais-2-8B-Chat): Extracts MCQ choice from various patterns (أ), أ., explicit statements, etc.)
- Confidence extraction: Extracts confidence percentages from model responses

Note: This is prediction pipeline only - evaluation functions have been removed.
Only performs direct mapping - does not attempt to extract MCQ letters from inconsistent model responses.
"""
import re
from typing import Optional, Dict, Any, Tuple

# Add scripts directory to path for imports (remove later)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from constants import MCQ_choice_map_en, MCQ_choice_map_ar

# Note: 're' is only used in clean_fanar_response for splitting on 'الثقة'

# confidence extraction pattern (handles both single values and ranges)
# Matches confidence keywords followed by either single percentage or range (X% إلى Y%)
_CONFIDENCE_PATTERN = re.compile(
    r'(?:\*\*)?'  # Optional bold start
    r'(?:'
        # Arabic patterns: درجة ثقتي / درجة الثقة / ثقتي / الثقة
        r'(?:درجة\s+)?(?:ثقتي|الثقة)'
        r'|'
        # Special Arabic pattern: "بدرجة ثقة" (with confidence)
        r'بدرجة\s+ثقة\s+(?:تصل\s+إلى\s+)?'
        r'|'
        # English patterns: Confidence / Confidence Level
        r'[Cc]onfidence(?:\s+[Ll]evel)?'
    r')'
    r'(?:\*\*)?'  # Optional bold end
    r'.*?'  # Match any text between keyword and percentage(s)
    r'(?:'
        # Range pattern: X% إلى Y% (captures both values)
        r'([0-9]+(?:\.[0-9]+)?)\s*%\s+إلى\s+([0-9]+(?:\.[0-9]+)?)\s*%'
        r'|'
        # Single percentage pattern
        r'\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)\s*%'
    r')',
    re.IGNORECASE | re.DOTALL
)


def clean_qwen_response(raw_response: str) -> str:
    """
    Extract the actual answer from qwen/qwen3-32b response.
    
    The model outputs its thinking process followed by '\n\n\n' or '\n\n' and then the final answer.
    We extract everything after the last occurrence of '\n\n\n' or '\n\n', or the last line if no separator found.
    
    Args:
        raw_response: The raw response from qwen model
        
    Returns:
        The cleaned response (answer only)
        
    Example:
        Input: "\\nOkay, let's think...\\n\\n\\nب"
        Output: "ب"
        Input: "\\nOkay, let's think...\\n\\nج"
        Output: "ج"
    """
    if not raw_response:
        return ""
    
    # First, try splitting by triple newline (most common pattern)
    parts = raw_response.split('\n\n\n')
    if len(parts) > 1:
        # Return the last part, which should be the answer
        return parts[-1].strip()
    
    # If no triple newline, try double newline
    parts = raw_response.split('\n\n')
    if len(parts) > 1:
        # Return the last part, which should be the answer
        return parts[-1].strip()
    
    # If no separator found, try to extract the last line (might be the answer)
    lines = raw_response.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # If the last line is short (likely an answer like "ج" or "C"), return it
        if len(last_line) <= 5:
            return last_line
    
    # Fallback: return the original (trimmed)
    return raw_response.strip()


def clean_fanar_response(raw_response: str) -> str:
    """
    Extract the actual answer from Islamic-RAG or Fanar response.
    
    These models sometimes append confidence levels in the format:
    "answer\\nالثقة: 95%"
    
    We extract everything before "الثقة" or just return the response if no confidence is present.
    
    Args:
        raw_response: The raw response from Fanar/Islamic-RAG model
        
    Returns:
        The cleaned response (answer only, without confidence level)
        
    Example:
        Input: "أ\\nالثقة: 95%"
        Output: "أ"
    """
    if not raw_response:
        return ""
    
    # Check if the response contains confidence marker
    if 'الثقة' in raw_response:
        # Split by various possible delimiters before 'الثقة'
        # Could be '\nالثقة' or ' الثقة' or just 'الثقة'
        parts = re.split(r'\s*الثقة\s*[:：]', raw_response)
        return parts[0].strip()
    
    return raw_response.strip()


def clean_allam_response(raw_response: str) -> str:
    """
    Extract the actual answer from allam-2-7b response (Groq API version).
    
    The model sometimes provides answers with explanations after a parenthesis.
    Handles two patterns:
    1. "X) text" - Extract everything before the first closing parenthesis
    2. "X( text" - Extract everything before the opening parenthesis
    
    Args:
        raw_response: The raw response from allam-2-7b model (Groq)
    
    Returns:
        The cleaned response (answer without explanation after parenthesis)
    
    Example:
        Input: "ج) المطلوب العلة, وجوابه بالبرهان"
        Output: "ج"
    """
    if not raw_response:
        return ""
    
    # First, try to extract before closing parenthesis (most common pattern)
    if ')' in raw_response:
        parts = raw_response.split(')', 1)  # Split on first ')'
        cleaned = parts[0].strip()
    # Fallback: extract before opening parenthesis
    elif '(' in raw_response:
        parts = raw_response.split('(', 1)  # Split on first '('
        cleaned = parts[0].strip()
    else:
        cleaned = raw_response.strip()
    
    return cleaned.strip()


def clean_local_allam_response(raw_response: str) -> str:
    """
    Extract the actual answer from local ALLaM model response.
    
    The local ALLaM model (ALLaM-AI/ALLaM-7B-Instruct-preview) typically provides
    answers in the format: "X \n\n[explanation text]"
    where X is an Arabic letter (أ, ب, ج, د) followed by space and newlines.
    
    This function extracts just the first letter before any whitespace/newlines.
    
    Args:
        raw_response: The raw response from local ALLaM model
    
    Returns:
        The cleaned response (just the Arabic letter)
    
    Examples:
        Input: "أ \n\nالإجابة الصحيحة هي \"أ\". شروط الحديث..."
        Output: "أ"
        
        Input: "ب \n\nشروط قبول الحديث..."
        Output: "ب"
        
        Input: "د \n\nالتوضيح: الإجابة الصحيحة..."
        Output: "د"
    """
    if not raw_response:
        return ""
    
    # Strip leading/trailing whitespace
    cleaned = raw_response.strip()
    
    if not cleaned:
        return ""
    
    # Split by newlines to get first line
    first_line = cleaned.split('\n')[0].strip()
    
    # Split by spaces to get first token
    first_token = first_line.split()[0] if first_line.split() else first_line
    
    # Check if it's a single Arabic letter (أ, ب, ج, د, ه, و, etc.)
    # Arabic letters in the range for MCQ choices
    arabic_letters = ['أ', 'ب', 'ج', 'د', 'ه', 'و', 'ز', 'ح', 'ط', 'ي']
    
    if first_token in arabic_letters:
        return first_token
    
    # If first token is longer but starts with an Arabic letter, extract just the letter
    if first_token and first_token[0] in arabic_letters:
        return first_token[0]
    
    # Fallback: return first token if it's short (likely an answer)
    if len(first_token) <= 2:
        return first_token
    
    # Last resort: return the first character if it's an Arabic letter
    if cleaned and cleaned[0] in arabic_letters:
        return cleaned[0]
    
    return cleaned.strip()


def extract_confidence_from_response(raw_response: str) -> Tuple[str, Optional[float]]:
    """
    Extract confidence level from response and return cleaned version.
    
    Uses a simplified mega-regex pattern to catch all confidence variations.
    Handles Arabic and English patterns with markdown bold markers.
    
    Patterns matched:
    - Arabic: "ثقتي في هذه الإجابة [هي] X%", "درجة ثقتي X%", "درجة الثقة: X%"
    - Arabic with "بإجابتي": "درجة ثقتي بإجابتي: X%", "درجة ثقتي في إجابتي: X%"
    - Arabic with "بدرجة ثقة": "بدرجة ثقة تصل إلى X%", "بدرجة ثقة X%"
    - Arabic range patterns: "ثقتي بهذه الإجابة تتراوح بين X% إلى Y%" (extracts Y%)
    - Arabic with markdown bold: "**درجة الثقة: X%**", "**درجة الثقة**: X%", "**درجة الثقة:** X%"
    - English: "Confidence: X%", "Confidence level: X%"
    - English with markdown bold: "**Confidence: X%**", "**Confidence Level: X%**"
    
    Only matches patterns with confidence keywords to avoid false positives
    (e.g., won't match "نسبة الزكاة 2.5%" which is actual content).
    
    Args:
        raw_response: Raw response from model
        
    Returns:
        Tuple of (cleaned_response, confidence_percentage)
        - cleaned_response: Response with confidence statement removed
        - confidence_percentage: Float 0-100, or None if not found
        
    Example:
        >>> extract_confidence_from_response("الجواب هو كذا.\\n\\nثقتي في هذه الإجابة 90%.")
        ("الجواب هو كذا.", 90.0)
    """
    if not raw_response:
        return "", None
    
    confidence = None
    cleaned = raw_response
    
    # Use combined pattern to find ALL matches and take the LAST one
    matches = list(_CONFIDENCE_PATTERN.finditer(raw_response))
    
    if matches:
        # Sort by position (start index) to ensure we get the last one in the text
        matches.sort(key=lambda m: m.start())
        last_match = matches[-1]
        
        try:
            # Check if it's a range pattern (groups 1 and 2) or single pattern (group 3)
            if last_match.group(1) and last_match.group(2):
                # Range pattern: X% إلى Y% - use higher value
                lower_val = float(last_match.group(1))
                higher_val = float(last_match.group(2))
                if 0 <= lower_val <= 100 and 0 <= higher_val <= 100:
                    confidence = max(lower_val, higher_val)
            elif last_match.group(3):
                # Single percentage pattern
                conf_value = float(last_match.group(3))
                if 0 <= conf_value <= 100:
                    confidence = conf_value
            
            if confidence is not None:
                # Remove the confidence statement from response
                start_pos = last_match.start()
                end_pos = last_match.end()
                
                # Check for markdown bold markers (**) before and after the match
                check_start = max(0, start_pos - 3)
                for i in range(check_start, start_pos):
                    if i + 2 <= start_pos and raw_response[i:i+2] == '**':
                        start_pos = i
                        break
                
                check_end = min(len(raw_response), end_pos + 3)
                for i in range(end_pos, check_end):
                    if i + 2 <= len(raw_response) and raw_response[i:i+2] == '**':
                        end_pos = i + 2
                        break
                
                # Check if confidence is on its own line (preceded by newline or start of string)
                line_start = start_pos
                while line_start > 0 and raw_response[line_start - 1] not in '\n':
                    line_start -= 1
                
                # Check if there's only whitespace before the confidence on this line
                prefix = raw_response[line_start:start_pos].strip()
                if not prefix:  # Confidence is at start of line
                    # Remove from line start
                    start_pos = line_start
                
                # Check what comes after
                suffix_start = end_pos
                while suffix_start < len(raw_response) and raw_response[suffix_start] in ' \t':
                    suffix_start += 1
                
                # If followed by newline or period+newline, include them in removal
                if suffix_start < len(raw_response) and raw_response[suffix_start] in '.\n':
                    if raw_response[suffix_start] == '.':
                        suffix_start += 1
                    # Skip trailing whitespace and newlines
                    while suffix_start < len(raw_response) and raw_response[suffix_start] in ' \t\n':
                        suffix_start += 1
                    end_pos = suffix_start
                
                cleaned = raw_response[:start_pos] + raw_response[end_pos:]
            else:
                # Invalid confidence value (outside 0-100 range)
                pass  # Don't extract invalid confidence
        except (ValueError, IndexError):
            pass  # Failed to parse, skip
    
    # Clean up any trailing whitespace/newlines after removal
    cleaned = cleaned.strip()
    
    return cleaned, confidence


def clean_jais2_response(raw_response: str) -> str:
    """
    Extract MCQ choice from Jais-2 model response.
    
    Jais-2 patterns observed:
    1. Most common: Starts with Arabic choice + parenthesis: "أ)", "ب)", "ج)", "د)"
    2. Some use period: "أ.", "ب.", "ج.", "د."
    3. Some have explicit confirmation: "الخيار الصحيح هو **ب)**"
    4. Some mention multiple choices: "أ) ... و ج) ... و د) كل الأجوبة صحيحة"
    5. Some have English translation: "د)كل الأجوبة صحيحَة. (D)All answers are correct."
    6. Some are in English: "ج) contact the chain..."
    
    Strategy:
    1. Try to extract from start: "أ)", "ب)", "ج)", "د)" or "أ.", "ب.", "ج.", "د."
    2. Search for explicit statements like "الخيار الصحيح هو" followed by choice
    3. If multiple choices, take the last one (usually "د) كل الأجوبة صحيحة")
    4. Look for English letters in parentheses: "(A)", "(B)", "(C)", "(D)"
    5. Fallback: extract first Arabic letter (أ, ب, ج, د)
    
    Args:
        raw_response: The raw response from Jais-2 model
    
    Returns:
        The extracted choice (Arabic letter: أ, ب, ج, د) or empty string
    """
    if not raw_response:
        return ""
    
    cleaned = raw_response.strip()
    if not cleaned:
        return ""
    
    # Pattern 1: Check for multiple choices first (before single choice extraction)
    # If multiple choices are mentioned, prefer the last one (usually "د) كل الأجوبة صحيحة")
    all_choices = re.findall(r'([أبجدهوزحطي])[.)]', cleaned)
    if len(all_choices) > 1:
        # Return the last choice when multiple are mentioned
        return all_choices[-1]
    
    # Pattern 2: Starts with Arabic choice + parenthesis or period
    # Match: "أ)", "ب)", "ج)", "د)" or "أ.", "ب.", "ج.", "د."
    start_pattern = re.match(r'^([أبجدهوزحطي])[.)]', cleaned)
    if start_pattern:
        return start_pattern.group(1)
    
    # Pattern 3: Explicit confirmation statements
    # Match: "الخيار الصحيح هو **ب)**" or "الجواب هو أ)" or "الإجابة هي ج)"
    explicit_patterns = [
        r'الخيار\s+الصحيح\s+هو\s+\*?\*?([أبجدهوزحطي])\*?\*?[.)]',
        r'الجواب\s+(?:هو|هي)\s+([أبجدهوزحطي])[.)]',
        r'الإجابة\s+(?:هي|هو)\s+([أبجدهوزحطي])[.)]',
        r'correct\s+answer\s+is\s+([أبجدهوزحطي])[.)]',
        r'answer\s+is\s+([أبجدهوزحطي])[.)]',
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Pattern 4: English letters in parentheses: "(A)", "(B)", "(C)", "(D)"
    english_pattern = re.search(r'\(([A-D])\)', cleaned, re.IGNORECASE)
    if english_pattern:
        english_letter = english_pattern.group(1).upper()
        # Map English to Arabic
        return MCQ_choice_map_ar.get(english_letter, "")
    
    # Pattern 5: English letter with parenthesis: "A)", "B)", "C)", "D)" (if single choice)
    english_pattern2 = re.search(r'\b([A-D])[.)]', cleaned, re.IGNORECASE)
    if english_pattern2:
        english_letter = english_pattern2.group(1).upper()
        return MCQ_choice_map_ar.get(english_letter, "")
    
    # Pattern 6: Fallback - extract first Arabic letter if it's a valid choice
    first_char = cleaned[0] if cleaned else ""
    if first_char in ['أ', 'ب', 'ج', 'د', 'ه', 'و', 'ز', 'ح', 'ط', 'ي']:
        return first_char
    
    # Last resort: search for any Arabic choice letter in first 50 chars
    early_text = cleaned[:50]
    choice_match = re.search(r'([أبجدهوزحطي])', early_text)
    if choice_match:
        return choice_match.group(1)
    
    return ""


def _clean_response_by_model(raw_response: str, model_name: Optional[str]) -> str:
    """
    Clean response based on model-specific patterns (DRY helper).
    
    Args:
        raw_response: Raw response from model
        model_name: Model name (optional)
        
    Returns:
        Cleaned response string
    """
    if not raw_response:
        return ""
    
    if not model_name:
        return raw_response.strip()
    
    model_lower = model_name.lower()
    
    # Apply model-specific cleaning
    if model_name in ['Islamic-RAG', 'Fanar']:
        return clean_fanar_response(raw_response)
    elif 'allam-ai' in model_lower or 'local_allam' in model_lower or model_name.startswith('ALLaM-AI'):
        # Local ALLaM model (ALLaM-AI/ALLaM-7B-Instruct-preview)
        return clean_local_allam_response(raw_response)
    elif 'allam' in model_lower and 'allam-ai' not in model_lower:
        # Groq allam-2-7b (different pattern)
        return clean_allam_response(raw_response)
    elif 'qwen' in model_lower:
        return clean_qwen_response(raw_response)
    elif 'jais' in model_lower or 'jais-2' in model_lower or 'jais2' in model_lower:
        # Jais-2 model (inceptionai/Jais-2-8B-Chat)
        return clean_jais2_response(raw_response)
    else:
        return raw_response.strip()


def post_process_mcq_response(
    raw_response: str,
    model_name: Optional[str]
) -> Optional[str]:
    """
    Post-process MCQ responses based on model-specific patterns.
    
    This function applies model-specific cleaning and then attempts to map
    the cleaned response to a standard MCQ choice (A, B, C, D, etc.).
    
    Args:
        raw_response: The raw response from the model
        model_name: The name of the model
        
    Returns:
        The mapped choice (A, B, C, D) or None
        
    Example:
        >>> post_process_mcq_response("\\nThinking...\\n\\n\\nب", "qwen/qwen3-32b")
        'B'
    """
    if not raw_response:
        return None
    
    # Step 1: Clean the response based on model (using DRY helper)
    cleaned = _clean_response_by_model(raw_response, model_name)
    
    if not cleaned:
        return None
    
    # Step 2: Map cleaned response to English letter
    # Strip any remaining whitespace
    cleaned = cleaned.strip()
    
    # First, try direct mapping (if cleaned is already an Arabic letter)
    if cleaned in MCQ_choice_map_en:
        return MCQ_choice_map_en[cleaned]
    
    # If cleaned response is already an English letter, return it
    cleaned_upper = cleaned.upper()
    if cleaned_upper in ['A', 'B', 'C', 'D', 'E', 'F']:
        return cleaned_upper
    
    return None   


def apply_mcq_post_processing(pred: Dict[str, Any], question_type: str) -> None:
    """
    Apply MCQ post-processing to a prediction dict (in-place).
    
    This is a convenience wrapper around post_process_mcq_response() that:
    1. Checks if question type is MCQ
    2. Extracts raw_response and model from the prediction dict
    3. Applies post-processing and stores result in pred["mapped_response"]
    
    Args:
        pred: Prediction dictionary with 'raw_response' and 'model' keys
        question_type: Question type ("MCQ", "COMP", "KNOW")
    """
    # Accept both string and enum types
    qt_value = question_type.value if hasattr(question_type, 'value') else question_type
    
    if qt_value == "MCQ":
        raw_resp = pred.get("raw_response", "")
        if raw_resp:
            raw_resp = raw_resp.strip()
        model_name = pred.get("model", "")
        pred["mapped_response"] = post_process_mcq_response(raw_resp, model_name)


def post_process_mcq_response_detailed(
    raw_response: str,
    model_name: str,
    current_mapped_response: Optional[str] = None
) -> Dict[str, Any]:
    """
    Post-process MCQ responses with detailed results (for batch processing).
    
    This is used by the batch post-processing script to get statistics.
    
    Args:
        raw_response: The raw response from the model
        model_name: The name of the model
        current_mapped_response: The currently mapped response (if any)
        
    Returns:
        Dictionary with:
        - 'cleaned_response': The cleaned response text
        - 'mapped_response': The mapped choice (A, B, C, D) or None
        - 'changed': Boolean indicating if the mapping changed
    """
    # Get the mapped response
    mapped = post_process_mcq_response(raw_response, model_name)
    
    # Get cleaned response for reporting (using DRY helper)
    cleaned = _clean_response_by_model(raw_response, model_name)
    
    # Determine if the mapping changed
    changed = (mapped != current_mapped_response) and (mapped is not None)
    
    return {
        'cleaned_response': cleaned,
        'mapped_response': mapped,
        'changed': changed
    }


def post_process_prediction_file(predictions: Dict[str, Any], question_type: str = "MCQ") -> Dict[str, Any]:
    """
    Post-process an entire prediction file.
    
    Args:
        predictions: The predictions dictionary loaded from JSON
        question_type: The type of questions (currently only "MCQ" is supported)
        
    Returns:
        The updated predictions dictionary with post-processed responses
        
    Stats:
        Prints statistics about how many responses were successfully mapped
    """
    if question_type != "MCQ":
        print(f"[WARNING] Post-processing for {question_type} not implemented yet")
        return predictions
    
    total_predictions = 0
    previously_unmapped = 0
    newly_mapped = 0
    mapping_changed = 0
    
    # Process each question
    for question in predictions.get('questions', []):
        for pred in question.get('predictions', []):
            total_predictions += 1
            
            model_name = pred.get('model', '')
            raw_response = pred.get('raw_response', '')
            current_mapped = pred.get('mapped_response')
            
            # Track if it was previously unmapped
            was_unmapped = (current_mapped is None)
            if was_unmapped:
                previously_unmapped += 1
            
            # Post-process the response
            result = post_process_mcq_response(
                raw_response,
                model_name
            )
            
            # Update mapped response if we got a new mapping
            if result is not None:
                if current_mapped is None:
                    newly_mapped += 1
                elif current_mapped != result:
                    mapping_changed += 1
                
                pred['mapped_response'] = result
    
    # Print statistics
    print("\n[STATS] Post-processing Statistics:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Previously unmapped: {previously_unmapped}")
    print(f"   Newly mapped: {newly_mapped}")
    print(f"   Mappings changed: {mapping_changed}")
    print(f"   Still unmapped: {previously_unmapped - newly_mapped}")
    print(f"   Success rate: {newly_mapped/previously_unmapped*100:.1f}%" if previously_unmapped > 0 else "   No unmapped responses")
    
    return predictions







# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            'model': 'qwen/qwen3-32b',
            'raw': '\nOkay, thinking process here...\n\n\nب',
            'expected': 'B'
        },
        {
            'model': 'Islamic-RAG',
            'raw': 'أ\nالثقة: 95%',
            'expected': 'A'
        },
        {
            'model': 'Fanar',
            'raw': 'ب',
            'expected': 'B'
        },
        {
            'model': 'allam-2-7b',
            'raw': 'د) كل الأجوبة صحيحة',
            'expected': 'D'
        },
        {
            'model': 'allam-2-7b',
            'raw': 'ج) يستعمل الماء. \n\nإذا لم ينظف...',
            'expected': 'C'
        },
        {
            'model': 'allam-2-7b',
            'raw': 'أ',
            'expected': 'A'
        },
        {
            'model': 'allam-2-7b',
            'raw': 'ج) المطلوب العلة, وجوابه بالبرهان',
            'expected': 'C'
        },
        {
            'model': 'allam-2-7b',
            'raw': 'د) ما',
            'expected': 'D'
        },
        {
            'model': 'ALLaM-AI/ALLaM-7B-Instruct-preview',
            'raw': 'أ \n\nالإجابة الصحيحة هي "أ". شروط الحديث الصحي بحسب السخاوى تشمل اتصال السند',
            'expected': 'A'
        },
        {
            'model': 'ALLaM-AI/ALLaM-7B-Instruct-preview',
            'raw': 'ب \n\nشروط قبول الحديث تشمل الاتصال والشهر، بالإضافة إلى عدالة وضبط الراوي.',
            'expected': 'B'
        },
        {
            'model': 'ALLaM-AI/ALLaM-7B-Instruct-preview',
            'raw': 'د \n\nالتوضيح: الإجابة الصحيحة هي "د" لأن جميع الخيارات المذكورة صحيحة',
            'expected': 'D'
        },
        {
            'model': 'ALLaM-AI/ALLaM-7B-Instruct-preview',
            'raw': 'أ \n\n(المعل فيه علة خفيّة، والشّاذ فيه مُخالفة للثّ',
            'expected': 'A'
        },
        {
            'model': 'ALLaM-AI/ALLaM-7B-Instruct-preview',
            'raw': 'ب \n\n(رواية الحديث بلا سند)',
            'expected': 'B'
        }
    ]
    
    print("Running test cases...\n")
    for i, test in enumerate(test_cases, 1):
        result = post_process_mcq_response(test['raw'], test['model'])
        status = "PASS" if result == test['expected'] else "FAIL"
        print(f"{status} Test {i}: {test['model']}")
        print(f"   Raw: {test['raw'][:50]}...")
        print(f"   Mapped: {result} (expected: {test['expected']})")
        print()