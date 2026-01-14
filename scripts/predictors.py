import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from typing import Optional
import json
import re
import os

import clients
from clients import ( client_openai, client_deepseek, client_groq, client_gemini, client_mistral)
from clients import get_fanar_credentials
from new_prompts import generate_prompt, generate_evaluation_prompt, generate_evaluation_prompt_granular
from constants import EVALUATION_JSON_SCHEMA, EVALUATION_JSON_SCHEMA_V2, normalize_question_type

# Local model inference (optional - only imported if used)
try:
    from jais_pred import get_response_jais
except ImportError:
    get_response_jais = None


# ========================================================================
# HELPER FUNCTIONS (DRY)
# ========================================================================



def _extract_tokens(response):
    """
    Extract token usage from API response.
    
    Args:
        response: API response object
        
    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens or None
    """
    if not response:
        return None
        
    if hasattr(response, 'usage_metadata'):  # Gemini
        return {
            "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', None),
            "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', None),
            "total_tokens": getattr(response.usage_metadata, 'total_token_count', None)
        }
    elif hasattr(response, 'usage'):  # OpenAI, DeepSeek, Groq, Mistral
        return {
            "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
            "completion_tokens": getattr(response.usage, 'completion_tokens', None),
            "total_tokens": getattr(response.usage, 'total_tokens', None)
        }
    return None


def _extract_finish_reason(response):
    """
    Extract finish_reason from Gemini API response to detect truncation.
    
    Args:
        response: Gemini API response object
        
    Returns:
        finish_reason string or None (e.g., "MAX_TOKENS", "STOP", "OTHER")
    """
    if not response:
        return None
    
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'finish_reason'):
            return candidate.finish_reason
    return None




# ========================================================================
# PREDICTOR FUNCTIONS
# ========================================================================

def get_prediction_gemini(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0,
    max_tokens: int = 1024,
    evaluation_mode: Optional[str] = None,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from Gemini model.
    
    Args:
        question: The question to answer
        model_version: Gemini model identifier (e.g., "gemini-2.5-pro")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens
        evaluation_mode: Enable structured JSON output for evaluation
        **kwargs: Additional parameters:
            - For standard mode: choice1-4, text, discipline, few_shots, 
              abstention, verbalized_elicitation, verbose_instructions, 
              show_cot, word_limit
            - For evaluation mode: text, discipline, correct_answer, 
              prediction, verbose_instructions
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    """
    if not client_gemini:
        print("Gemini client not initialized or missing API key.")
        return None
    
    question_type = normalize_question_type(question_type)
    
    # Generate prompt
    if evaluation_mode == "standard":
        prompt = generate_evaluation_prompt(
            question_type=question_type,
            question=question,
            **kwargs
        )
    elif evaluation_mode == "granular": 
        prompt = generate_evaluation_prompt_granular(
            question_type=question_type,
            question=question,
            **kwargs   
        )
    else:
        prompt = generate_prompt(
            question=question,
            question_type=question_type,
            **kwargs
        )

    # Check if prompt is empty (question validation failed)
    if not prompt:
        print("[ERROR] Prompt is empty")
        return None

    try:
        # Configure structured output for evaluation mode
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        
        # Note: Structured output not supported with older google.generativeai package
        # Evaluation mode will work with text-based prompts instead
        
        response = client_gemini.generate_content(
            model=model_version,
            contents=prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        # Extract finish_reason to detect truncation (proven solution for truncation issues)
        finish_reason = _extract_finish_reason(response)
        response_text = response.text.strip() if response.text else ""
        
        return {
            "text": response_text,
            "tokens": _extract_tokens(response),
            "finish_reason": finish_reason  # Include finish_reason to detect truncation
        }

    except TimeoutError:
        print(f"Timeout error with Gemini model ({model_version}): Request exceeded 300 seconds")
        raise  # Re-raise so retry logic can handle it with proper delays
    except Exception as e:
        print(f"Error with Gemini model ({model_version}): {e}")
        raise  # Re-raise the exception so retry logic can handle it with proper delays


def get_prediction_chatgpt(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0,
    max_tokens: int = 1024,
    evaluation_mode: Optional[str] = None,  # None, "standard", or "granular"
    **kwargs) -> Optional[dict]:
    """
    Get prediction from ChatGPT model.
    
    Args:
        question: The question to answer
        model_version: OpenAI model identifier (e.g., "gpt-4o", "gpt-4-turbo")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum output tokens
        evaluation_mode: Evaluation mode - None (normal), "standard" (v1), or "granular" (v2)
        **kwargs: Additional parameters:
            - For standard mode: choice1-4, text, discipline, few_shots, 
              abstention, verbalized_elicitation, verbose_instructions, 
              show_cot, word_limit
            - For evaluation mode: text, discipline, correct_answer, 
              prediction, verbose_instructions
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    """
    if not client_openai:
        print("OpenAI client not initialized or missing API key.")
        return None

    question_type = normalize_question_type(question_type)
    
    # Generate prompt (handle both boolean for backward compat and string)
    if evaluation_mode == True or evaluation_mode == "standard":
        prompt = generate_evaluation_prompt(
            question_type=question_type,
            question=question,
            **kwargs
        )
    elif evaluation_mode == "granular":
        prompt = generate_evaluation_prompt_granular(
            question_type=question_type,
            question=question,
            **kwargs
        )
    else:
        prompt = generate_prompt(
            question=question,
            question_type=question_type,
            **kwargs
        )
    
    # Check if prompt is empty (question validation failed)
    if not prompt:
        return None

    try:
        # Configure API parameters
        api_params = {
            "model": model_version,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens
        }
        
        # Add structured output for evaluation mode
        if evaluation_mode in ["standard", "granular"] or evaluation_mode == True:
            # Use appropriate schema based on mode
            schema = EVALUATION_JSON_SCHEMA_V2 if evaluation_mode == "granular" else EVALUATION_JSON_SCHEMA
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evaluation_response",
                    "schema": schema,
                    "strict": True
                }
            }
        
        response = client_openai.chat.completions.create(**api_params)
        
        return {
            "text": response.choices[0].message.content.strip(),
            "tokens": _extract_tokens(response)
        }

    except TimeoutError:
        print(f"Timeout error with ChatGPT model ({model_version}): Request exceeded 300 seconds")
        raise  # Re-raise so retry logic can handle it with proper delays
    except Exception as e:
        print(f"Error with ChatGPT model ({model_version}): {e}")
        raise  # Re-raise the exception so retry logic can handle it with proper delays


def get_prediction_deepseek(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    evaluation_mode: Optional[str] = None,  # None, "standard", or "granular"
    **kwargs) -> Optional[dict]:
    """
    Get prediction from DeepSeek model.
    
    Args:
        question: The question to answer
        model_version: DeepSeek model identifier (e.g., "deepseek-chat")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum output tokens
        evaluation_mode: Enable JSON output for evaluation (uses json_object mode)
        **kwargs: Additional parameters:
            - For standard mode: choice1-4, text, discipline, few_shots, 
              abstention, verbalized_elicitation, verbose_instructions, 
              show_cot, word_limit
            - For evaluation mode: text, discipline, correct_answer, 
              prediction, verbose_instructions
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    
    Note: DeepSeek supports JSON mode but not full JSON schema validation.
          For evaluation_mode, the prompt must specify the desired JSON structure.
    """
    if not client_deepseek:
        print("DeepSeek client not initialized or missing API key.")
        return None

    question_type = normalize_question_type(question_type)
    
    # Generate prompt (handle both boolean for backward compat and string)
    if evaluation_mode == True or evaluation_mode == "standard":
        prompt = generate_evaluation_prompt(
            question_type=question_type,
            question=question,
            **kwargs
        )
    elif evaluation_mode == "granular":
        prompt = generate_evaluation_prompt_granular(
            question_type=question_type,
            question=question,
            **kwargs
        )
    else:
        prompt = generate_prompt(
            question=question,
            question_type=question_type,
            **kwargs
        )
    
    # Check if prompt is empty (question validation failed)
    if not prompt:
        return None

    try:
        # Configure API parameters
        api_params = {
            "model": model_version,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
            "max_tokens": max_tokens
        }
        
        # Add JSON mode for evaluation (DeepSeek doesn't support full json_schema)
        if evaluation_mode in ["standard", "granular"] or evaluation_mode == True:
            api_params["response_format"] = {'type': 'json_object'}
        
        response = client_deepseek.chat.completions.create(**api_params)
        
        return {
            "text": response.choices[0].message.content.strip(),
            "tokens": _extract_tokens(response)
        }

    except TimeoutError:
        print(f"Timeout error with DeepSeek model ({model_version}): Request exceeded 300 seconds")
        raise  # Re-raise so retry logic can handle it with proper delays
    except Exception as e:
        print(f"Error with DeepSeek model ({model_version}): {e}")
        raise  # Re-raise the exception so retry logic can handle it with proper delays


def get_prediction_fanar(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0,
    max_tokens: int = 1024,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from Fanar Islamic-RAG model.
    
    Note: Fanar model embraces Quranic verses with XML tags (doesn't apply for hadith)
    E.g., <quran_start>...أَن لَّآ إِلَـٰهَ إِلَّا... [٨٧](https://quran.com/21/87)<quran_end>
    
    Args:
        question: The question to answer
        model_version: Fanar model identifier (e.g., "Islamic-RAG")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        **kwargs: Additional parameters (choice1-4, text, discipline, few_shots, 
            abstention, verbalized_elicitation, verbose_instructions, 
            show_cot, word_limit)
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    """
    question_type = normalize_question_type(question_type)
    url, headers = get_fanar_credentials()

    prompt = generate_prompt(
        question=question,
        question_type=question_type,
        **kwargs
    )
    
    # Check if prompt is empty (question validation failed)
    if not prompt:
        return None

    data = {
        "model": model_version,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        # Add timeout to prevent indefinite hanging (300 seconds total timeout)
        response = requests.post(url, json=data, headers=headers, timeout=300)
        response_json = response.json()
        
        if response.status_code == 200:
            tokens = None
            if "usage" in response_json:
                usage = response_json["usage"]
                tokens = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens")
                }
            
            return {
                "text": response_json["choices"][0]["message"]["content"].strip(),
                "tokens": tokens
            }
        else:
            print(f"Error with Fanar API ({response.status_code}): {response.text}")
            return None

    except requests.exceptions.Timeout:
        print(f"Timeout error with Fanar model ({model_version}): Request exceeded 300 seconds")
        raise  # Re-raise so retry logic can handle it with proper delays
    except Exception as e:
        print(f"Error with Fanar model ({model_version}): {str(e)}")
        raise


def get_prediction_groq(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from Groq model.
    
    Args:
        question: The question to answer
        model_version: Groq model identifier (e.g., "llama-3.3-70b", "qwen/qwen3-32b")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        **kwargs: Additional parameters (choice1-4, text, discipline, few_shots, 
            abstention, verbalized_elicitation, verbose_instructions, 
            show_cot, word_limit)
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    """
    if not client_groq:
        print("Groq client not initialized or missing API key.")
        return None

    question_type = normalize_question_type(question_type)
    
    prompt = generate_prompt(
        question=question,
        question_type=question_type,
        **kwargs
    )
    
    # Check if prompt is empty (question validation failed)
    if not prompt:
        return None

    # [NOTE] For qwen models, use reasoning_effort parameter to disable chain-of-thought thinking process
    # According to Groq API docs: reasoning_effort="none" disables reasoning tokens for Qwen 3 32B
    # See: https://console.groq.com/docs/reasoning
    api_params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model_version,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Disable reasoning for qwen models when show_cot is False
    # Support both "qwen/qwen3-32b" and "qwen3-32b" model name formats
    show_cot = kwargs.get("show_cot", False)
    if ("qwen" in model_version.lower() and "qwen3-32b" in model_version.lower()) and not show_cot:
        # api_params["reasoning_effort"] = "none"
        api_params["reasoning_format"] = "hidden"

    try:
        response = client_groq.chat.completions.create(**api_params)
        
        # [NOTE] With reasoning_effort="none", qwen models should not output reasoning tokens
        # If reasoning is still present (e.g., when show_cot=True), it may be wrapped in tags
        # For now, we just extract the content directly

        return {
            "text": response.choices[0].message.content.strip(),
            "tokens": _extract_tokens(response)
        }

    except TimeoutError:
        print(f"Timeout error with Groq model ({model_version}): Request exceeded 300 seconds")
        raise  # Re-raise so retry logic can handle it with proper delays
    except Exception as e:
        # Print error for logging, but re-raise so retry logic in main.py can handle it properly
        # This is especially important for rate limit errors (429) which need longer retry delays
        print(f"Error with Groq model ({model_version}): {e}")
        raise  # Re-raise the exception so retry logic can handle it with proper delays


def get_prediction_mistral(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from Mistral model.
    
    Args:
        question: The question to answer
        model_version: Mistral model identifier (e.g., "mistral-large-latest")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
        **kwargs: Additional parameters (choice1-4, text, discipline, few_shots, 
            abstention, verbalized_elicitation, verbose_instructions, 
            show_cot, word_limit)
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    """
    if not client_mistral:
        print("Mistral client not initialized or missing API key.")
        return None

    question_type = normalize_question_type(question_type)
    
    prompt = generate_prompt(
        question=question,
        question_type=question_type,
        **kwargs
    )
    
    if not prompt:
        return None

    try:
        response = client_mistral.chat.completions.create(
            model=model_version,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "text": response.choices[0].message.content.strip(),
            "tokens": _extract_tokens(response)
        }

    except TimeoutError:
        print(f"Timeout error with Mistral model ({model_version}): Request exceeded 300 seconds")
        raise
    except Exception as e:
        print(f"Error with Mistral model ({model_version}): {str(e)}")
        raise


def get_prediction_local_jais(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0.3,
    max_tokens: int = 1024,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from local JAIS model.
    
    Args:
        question: The question to answer
        model_version: JAIS model identifier (e.g., "jais_13b") - currently handled internally
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature - currently handled internally by jais_pred
        max_tokens: Maximum output tokens - currently handled internally by jais_pred
        **kwargs: Additional parameters (choice1-4, text, discipline, few_shots, 
            abstention, verbalized_elicitation, verbose_instructions, 
            show_cot, word_limit)
    
    Returns:
        Optional[dict]: Dictionary with response, or None on error
    
    Note: model_version, temperature, max_tokens accepted for interface consistency but
          currently managed internally by get_response_jais().
    """
    question_type = normalize_question_type(question_type)
    
    prompt = generate_prompt_jais(
        question=question,
        question_type=question_type,
        **kwargs
    )
    
    if not prompt:
        return None
    
    # Note: get_response_jais handles its own model/temperature/max_tokens configuration
    return get_response_jais(prompt, question_type=question_type)


def get_prediction_local_jais2(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from local Jais-2 model using chat templates.
    
    Based on official Jais-2 usage from: https://huggingface.co/inceptionai/Jais-2-8B-Chat
    
    Args:
        question: The question to answer
        model_version: Jais-2 model path (e.g., "inceptionai/Jais-2-8B-Chat")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature (0-1). Default 0 for deterministic output.
        max_tokens: Maximum output tokens. If None, uses question-type-specific default.
                    If set in config.yaml, that value is used. Minimum: 20 tokens.
        **kwargs: Additional parameters (choice1-4, text, discipline, few_shots, 
            abstention, verbalized_elicitation, verbose_instructions, 
            show_cot, word_limit)
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    
    Note:
        - max_tokens is read from config.yaml if set there
        - If max_tokens is None in config.yaml, question-type defaults are used:
          * MCQ: 512 tokens (short answers)
          * COMP: 2000 tokens (reading comprehension needs longer responses)
          * KNOW: 2000 tokens (knowledge questions need detailed answers)
          * Other: 1024 tokens
        - Minimum constraint: 20 tokens (below this may cause incomplete outputs)
        - Maximum constraint: Model-size-aware context length:
          * Jais-2-8B: 8,192 tokens (verified from Hugging Face docs)
          * Jais-2-70B: 32,768 tokens (typical for 70B models)
        - CoT (Chain of Thought) is handled via show_cot parameter in prompt generation
    """
    question_type = normalize_question_type(question_type)
    
    # Detect model size early for max_tokens validation (before loading model)
    model_version_lower = model_version.lower()
    is_70b_model = "70b" in model_version_lower or "70billion" in model_version_lower
    model_size = "70B" if is_70b_model else "8B"
    
    # Set default max_tokens based on question type if not provided
    if max_tokens is None:
        if question_type == "MCQ":
            max_tokens = 512  # MCQ answers are typically short
        elif question_type in ["COMP", "KNOW"]:
            max_tokens = 2000  # COMP and KNOW need longer, detailed responses
        else:
            max_tokens = 1024  # Default for other question types
    
    # Validate minimum constraint (below 20 tokens may cause incomplete outputs)
    if max_tokens < 20:
        print(f"[WARNING] max_tokens={max_tokens} from config is below provider minimum (20). Setting to 20.")
        max_tokens = 20
    
    # Maximum constraint: Model-size-aware context length
    # Jais-2-8B: 8,192 tokens (verified from Hugging Face docs)
    # Jais-2-70B: 32,768 tokens (typical for 70B models, verify with provider if different)
    max_context_length = 32768 if is_70b_model else 8192  # 70B: 32K, 8B: 8K
    
    if max_tokens > max_context_length:
        print(f"[WARNING] max_tokens={max_tokens} from config exceeds provider maximum ({max_context_length} for Jais-2-{model_size}). Setting to {max_context_length}.")
        max_tokens = max_context_length

    # Lazy load model with memory optimization
    # Check for failed auth marker to avoid repeated attempts
    if clients.jais2_model == "FAILED_AUTH" or clients.jais2_tokenizer == "FAILED_AUTH":
        # Model failed to load due to authentication/access - skip this question
        return None
    
    # Check if model is already loaded (reuse existing instance)
    model_already_loaded = clients.jais2_model is not None and clients.jais2_tokenizer is not None
    if model_already_loaded:
        # Model already loaded - show current mode if available
        if isinstance(clients.jais2_model, str):
            # Invalid state
            return None
        # Model is loaded and ready - show mode status
        current_mode = clients.jais2_loading_mode
        # if current_mode:
            # print(f"[INFO] 🔄 Jais-2 model already loaded (Mode: {current_mode})")
        # Proceed with inference
        pass
    elif not clients.jais2_model or not clients.jais2_tokenizer:
        print(f"[INFO] Loading Jais-2 model: {model_version}...")
        
        # Model size already detected above for max_tokens validation
        # Reuse the detection here for memory/device settings
        print(f"[INFO] Detected Jais-2-{model_size} model")
        
        # Get authentication token
        hf_token = clients.huggingface_token
        if not hf_token and not os.path.exists(os.path.expanduser("~/.cache/huggingface/hub")):
            print(f"[WARNING] No Hugging Face token found and no cache detected.")
            print(f"[INFO] For gated models, you need:")
            print(f"  1. Token from: https://huggingface.co/settings/tokens")
            print(f"  2. Add to .env: HUGGINGFACE_TOKEN=your_token")
            print(f"  3. Accept access: https://huggingface.co/{model_version}")
        
        # Memory optimization based on model size
        if is_70b_model:
            # 70B model requires more aggressive optimization
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "auto"  # Let transformers handle multi-GPU
            print(f"[INFO] Using {dtype} precision for 70B model (requires ~140GB VRAM or multi-GPU)")
        else:
            # 8B model - standard optimization
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device_map = "auto"
            print(f"[INFO] Using {dtype} precision for 8B model (requires ~16GB VRAM)")
        
        # Check if it's a local path (absolute/relative path or existing directory)
        is_local_path = (
            ("/" in model_version or "\\" in model_version) and
            (os.path.exists(model_version) and os.path.isdir(model_version))
        )
        
        if is_local_path:
            print(f"[INFO] Detected local model path: {model_version}")
            print(f"[INFO] Loading from local directory...")
            
            # Load directly from local path
            try:
                clients.jais2_tokenizer = AutoTokenizer.from_pretrained(
                    model_version,
                    local_files_only=True,  # Force offline for local paths
                    use_fast=False  # Use slow tokenizer for compatibility
                )
                
                clients.jais2_model = AutoModelForCausalLM.from_pretrained(
                    model_version,
                    torch_dtype=dtype,
                    device_map=device_map,
                    local_files_only=True,  # Force offline for local paths
                    trust_remote_code=False
                )
                
                loading_mode = "LOCAL_PATH"
                clients.jais2_loading_mode = "LOCAL_PATH"
                print(f"[OK]  Jais-2-{model_size} model loaded successfully from local path")
                success = True
                
            except Exception as local_error:
                print(f"[ERROR] Failed to load from local path: {local_error}")
                raise local_error
        else:
            # Hybrid loading approach: Try offline first (from cache), then online if needed
            success = False
            loading_mode = None
            
            # Step 1: Try offline loading first (fastest, no network needed)
            try:
                print(f"[INFO] Attempting OFFLINE loading from cache...")
                
                clients.jais2_tokenizer = AutoTokenizer.from_pretrained(
                    model_version,
                    local_files_only=True,  # Offline: use cached files only, no download
                    token=hf_token
                )
                
                clients.jais2_model = AutoModelForCausalLM.from_pretrained(
                    model_version,
                    torch_dtype=dtype,
                    device_map=device_map,
                    local_files_only=True,  # Offline: use cached files only, no download
                    token=hf_token,
                    trust_remote_code=False
                )
                
                loading_mode = "OFFLINE"
                clients.jais2_loading_mode = "OFFLINE"
                print(f"[OK]  Jais-2-{model_size} model loaded successfully via OFFLINE mode")
                print(f"[INFO] Status: Using cached files (no download, no network required)")
                success = True
                
            except Exception as offline_error:
                offline_error_msg = str(offline_error).lower()
                print(f"[INFO] Offline loading failed: {type(offline_error).__name__}")
                
                # Check if it's just a cache miss (normal) vs other errors
                is_cache_miss = any(keyword in offline_error_msg for keyword in [
                    "local_files_only", "not found in the cached files", "localentrynotfound"
                ])
                
                if is_cache_miss:
                    print(f"[INFO]  Model not cached - will attempt online download...")
                else:
                    # Other error (authentication, memory, etc.) - propagate it
                    raise offline_error
        
        # Step 2: Try online loading if offline failed
        if not success:
            try:
                # Show download warning for large models
                if is_70b_model:
                    print(f"[WARNING]  Downloading Jais-2-70B (~140GB). This will take hours!")
                    print(f"[INFO] Consider using a pre-downloaded model or smaller variant")
                else:
                    print(f"[INFO]  Downloading Jais-2-8B (~16GB, 10-30 minutes depending on connection)...")
                
                print(f"[INFO] Loading model from Hugging Face Hub...")
                
                clients.jais2_tokenizer = AutoTokenizer.from_pretrained(
                    model_version,
                    token=hf_token  # Pass token for gated repositories
                )
                
                clients.jais2_model = AutoModelForCausalLM.from_pretrained(
                    model_version,
                    torch_dtype=dtype,
                    device_map=device_map,
                    token=hf_token,
                    trust_remote_code=False  # Security: don't execute remote code
                )
                
                loading_mode = "ONLINE"
                clients.jais2_loading_mode = "ONLINE"
                print(f"[OK]  Jais-2-{model_size} model loaded successfully via ONLINE mode")
                print(f"[INFO] Status: Model cached for future use (will use OFFLINE mode next time)")
                success = True
                
            except Exception as download_error:
                download_error_msg = str(download_error).lower()
                print(f"[ERROR] Online loading failed: {type(download_error).__name__}")
                
                # Check for authentication/access errors
                is_auth_error = any(keyword in download_error_msg for keyword in [
                    "gated repo", "401", "403", "authentication", "access denied",
                    "forbidden", "unauthorized", "token", "must be authenticated"
                ])
                
                if is_auth_error:
                    print(f"[ERROR] Authentication/access required for gated repository")
                    print(f"[INFO] To fix:")
                    print(f"  1. Get token: https://huggingface.co/settings/tokens")
                    print(f"  2. Accept access: https://huggingface.co/{model_version}")
                    print(f"  3. Add to .env: HUGGINGFACE_TOKEN=your_token")
                    print(f"  4. Wait 1-5 minutes for access to propagate")
                else:
                    print(f"[ERROR] Download failed: {str(download_error)[:200]}")
                
                # Mark as failed
                clients.jais2_model = "FAILED_AUTH"
                clients.jais2_tokenizer = "FAILED_AUTH"
                print(f"[SKIP] Jais-2 model will be skipped for remaining questions")
                raise download_error

    # Generate prompt using the standard prompt generator
    # CoT is handled via show_cot parameter in generate_prompt
    prompt = generate_prompt(
        question=question,
        question_type=question_type,
        **kwargs
    )
    
    # Check if prompt is empty (question validation failed)
    if not prompt:
        return None

    try:
        # Ensure pad_token is set (required for proper attention mask)
        if clients.jais2_tokenizer.pad_token is None:
            clients.jais2_tokenizer.pad_token = clients.jais2_tokenizer.eos_token
        
        # Use chat template for proper Jais-2 formatting
        # Following official Jais-2 usage pattern from Hugging Face documentation:
        # https://huggingface.co/docs/transformers/main/en/model_doc/jais2
        # Format as chat messages (system + user for better Arabic responses)
        system_prompt = "أجب باللغة العربية بطريقة رسمية وواضحة."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template using official pattern:
        # apply_chat_template with tokenize=True, return_dict=True, return_tensors="pt"
        # This is more efficient than tokenizing separately
        if hasattr(clients.jais2_tokenizer, 'apply_chat_template') and clients.jais2_tokenizer.chat_template:
            inputs = clients.jais2_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            # Move to model device
            inputs = {k: v.to(clients.jais2_model.device) for k, v in inputs.items()}
        else:
            # Fallback to direct tokenization (should not happen with Jais-2)
            inputs = clients.jais2_tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
                padding=True
            )
            inputs = {k: v.to(clients.jais2_model.device) for k, v in inputs.items()}
        
        # CRITICAL: Remove token_type_ids if present (required for Jais-2 as per documentation)
        # This is explicitly mentioned in the Hugging Face example
        inputs.pop("token_type_ids", None)
        
        # Get token IDs for generation
        pad_token_id = clients.jais2_tokenizer.pad_token_id
        eos_token_id = clients.jais2_tokenizer.eos_token_id
        
        # Generate response
        # Build generation kwargs based on temperature
        # When temperature=0, use greedy decoding (do_sample=False)
        # When temperature>0, use sampling with temperature
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "min_new_tokens": 10,  # Ensure minimum generation length
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.1,  # Help prevent repetition
            "no_repeat_ngram_size": 3  # Prevent repeating 3-grams
        }
        
        # Only add sampling parameters when do_sample=True
        # These parameters are ignored when do_sample=False, causing warnings
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_k": 50,
                "top_p": 0.95
            })
        
        with torch.no_grad():  # Save memory during inference
            outputs = clients.jais2_model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode response
        # Extract only the new tokens (assistant's response), not the input prompt
        # Following official pattern: decode from input_length onwards
        # https://huggingface.co/docs/transformers/main/en/model_doc/jais2#transformers.Jais2ForCausalLM
        input_length = inputs["input_ids"].shape[-1]  # Use shape[-1] for consistency
        response_tokens = outputs[0][input_length:]
        response_text = clients.jais2_tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Calculate token counts
        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = len(response_tokens)
        
        return {
            "text": response_text.strip(),
            "tokens": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Error with Jais-2 model: {str(e)}")
        raise


def get_prediction_local_allam(
    question: str,
    model_version: str,
    question_type: str = "MCQ",
    temperature: float = 0.6,
    max_tokens: Optional[int] = None,
    **kwargs) -> Optional[dict]:
    """
    Get prediction from local ALLaM model using chat templates.
    
    Args:
        question: The question to answer
        model_version: ALLaM model path (e.g., "ALLaM-AI/ALLaM-7B-Instruct-preview")
        question_type: Type of question (MCQ, TF, FRQ, COMP, etc.)
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum output tokens. If None, uses question-type-specific default.
                    If set in config.yaml, that value is used. Minimum: 20 tokens.
        **kwargs: Additional parameters (choice1-4, text, discipline, few_shots, 
            abstention, verbalized_elicitation, verbose_instructions, 
            show_cot, word_limit)
    
    Returns:
        Optional[dict]: Dictionary with 'text' and 'tokens' keys, or None on error
    
    Note:
        - max_tokens is read from config.yaml if set there
        - If max_tokens is None in config.yaml, question-type defaults are used:
          * MCQ: 512 tokens (short answers)
          * COMP: 2000 tokens (reading comprehension needs longer responses)
          * KNOW: 2000 tokens (knowledge questions need detailed answers)
          * Other: 1024 tokens
        - Minimum constraint: 20 tokens (below this may cause incomplete outputs)
        - Maximum constraint: Model context length
          * ALLaM-7B: 4096 tokens
          * ALLaM-34B: 8192 tokens (or as per model specification)
    """
    question_type = normalize_question_type(question_type)
    
    # Set default max_tokens based on question type if not provided
    if max_tokens is None:
        if question_type == "MCQ":
            max_tokens = 512  # MCQ answers are typically short
        elif question_type in ["COMP", "KNOW"]:
            max_tokens = 2000  # COMP and KNOW need longer, detailed responses
        else:
            max_tokens = 1024  # Default for other question types
    
    # Validate minimum constraint (below 20 tokens may cause incomplete outputs)
    # Based on Hugging Face recommendations and model behavior
    if max_tokens < 20:
        print(f"[WARNING] max_tokens={max_tokens} from config is below provider minimum (20). Setting to 20.")
        max_tokens = 20
    
    # Detect model size for max_tokens validation
    model_version_lower = model_version.lower()
    is_34b_model = "34b" in model_version_lower or "34" in model_version_lower
    is_7b_model = "7b" in model_version_lower or "7" in model_version_lower
    
    # Maximum constraint based on model size
    if is_34b_model:
        max_context_length = 8192  # ALLaM-34B typically has 8K context
        if max_tokens > max_context_length:
            print(f"[WARNING] max_tokens={max_tokens} from config exceeds provider maximum ({max_context_length} for ALLaM-34B). Setting to {max_context_length}.")
            max_tokens = max_context_length
    elif is_7b_model:
        max_context_length = 4096  # ALLaM-7B has 4K context
        if max_tokens > max_context_length:
            print(f"[WARNING] max_tokens={max_tokens} from config exceeds provider maximum ({max_context_length} for ALLaM-7B). Setting to {max_context_length}.")
            max_tokens = max_context_length
    else:
        # Default: assume 4K context (conservative)
        max_context_length = 4096
        if max_tokens > max_context_length:
            print(f"[WARNING] max_tokens={max_tokens} from config exceeds default maximum ({max_context_length}). Setting to {max_context_length}.")
            max_tokens = max_context_length

    # Lazy load model with memory optimization
    # Check for failed auth marker to avoid repeated attempts
    if clients.allam_model == "FAILED_AUTH" or clients.allam_tokenizer == "FAILED_AUTH":
        # Model failed to load due to authentication - skip this question
        return None
    
    if not clients.allam_model or not clients.allam_tokenizer:
        try:
            print(f"[INFO] Loading ALLaM model: {model_version}...")
            
            # Detect model size from path/name for memory optimization
            model_version_lower = model_version.lower()
            is_34b_model = "34b" in model_version_lower or "34" in model_version_lower
            is_7b_model = "7b" in model_version_lower or "7" in model_version_lower
            
            # Check if it's a local path (absolute/relative path or existing directory)
            # Local paths should contain "/" and exist as directories
            is_local_path = (
                ("/" in model_version or "\\" in model_version) and
                (os.path.exists(model_version) and os.path.isdir(model_version))
            )
            
            if is_local_path:
                print(f"[INFO] Detected local model path: {model_version}")
                print(f"[INFO] Loading from local directory...")
            else:
                print(f"[INFO] Loading from Hugging Face Hub...")
            
            # Use slow tokenizer (use_fast=False) for SentencePiece-based tokenizers
            # ALLaM uses SentencePiece which doesn't have a fast tokenizer converter
            hf_token = clients.huggingface_token
            
            # For local paths, don't pass token (not needed) and use local_files_only=True
            # For Hugging Face paths, pass token if available (for gated repositories)
            tokenizer_kwargs = {
                "use_fast": False,  # Required for SentencePiece tokenizers
            }
            if is_local_path:
                tokenizer_kwargs["local_files_only"] = True  # Force offline loading for local paths
            elif hf_token:
                tokenizer_kwargs["token"] = hf_token
            
            clients.allam_tokenizer = AutoTokenizer.from_pretrained(
                model_version,
                **tokenizer_kwargs
            )
            
            # Memory optimization based on model size
            if is_34b_model:
                # 34B model requires more aggressive optimization
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map = "auto"  # Let transformers handle multi-GPU if needed
                print(f"[INFO] Detected ALLaM-34B model - requires ~68GB VRAM (or multi-GPU)")
                print(f"[INFO] Using {dtype} precision for 34B model")
            elif is_7b_model:
                # 7B model - standard optimization
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map = "auto"
                print(f"[INFO] Detected ALLaM-7B model - requires ~14GB VRAM")
                print(f"[INFO] Using {dtype} precision for 7B model")
            else:
                # Default: assume larger model, use conservative settings
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                device_map = "auto"
                print(f"[WARNING] Model size not detected - using conservative memory settings")
            
            # For local paths, don't pass token and use local_files_only=True
            # For Hugging Face paths, pass token if available
            model_kwargs = {
                "torch_dtype": dtype,  # Use torch_dtype instead of dtype for consistency
                "device_map": device_map,
                "trust_remote_code": False,  # Security: don't execute remote code
            }
            if is_local_path:
                model_kwargs["local_files_only"] = True  # Force offline loading for local paths
            elif hf_token:
                model_kwargs["token"] = hf_token
            
            clients.allam_model = AutoModelForCausalLM.from_pretrained(
                model_version, 
                **model_kwargs
            )
            print(f"[OK] ALLaM model loaded successfully")
        except Exception as e:
            error_msg = str(e)
            # Check for authentication/access errors (401, 403, gated repo)
            is_auth_error = any(keyword in error_msg.lower() for keyword in [
                "gated repo", "401", "403", "authentication", "access denied",
                "cannot access gated repo", "restricted", "must be authenticated",
                "forbidden"
            ])
            
            if is_auth_error:
                # Mark as failed to prevent repeated attempts
                clients.allam_model = "FAILED_AUTH"
                clients.allam_tokenizer = "FAILED_AUTH"
                
                if "403" in error_msg or "forbidden" in error_msg.lower():
                    print(f"[ERROR] Failed to load ALLaM model: Repository access not granted")
                    print(f"[INFO] Your token is valid, but you need to:")
                    print(f"  1. Visit: https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview")
                    print(f"  2. Click 'Agree and access repository' or 'Request access'")
                    print(f"  3. Wait 1-5 minutes for access to propagate")
                    print(f"  4. Restart the pipeline")
                else:
                    print(f"[ERROR] Failed to load ALLaM model: Authentication required")
                    print(f"[INFO] This model requires:")
                    print(f"  1. Hugging Face account with access granted")
                    print(f"  2. Request access at: https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview")
                    print(f"  3. Set HUGGINGFACE_TOKEN in .env file after access is granted")
            else:
                print(f"[ERROR] Failed to load ALLaM model: {e}")
            raise

    # Generate prompt using the standard prompt generator
    prompt = generate_prompt(
        question=question,
        question_type=question_type,
        **kwargs
    )
    
    # Check if prompt is empty (question validation failed)
    if not prompt:
        return None

    try:
        # Ensure pad_token is set (required for proper attention mask)
        if clients.allam_tokenizer.pad_token is None:
            clients.allam_tokenizer.pad_token = clients.allam_tokenizer.eos_token
        
        # Use chat template for proper instruct model formatting
        # Format as chat messages for the model
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template if available, otherwise use direct tokenization
        # Following official ALLaM usage pattern from Hugging Face model card
        if hasattr(clients.allam_tokenizer, 'apply_chat_template') and clients.allam_tokenizer.chat_template:
            # Official pattern: apply_chat_template with tokenize=False, then tokenize separately
            # This ensures proper formatting as per ALLaM documentation
            formatted_prompt = clients.allam_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Tokenize the formatted prompt
            inputs = clients.allam_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                return_token_type_ids=False
            )
        else:
            # Fallback to direct tokenization
            inputs = clients.allam_tokenizer(
                prompt,
                return_tensors="pt",
                return_token_type_ids=False,
                padding=True  # This will create attention_mask automatically
            )
        
        # Move inputs to model device
        inputs = {k: v.to(clients.allam_model.device) for k, v in inputs.items()}
        
        # Get token IDs for generation
        pad_token_id = clients.allam_tokenizer.pad_token_id
        eos_token_id = clients.allam_tokenizer.eos_token_id
        
        # Generate response
        # Build generation kwargs based on temperature
        # When temperature=0, use greedy decoding (do_sample=False)
        # When temperature>0, use sampling with temperature, top_k, top_p
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "min_new_tokens": 10,  # Ensure minimum generation length
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "repetition_penalty": 1.1,  # Help prevent repetition
            "no_repeat_ngram_size": 3  # Prevent repeating 3-grams
        }
        
        # Only add sampling parameters when do_sample=True
        # These parameters are ignored when do_sample=False, causing warnings
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_k": 50,  # Official example uses top_k=50
                "top_p": 0.95  # Official example uses top_p=0.95
            })
        
        with torch.no_grad():  # Save memory during inference
            outputs = clients.allam_model.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Decode response
        # Extract only the new tokens (assistant's response), not the input prompt
        # This ensures we only get the model's response, not the full conversation
        input_length = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_length:]
        response_text = clients.allam_tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Calculate token counts
        input_token_count = inputs["input_ids"].shape[1]
        output_token_count = len(response_tokens)
        
        return {
            "text": response_text.strip(),
            "tokens": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Error with ALLaM model: {str(e)}")
        raise
