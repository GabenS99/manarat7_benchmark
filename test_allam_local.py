"""
Test script for get_prediction_local_allam function.
"""
import sys
from pathlib import Path

# Add the 'scripts' directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from predictors import get_prediction_local_allam

def test_allam_local(show_cot=False):
    """Test the local ALLaM prediction function.
    
    Args:
        show_cot: Whether to test with Chain of Thought reasoning enabled
    """
    print("=" * 80)
    print("Testing get_prediction_local_allam")
    if show_cot:
        print("WITH Chain of Thought (CoT) enabled")
    print("=" * 80)
    
    # Test parameters
    # NOTE: For ALLaM-34B providers, update model_version to your local model path
    # Example: model_version = "/path/to/your/allam-34b-model"
    test_question = "ما هي شروط الحديث الصحيح بحسب السخاوي؟"
    
    # Try to read from config.yaml, otherwise use default
    import yaml
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            models = config.get("prediction_models", {}).get("local_allam", [])
            if models and len(models) > 0:
                model_version = models[0]
                print(f"[INFO] Using model from config.yaml: {model_version}")
            else:
                model_version = "ALLaM-AI/ALLaM-7B-Instruct-preview"
                print(f"[INFO] No model in config.yaml, using default: {model_version}")
    except Exception as e:
        model_version = "ALLaM-AI/ALLaM-7B-Instruct-preview"
        print(f"[WARNING] Could not read config.yaml: {e}")
        print(f"[INFO] Using default model: {model_version}")
    
    question_type = "MCQ"
    
    # Test with MCQ choices
    test_choices = {
        "choice1": "اتصال السند وعدالة الرواة وضبطهم",
        "choice2": "أن يكون متواترًا فقط",
        "choice3": "أن يُروى عن الصحابة فقط",
        "choice4": "كل الأجوبة خاطئة"
    }
    
    print(f"\n[TEST] Question: {test_question}")
    print(f"[TEST] Model: {model_version}")
    print(f"[TEST] Question Type: {question_type}")
    print(f"[TEST] Choices: {len(test_choices)} options")
    print(f"[TEST] Chain of Thought: {show_cot}")
    print("\n[INFO] Calling get_prediction_local_allam...")
    print("[INFO] Note: First call will download/load the model (may take time)\n")
    
    try:
        result = get_prediction_local_allam(
            question=test_question,
            model_version=model_version,
            question_type=question_type,
            temperature=0.6,
            max_tokens=512,  # Reduced for faster testing
            show_cot=show_cot,
            **test_choices
        )
        
        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        
        if result is None:
            print("[FAIL] Function returned None")
            return False
        
        print(f"\n[SUCCESS] Function returned result:")
        print(f"  Type: {type(result)}")
        
        if isinstance(result, dict):
            print(f"\n  Keys: {list(result.keys())}")
            
            if "text" in result:
                print(f"\n  Response Text:")
                print(f"    {result['text']}")
            
            if "tokens" in result:
                tokens = result["tokens"]
                print(f"\n  Token Usage:")
                if isinstance(tokens, dict):
                    print(f"    Prompt tokens: {tokens.get('prompt_tokens', 'N/A')}")
                    print(f"    Completion tokens: {tokens.get('completion_tokens', 'N/A')}")
                    print(f"    Total tokens: {tokens.get('total_tokens', 'N/A')}")
                else:
                    print(f"    {tokens}")
        else:
            print(f"\n  Value: {result}")
        
        print("\n" + "=" * 80)
        print("[OK] Test completed successfully!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("[ERROR] Test failed with exception:")
        print("=" * 80)
        print(f"  Exception Type: {type(e).__name__}")
        print(f"  Error Message: {str(e)}")
        import traceback
        print("\n  Full Traceback:")
        traceback.print_exc()
        print("=" * 80)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test local ALLaM prediction function")
    parser.add_argument("--cot", action="store_true", 
                       help="Test with Chain of Thought (CoT) enabled")
    args = parser.parse_args()
    
    success = test_allam_local(show_cot=args.cot)
    sys.exit(0 if success else 1)
