"""
Test script for get_prediction_local_jais2 function.
"""
import sys
from pathlib import Path

# Add the 'scripts' directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from predictors import get_prediction_local_jais2

def test_jais2_local(show_cot=False):
    """Test the local Jais-2 prediction function.
    
    Args:
        show_cot: Whether to test with Chain of Thought reasoning enabled
    """
    print("=" * 80)
    print("Testing get_prediction_local_jais2")
    if show_cot:
        print("WITH Chain of Thought (CoT) enabled")
    print("=" * 80)
    
    # Test parameters
    test_question = "ما هي شروط الحديث الصحيح بحسب السخاوي؟"
    model_version = "inceptionai/Jais-2-8B-Chat"
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
    print("\n[INFO] Calling get_prediction_local_jais2...")
    print("[INFO] Note: First call will download/load the model (may take time)")
    print("[INFO] Model will be cached for subsequent runs\n")
    
    try:
        result = get_prediction_local_jais2(
            question=test_question,
            model_version=model_version,
            question_type=question_type,
            temperature=0,
            max_tokens=512,  # Reduced for faster testing
            show_cot=show_cot,
            **test_choices
        )
        
        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        
        if result is None:
            print("[FAIL] Function returned None")
            print("[INFO] This may indicate:")
            print("  - Model failed to load (check authentication/access)")
            print("  - Model is marked as FAILED_AUTH (check previous errors)")
            print("  - Prompt generation failed")
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
        print("\n[INFO] Common issues:")
        print("  - Authentication: Check HUGGINGFACE_TOKEN in .env file")
        print("  - Repository access: Visit https://huggingface.co/inceptionai/Jais-2-8B-Chat")
        print("  - Model download: First run requires internet connection (~16GB download)")
        print("  - GPU memory: Ensure sufficient VRAM (16GB+ for 8B model)")
        print("=" * 80)
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test local Jais-2 prediction function")
    parser.add_argument("--cot", action="store_true", 
                       help="Test with Chain of Thought (CoT) enabled")
    args = parser.parse_args()
    
    success = test_jais2_local(show_cot=args.cot)
    sys.exit(0 if success else 1)
