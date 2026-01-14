from typing import Optional, Union, Dict, Any


# ========================================================================
# PREDICTION PROMPT GENERATION
# ========================================================================


def generate_prompt(
    question: Optional[str] = None, 
    choice1: Optional[str] = None,
    choice2: Optional[str] = None,
    choice3: Optional[str] = None,
    choice4: Optional[str] = None, 
    question_type: str = "MCQ", 
    text: Optional[str] = None, 
    discipline: Optional[str] = None, 
    few_shots: bool = False,
    abstention: bool = False,
    verbalized_elicitation: bool = True,
    verbose_instructions: bool = True,
    show_cot: bool = False,
    word_limit: Optional[int] = None) -> str:
    """
    Generate prompts based on the agreed 5-part structure:
    1. System/persona definition with discipline
    2. Task clarification
    3. Few-shot examples
    4. Question with choices (for MCQ)
    5. Verbose/enforcement instructions
    
    Args:
        question: The question text
        choice1-4: MCQ choices (optional)
        question_type: "MCQ", "KNOW", or "COMP"
        text: Required for COMP questions
        discipline: Islamic studies discipline (default: "العلوم الشرعية")
        few_shots: Whether to include few-shot examples
        abstention: Whether to include abstention instructions
        verbose_instructions: Whether to include detailed instructions
        show_cot: Whether to show chain-of-thought reasoning
    """
    if question_type == "MCQ":
        if not all([question, choice1, choice2, choice3, choice4]):
            print("[Error]: MCQ question requires all parameters (question, choice1, choice2, choice3, choice4)")
            return None
    elif question_type == "KNOW":
        if not question:
            print("[Error]: KNOW question requires question parameter")
            return None
    elif question_type == "COMP":
        if not question or not text:
            print("[Error]: COMP question requires both question and text parameters")
            return None
    
    # Set default discipline
    if discipline is None or not discipline.strip():
        discipline = "العلوم الشرعية"
    
    
    # ========================================================================
    # PART 1: SYSTEM/PERSONA DEFINITION
    # ========================================================================
    persona = f"أنت خبير متخصص في {discipline}، تتمتع بمعرفة عميقة ودقيقة في هذا المجال."
    
    # ========================================================================
    # PART 2: TASK CLARIFICATION
    # ========================================================================
    if question_type == "MCQ":
        task_clarification = "مهمتك هي اختيار الإجابة الصحيحة من بين الخيارات المتاحة."
    elif question_type == "KNOW":
        task_clarification = "مهمتك هي الإجابة عن السؤال بدقة وإيجاز، مع الاستشهاد بالأدلة الشرعية عند الحاجة."
    elif question_type == "COMP":
        task_clarification = "مهمتك هي قراءة النص بعناية والإجابة عن السؤال بناءً على المعلومات الواردة فيه."
    else:
        task_clarification = ""
    
    # ========================================================================
    # PART 3: FEW-SHOT EXAMPLES
    # ========================================================================
    few_shot_examples = ""
    if few_shots:
        if question_type == "MCQ":
            if show_cot:
                few_shot_examples = """أمثلة توضيحية:

**مثال 1:**
السؤال: ما هو المصطلح الذي يُطلق على أعلى درجات الجنة وأفضلها، والتي أُعدت لأكرم الخلق على الله؟
الخيارات:
أ) جنة عدن
ب) جنة المأوى
ج) دار السلام
د) الفردوس الأعلى

<تفكير>
الخيار أ (جنة عدن) هو اسم لجنة من الجنان، لكن ليس أعلاها.
الخيار ب (جنة المأوى) هو اسم لجنة أخرى، لكن ليس أعلاها.
الخيار ج (دار السلام) هو اسم للجنة، لكن ليس أعلاها.
الخيار د (الفردوس الأعلى) هو المصطلح الصحيح الذي يُطلق على أعلى درجات الجنة وأفضلها، والتي أعدت لأكرم الخلق على الله.
</تفكير>
الإجابة: د

**مثال 2:**
السؤال: قارن بين أذى قريش للرسول صلى الله عليه وسلم في بيته قبل وفاة أبي طالب، وما فعلوه بعد وفاته، وماذا كان رد فعل الرسول على الأذى؟
الخيارات:
أ) قبل وفاة أبي طالب لم يجرؤوا على أذيته، وبعد وفاته اشتد الأذى. وكان الرسول يخرج الأذى ويلقيه على بابهم ويسأل: "أي جوار هذا؟".
ب) قبل وفاة أبي طالب، كانوا يؤذونه بإلقاء الأذى عليه في بيته، وبعد وفاته ازداد الأذى. وكان الرسول يخرج الأذى ويلقيه على بابهم ويسأل: "أي جوار هذا؟".
ج) قبل وفاة أبي طالب، كانوا يؤذونه لفظيًا فقط، وبعد وفاته أذوه جسديًا. وكان الرسول يتجاهل الأذى.
د) لم يؤذوه أبدًا قبل وفاة أبي طالب، وبعد وفاته لم يستطيعوا أذيته كثيرًا. وكان الرسول يدعو عليهم.

<تفكير>
الخيار أ يقول إنهم لم يجرؤوا على أذيته قبل وفاة أبي طالب، وهذا غير صحيح لأنهم كانوا يؤذونه.
الخيار ب يقول إنهم كانوا يؤذونه بإلقاء الأذى عليه في بيته قبل وفاة أبي طالب، وبعد وفاته ازداد الأذى، وكان الرسول يخرج الأذى ويلقيه على بابهم ويسأل "أي جوار هذا؟". هذا هو الصحيح.
الخيار ج يقول إنهم كانوا يؤذونه لفظيًا فقط قبل وفاة أبي طالب، وهذا غير دقيق.
الخيار د يقول إنهم لم يؤذوه أبدًا، وهذا غير صحيح.
</تفكير>
الإجابة: ب

**مثال 3:**
السؤال: من القائل:  مات النبي صلى الله عليه وسلم ولم يجمع القرآن غير أربعة: أبو الدرداء، ومعاذ بن جبل، وزيد بن ثابت، وأبو زيد  ؟
الخيارات:
أ) عثمان بن عفان
ب) عمر بن الخطاب
ج) أنس بن مالك
د) علي بن أبي طالب

<تفكير>
هذا القول منسوب إلى أنس بن مالك رضي الله عنه، وهو يذكر أن النبي صلى الله عليه وسلم مات ولم يجمع القرآن غير أربعة من الصحابة. الخيار ج (أنس بن مالك) هو الصحيح.
</تفكير>
الإجابة: ج

"""
            else:
                few_shot_examples = """أمثلة توضيحية:

**مثال 1:**
السؤال: ما هو المصطلح الذي يُطلق على أعلى درجات الجنة وأفضلها، والتي أُعدت لأكرم الخلق على الله؟
أ) جنة عدن
ب) جنة المأوى
ج) دار السلام
د) الفردوس الأعلى
الإجابة: د

**مثال 2:**
السؤال: قارن بين أذى قريش للرسول صلى الله عليه وسلم في بيته قبل وفاة أبي طالب، وما فعلوه بعد وفاته، وماذا كان رد فعل الرسول على الأذى؟
أ) قبل وفاة أبي طالب لم يجرؤوا على أذيته، وبعد وفاته اشتد الأذى. وكان الرسول يخرج الأذى ويلقيه على بابهم ويسأل: "أي جوار هذا؟".
ب) قبل وفاة أبي طالب، كانوا يؤذونه بإلقاء الأذى عليه في بيته، وبعد وفاته ازداد الأذى. وكان الرسول يخرج الأذى ويلقيه على بابهم ويسأل: "أي جوار هذا؟".
ج) قبل وفاة أبي طالب، كانوا يؤذونه لفظيًا فقط، وبعد وفاته أذوه جسديًا. وكان الرسول يتجاهل الأذى.
د) لم يؤذوه أبدًا قبل وفاة أبي طالب، وبعد وفاته لم يستطيعوا أذيته كثيرًا. وكان الرسول يدعو عليهم.
الإجابة: ب

**مثال 3:**
السؤال: من القائل:  مات النبي صلى الله عليه وسلم ولم يجمع القرآن غير أربعة: أبو الدرداء، ومعاذ بن جبل، وزيد بن ثابت، وأبو زيد  ؟
أ) عثمان بن عفان
ب) عمر بن الخطاب
ج) أنس بن مالك
د) علي بن أبي طالب
الإجابة: ج

"""
        elif question_type == "KNOW":
            if show_cot:
                few_shot_examples = """أمثلة توضيحية:

**مثال 1:**
السؤال: ماذا كانت رابعة تقول لسفيان الثوري،وفق ما نقله ابن عباد في شرحه للحكم العطائية؟
<تفكير>
السؤال يطلب ما كانت رابعة العدوية تقوله لسفيان الثوري، وفق ما نقله ابن عباد في شرحه للحكم العطائية. هذا سؤال عن نص محدد منسوب إلى رابعة العدوية في شرح ابن عباد.
</تفكير>
الإجابة: كانت تقول له: "نعم الرجل أنت لولا أنك تحب الدنيا."

**مثال 2:**
السؤال: كيف يرد القاضي على من استدل بقوله تعالى: {قُمِ اللَّيْلَ إِلَّا قَلِيلًا} على جواز استثناء الأكثر؟
<تفكير>
السؤال يطلب كيفية الرد على استدلال بآية قرآنية. المستدل استدل بآية {قُمِ اللَّيْلَ إِلَّا قَلِيلًا} على جواز استثناء الأكثر. الرد يكون ببيان أن الآية التالية {نِصْفَهُ} توضح أن المقصود هو قم نصفه، وليس استثناء الأكثر.
</تفكير>
الإجابة: بأن قوله تعالى: {نِصْفَهُ} أي: قم نصفه، وليس باستثناء

**مثال 3:**
السؤال: في أي طبقة من طبقات الصحابة عد الحاكم (أهل بدر) في كتابه علوم الحديث؟
<تفكير>
السؤال يطلب تحديد الطبقة التي عد فيها الحاكم أهل بدر في كتابه علوم الحديث. هذا سؤال عن تصنيف الحاكم للصحابة في طبقاتهم.
</تفكير>
الإجابة: الطبقة السابعة

"""
            else:
                few_shot_examples = """أمثلة توضيحية:

**مثال 1:**
السؤال: ماذا كانت رابعة تقول لسفيان الثوري،وفق ما نقله ابن عباد في شرحه للحكم العطائية؟
الإجابة: كانت تقول له: "نعم الرجل أنت لولا أنك تحب الدنيا."

**مثال 2:**
السؤال: كيف يرد القاضي على من استدل بقوله تعالى: {قُمِ اللَّيْلَ إِلَّا قَلِيلًا} على جواز استثناء الأكثر؟
الإجابة: بأن قوله تعالى: {نِصْفَهُ} أي: قم نصفه، وليس باستثناء

**مثال 3:**
السؤال: في أي طبقة من طبقات الصحابة عد الحاكم (أهل بدر) في كتابه علوم الحديث؟
الإجابة: الطبقة السابعة

"""
        elif question_type == "COMP":
            if show_cot:
                few_shot_examples = """**مثال توضيحي:**
              
النص: [ميثاق الظلم والعدوان] اجتمعوا في خيف بني كنانة من وادي المحصب فتحالفوا، على بني هاشم وبني المطلب ألايناكحوهم، ولا يبايعوهم، ولا يجالسوهم، ولا يخالطوهم، ولا يدخلوا بيوتهم، ولا يكلموهم، حتى يسلموا إليهم رسول الله صلى الله عليه وسلم للقتل، وكتبوا بذلك صحيفة فيها عهود ومواثيق «ألايقبلوا من بني هاشم صلحا أبدا، ولا تأخذهم بهم رأفة حتى يسلموه، للقتل» قال ابن القيم: يقال: كتبها منصور بن عكرمة بن عامر بن هاشم، ويقال: نضر بن الحارث، والصحيح أنه بغيض بن عامر بن هاشم، فدعا عليه رسول الله صلى الله عليه وسلم فشلت يده «١» . تم هذا الميثاق، وعلقت الصحيفة في جوف الكعبة، فانحاز بنو هاشم وبنو المطلب مؤمنهم وكافرهم- إلا أبا لهب- وحبسوا في شعب أبي طالب ليلة هلال المحرم سنة سبع من البعثة.

السؤال: متى حبس بنو هاشم وبنو المطلب في شعب أبي طالب؟

<تفكير>
السؤال يطلب تحديد الوقت الذي حُبس فيه بنو هاشم وبنو المطلب في شعب أبي طالب. أحتاج للبحث في النص عن هذه المعلومة المحددة. في نهاية النص، أجد: "وحبسوا في شعب أبي طالب ليلة هلال المحرم سنة سبع من البعثة." هذه هي الإجابة المباشرة من النص.
</تفكير>
الإجابة: ليلة هلال المحرم سنة سبع من البعثة.

"""
            else:   
                few_shot_examples = """**مثال توضيحي:**

النص: [ميثاق الظلم والعدوان] اجتمعوا في خيف بني كنانة من وادي المحصب فتحالفوا، على بني هاشم وبني المطلب ألايناكحوهم، ولا يبايعوهم، ولا يجالسوهم، ولا يخالطوهم، ولا يدخلوا بيوتهم، ولا يكلموهم، حتى يسلموا إليهم رسول الله صلى الله عليه وسلم للقتل، وكتبوا بذلك صحيفة فيها عهود ومواثيق «ألايقبلوا من بني هاشم صلحا أبدا، ولا تأخذهم بهم رأفة حتى يسلموه، للقتل» قال ابن القيم: يقال: كتبها منصور بن عكرمة بن عامر بن هاشم، ويقال: نضر بن الحارث، والصحيح أنه بغيض بن عامر بن هاشم، فدعا عليه رسول الله صلى الله عليه وسلم فشلت يده «١» . تم هذا الميثاق، وعلقت الصحيفة في جوف الكعبة، فانحاز بنو هاشم وبنو المطلب مؤمنهم وكافرهم- إلا أبا لهب- وحبسوا في شعب أبي طالب ليلة هلال المحرم سنة سبع من البعثة.

السؤال: متى حبس بنو هاشم وبنو المطلب في شعب أبي طالب؟

الإجابة: ليلة هلال المحرم سنة سبع من البعثة.

"""
    
    # ========================================================================
    # PART 4: QUESTION AND CHOICES
    # ========================================================================
    question_section = ""
    if question_type == "MCQ":
        question_section = f"""السؤال: {question}

الخيارات:
أ) {choice1}
ب) {choice2}
ج) {choice3}
د) {choice4}
"""
    elif question_type == "KNOW":
        question_section = f"السؤال: {question}\n"
    elif question_type == "COMP":
        question_section = f"""النص:
{text}

السؤال: {question}
"""
    
    # ========================================================================
    # PART 5: VERBOSE/ENFORCEMENT INSTRUCTIONS
    # ========================================================================
    instructions = ""
    if verbose_instructions:
        instructions = """تعليمات الإجابة:"""
         
        if question_type == "MCQ":
            instructions += """
    - أجب بحرف عربي واحد فقط من الخيارات التالية: أ، ب، ج، د
    - لا تكتب أي شرح أو تفسير إضافي
    - لا تكتب كلمات إضافية غير الحرف"""
        
        elif question_type == "KNOW":
            word_limit_text = f"حد أقصى {word_limit} كلمة" if word_limit else "بإيجاز دون التطرق إلى معلومات غير مطلوبة"
            instructions += f"""
    - أجب باللغة العربية فقط
    - {word_limit_text}
    - إن كانت الإجابة تتضمن ذكر دليل شرعي، تحقق من الآية أو الحديث قبل كتابة الإجابة"""
        
        elif question_type == "COMP":
            word_limit_text = f"حد أقصى {word_limit} كلمة" if word_limit else "بإيجاز دون التطرق إلى معلومات غير مطلوبة"
            instructions += f"""
    - أجب باللغة العربية فقط
    - {word_limit_text}
    - استخدم النص كمصدر أساسي
    - يمكنك إضافة معلومات عامة للمقارنة أو التحليل **إذا طلب السؤال ذلك**
    - إن كانت الإجابة تتضمن ذكر دليل شرعي، تحقق من الآية أو الحديث قبل كتابة الإجابة"""
            
        if abstention:
            instructions += """
    - **أو** إذا كنت لا تعلم الإجابة، اكتب: [لا أعلم]
    """
            
        if show_cot:
            instructions += """
    - فكر بتسلسل منطقي قبل الإجابة
    - اعرض تفكيرك بين <تفكير>...</تفكير> قبل كتابة الإجابة"""

        # Add verbalized elicitation (confidence request) for COMP/KNOW
        if verbalized_elicitation and question_type in ("COMP", "KNOW"):
            instructions += """
    - اكتب درجة ثقتك بإجابتك من 0% إلى 100%"""

    # ========================================================================
    # CONSTRUCT FINAL PROMPT
    # ========================================================================
    prompt_parts = [
        persona,
        task_clarification,
        few_shot_examples,
        question_section,
        instructions
    ]
    
    # Filter out empty parts and join
    prompt = "\n\n".join([part for part in prompt_parts if part.strip()])
    
    # print("prompt")
    # print(prompt.strip())
    
    return prompt.strip()


def generate_prompt_jais(
    question: Optional[str] = None,
    question_type: str = "MCQ",
    **kwargs) -> str:
    """
    Generate prompt for JAIS model with special formatting.
    
    Args:
        question: The question text
        question_type: "MCQ", "KNOW", or "COMP"
        **kwargs: All other arguments passed to generate_prompt (choice1-4, text, discipline, etc.)
    
    Returns:
        Formatted prompt string with JAIS-specific wrapper
    """
    standard_prompt = generate_prompt(
        question=question,
        question_type=question_type,
        **kwargs
    )
    return f"""
### Instruction: أكمل المحادثة بين [|Human|] و[|AI|]
### Input:[|Human|]
{standard_prompt}
[|AI|]
### Response :
"""

# ========================================================================
# EVALUATION PROMPT GENERATION
# ========================================================================

def generate_evaluation_prompt_granular(    
    question_type: str = "COMP",
    text: Optional[str] = None,
    question: Optional[str] = None,
    correct_answer: Optional[Union[str, Dict[str, Any]]] = None,
    prediction: Optional[str] = None,
    discipline: Optional[str] = None,
    no_requested_items: Optional[int] = None,
    verbose_instructions: bool = True,
    **kwargs) -> Optional[str]:
    return None

def generate_evaluation_prompt(    
    question_type: str = "COMP",
    text: Optional[str] = None,
    question: Optional[str] = None,
    correct_answer: Optional[Union[str, Dict[str, Any]]] = None,
    prediction: Optional[str] = None,
    discipline: Optional[str] = None,
    no_requested_items: Optional[int] = None,
    **kwargs) -> Optional[str]:
    return None