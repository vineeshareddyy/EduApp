# core/prompts.py - MERGED: Daily Standup + Weekly Interview + Weekend Mocktest
"""
Unified prompts module for all three modules:
- Daily Standup (direct technical questions from summary + extended web-based questions)
- Weekend Mocktest (question generation + evaluation)
- Weekly Interview (dynamic question generation + natural conversation + scoring)

ALL questions generated dynamically via LLM - NO hardcoded questions.
"""

from __future__ import annotations

from typing import List, Dict, Any
from .config import config
import json
import openai
from core.database import get_db_manager

# ---- Reusable boundary policy appended to Daily Standup prompts ----
BOUNDARY_POLICY = f"""
BOUNDARIES:
- Stay strictly on the CURRENT project/work topics in this conversation.
- If the user goes off-topic: give one brief, courteous redirect (≤ {getattr(config, 'REDIRECT_MAX_WORDS', 18)} words) and then ask ONE on-topic question.
- If the user uses vulgar/abusive language: do not repeat it; issue a short warning and restate the topic.
- After {getattr(config, 'MAX_VULGAR_STRIKES', 2)} warnings for vulgar language, end politely.
- Never generate sexual or hateful content. Tone is {getattr(config, 'BOUNDARY_TONE', 'calm, brief, professional')}.
""".strip()

def _append_boundaries(block: str) -> str:
    return f"{block.rstrip()}\n\n{BOUNDARY_POLICY}"


# =============================================================================
# RESPONSE PHRASES FOR DIFFERENT SITUATIONS (used across modules)
# =============================================================================

# When user gives wrong/unclear answer
WRONG_ANSWER_RESPONSES = [
    "No worries, that's okay.",
    "That's alright, let me ask it differently.",
    "No problem, try to explain a bit more clearly.",
    "That's fine, take your time.",
    "Okay, don't worry about that.",
    "That's okay, let's try another approach.",
    "No issues, we can move on.",
    "Alright, no problem at all.",
]

# When user is silent
SILENCE_ENCOURAGEMENT_RESPONSES = [
    "Take your time, there's no rush.",
    "I noticed you're quiet - would you like me to rephrase?",
    "No pressure, just share what comes to mind.",
    "It's okay to think out loud.",
    "Would you like to skip this one?",
    "Take a moment if you need.",
    "Feel free to ask if you need clarification.",
    "No worries, we can come back to this.",
]

# When user says "I don't know" / "can't answer" / "skip"
CANT_ANSWER_RESPONSES = [
    "No worries, let's move to a different question.",
    "That's perfectly fine, let me ask something else.",
    "No problem at all, here's another one.",
    "Okay, let's try a different topic.",
    "That's alright, moving on.",
    "No issues, let's continue with something else.",
]


# =============================================================================
# ██████╗  █████╗ ██╗██╗  ██╗   ██╗    ███████╗████████╗ █████╗ ███╗   ██╗██████╗ ██╗   ██╗██████╗
# ██╔══██╗██╔══██╗██║██║  ╚██╗ ██╔╝    ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██╔══██╗██║   ██║██╔══██╗
# ██║  ██║███████║██║██║   ╚████╔╝     ███████╗   ██║   ███████║██╔██╗ ██║██║  ██║██║   ██║██████╔╝
# ██║  ██║██╔══██║██║██║    ╚██╔╝      ╚════██║   ██║   ██╔══██║██║╚██╗██║██║  ██║██║   ██║██╔═══╝
# ██████╔╝██║  ██║██║███████╗██║       ███████║   ██║   ██║  ██║██║ ╚████║██████╔╝╚██████╔╝██║
# ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝       ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝
# =============================================================================

class DailyStandupPrompts:
    """Direct technical questions that match real standup style"""

    # ---------- Summary → chunk topics ----------
    @staticmethod
    def summary_splitting_prompt(summary: str) -> str:
        return f"""You're a curious person who wants to chat about this project work. Break it into {config.SUMMARY_CHUNKS} interesting topics.

PROJECT WORK:
{summary}

Think like you're genuinely interested:
- What sounds cool or challenging?
- What would you be curious about?
- What technical stuff catches your attention?
- What problems or solutions interest you?

Give me topics separated by '###CHUNK###' only."""

    # ---------- Base questions for a chunk ----------
    @staticmethod
    def base_questions_prompt(chunk_content: str) -> str:
        core = f"""You just read this project/work chunk:

    {chunk_content}

    TASK:
    Ask {config.BASE_QUESTIONS_PER_CHUNK} questions that show real technical curiosity about THIS chunk only.

    STRICT RULES:
    - Only ask questions directly related to THIS chunk/topic.
    - No personal or off-topic questions.
    - Vary phrasing so nothing feels repetitive.
    - Be professional, concise, human (not poetic).

    Mix question types:
    - Technical details / design choices
    - Challenges faced / trade-offs
    - Learnings / debugging insights
    - What's next / roadmap

    FORMAT:
    - Numbered list of unique questions (no answers)."""
        return _append_boundaries(core)

    # ---------- Follow-up analysis after user answered ----------
    @staticmethod
    def followup_analysis_prompt(chunk_content: str, user_response: str) -> str:
        core = f"""You asked about: "{chunk_content[:100]}..."

They replied: "{user_response}"

Put yourself in a real conversation. What would you naturally do?

If you're satisfied with their answer → say: COMPLETE
If you're still curious and would naturally ask more → create 1-2 follow-up questions

Be creative with follow-ups. Don't use standard boring questions. Think about what a real curious person would ask based on what they actually said.

FORMAT:
FOLLOWUP: [Your creative question]
FOLLOWUP: [Another creative one if needed]"""
        return _append_boundaries(core)

    # ---------- GREETING (directly ask about known domain, e.g., SAP) ----------
    @staticmethod
    def dynamic_greeting_response(*args, **kwargs) -> str:
        # --- argument normalization ---
        if len(args) >= 4:
            _, user_input, greeting_count, context = args[:4]
        elif len(args) >= 3:
            user_input = args[0]
            greeting_count = args[1]
            context = args[2]
        elif len(args) == 2:
            user_input, greeting_count = args
            context = {}
        else:
            user_input = kwargs.get("user_input", "")
            greeting_count = kwargs.get("greeting_count", 0)
            context = kwargs.get("context", {})

        # --- context normalization ---
        ctx = context or {}
        user_name = (ctx.get("user_name") or ctx.get("name") or "").strip()
        domain = (ctx.get("domain") or "").strip()

        # --- fetch latest summary topic from MongoDB ---
        try:
            db = get_db_manager()
            latest_summary = db._sync_get_summary()
            if latest_summary:
                first_line = latest_summary.split("\n")[0].strip()
                if len(first_line.split()) <= 6:
                    domain = first_line
        except Exception:
            pass

        # --- Check conversation phase ---
        user_input_lower = user_input.lower().strip()

        # **PHASE 1: Initial Greeting**
        if not user_input or user_input == "(session start)" or greeting_count == 0:
            time_of_day = ctx.get("time_of_day", "morning")
            if time_of_day == "morning":
                greeting_example = f"Good morning {user_name}! How are you today?"
            elif time_of_day == "afternoon":
                greeting_example = f"Good afternoon {user_name}! How are you doing?"
            elif time_of_day == "evening":
                greeting_example = f"Good evening {user_name}! How are you?"
            else:
                greeting_example = f"Hello {user_name}! How are you today?"

            core = f"""
        You are a friendly mentor starting a daily standup with {user_name}.

        TASK:
        - Greet them based on time of day: It's {time_of_day} now
        - Use appropriate greeting: "Good {time_of_day}" (e.g., "{greeting_example}")
        - Keep it warm, brief, and human
        - DO NOT ask about work topics yet
        - Max 15 words

        OUTPUT: One natural greeting sentence only.
        """
            return _append_boundaries(core)

        # **PHASE 2: User replied to greeting**
        positive_words = ["good", "great", "fine", "well", "okay", "ok", "nice", "alright", "yes", "yeah", "yep", "sure"]
        is_positive = any(word in user_input_lower for word in positive_words)

        negative_words = ["not", "bad", "no", "tired", "stressed", "low", "difficult", "tough"]
        is_negative = any(word in user_input_lower for word in negative_words)

        reciprocal_phrases = ["how about you", "what about you", "how are you", "how's your", "and you"]
        asked_reciprocal = any(phrase in user_input_lower for phrase in reciprocal_phrases)

        # **Handle reciprocal greeting**
        if asked_reciprocal:
            if is_negative:
                core = f"""
User said: "{user_input}"
They asked how you're doing but they seem low.

TASK:
- Show empathy
- Briefly say you're doing okay
- Reassure them about {domain} discussion
- Max 22 words

EXAMPLE: "I appreciate you asking — I'm doing well. Take your time, and we can discuss your {domain} updates whenever you're ready."

OUTPUT: One empathetic response.
"""
            else:
                core = f"""
User said: "{user_input}"
They asked how you're doing.

TASK:
- Answer briefly that you're doing great
- Transition to asking if they're ready for {domain} discussion
- Max 20 words

EXAMPLE: "I'm doing great, thanks for asking! Shall we discuss your {domain} updates?"

OUTPUT: One warm, transitional response.
"""
            return _append_boundaries(core)

        # **Handle emotional state**
        if is_negative:
            core = f"""
User replied: "{user_input}"
They seem low.

TASK:
- Respond with empathy
- Tell them you can discuss {domain} whenever comfortable
- Max 18 words

EXAMPLE: "I understand. We can go over your {domain} updates whenever you're ready."

OUTPUT: One empathetic response.
"""
            return _append_boundaries(core)

        elif is_positive:
            core = f"""
User replied: "{user_input}"
They're doing well.

TASK:
- Acknowledge warmly(DO NOT repeat "Good morning" or any greeting)
- Ask if ready to start {domain} discussion
- Max 18 words

EXAMPLE: "That's great! Shall we go over your {domain} updates?"

OUTPUT: One warm, transitional question.
"""
            return _append_boundaries(core)

        else:
            core = f"""
User replied: "{user_input}"

TASK:
- Acknowledge response
- Ask if ready to discuss {domain}
- Max 18 words

EXAMPLE: "Thank you! Are you ready to discuss your {domain} work today?"

OUTPUT: One transitional question.
"""
            return _append_boundaries(core)

    # ============================================================================
    # ✅ First Technical Question - DIRECT QUIZ STYLE
    # ============================================================================
    @staticmethod
    def generate_first_technical_question(concept_title: str, concept_content: str, user_greeting: str = "") -> str:
        """
        Generate DIRECT technical questions like real standup questions.
        Style: "What is X?", "How to do Y?", "Which command for Z?"
        """
        return f"""
    You are conducting a technical knowledge check about: {concept_title}

    SUMMARY CONTENT:
    {concept_content[:1500] if concept_content else "General technical discussion"}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ CRITICAL RULE: THE ANSWER MUST BE IN THE SUMMARY ABOVE ⚠️
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1 - IDENTIFY A FACT (not an instruction):
    ✅ FACTS (use these):
    - "RFC enables communication between SAP systems" → FACT
    - "AI algorithms analyze medical images" → FACT
    - "TensorFlow is a development platform" → FACT
    - "Transaction SM59 configures connections" → FACT
    - "Port 3320 is used for target system" → FACT

    ❌ INSTRUCTIONS (ignore these):
    - "Identify the specific problem you want to solve" → INSTRUCTION (no answer!)
    - "Gather relevant data" → INSTRUCTION (no answer!)
    - "Define the problem statement" → INSTRUCTION (no answer!)
    - "Choose the right model" → INSTRUCTION (no answer!)

    STEP 2 - TURN THE FACT INTO A QUESTION:
    Pattern: If summary says "X does Y", ask "What does X do?" → Answer: "Y"

    QUESTION PATTERNS:
    - "What is [term] used for?"
    - "What does [subject] do/analyze/enable?"
    - "Which [tool/transaction] is used for [task]?"
    - "What are the [requirements] for [topic]?"
    - "How does [subject] assist [object]?"

    STRICT RULES:
    1. Length: 8-15 words
    2. Answer MUST be stated in the summary
    3. NO "you/your": "what do you want", "in your project"
    4. NO hypotheticals: "what would you", "how would you"
    5. Direct quiz-style question only

    EXAMPLES BY DOMAIN:

    SAP Summary: "Transaction SM59 configures RFC connections"
    ✅ "What is transaction SM59 used for?" → Answer: "Configuring RFC connections"

    SAP Summary: "RFC enables communication between SAP systems"
    ✅ "What does RFC enable?" → Answer: "Communication between SAP systems"

    AI Summary: "AI algorithms analyze medical images to detect abnormalities"
    ✅ "What do AI algorithms analyze in healthcare?" → Answer: "Medical images"
    ❌ "What problem do you want AI to solve?" → NO ANSWER IN SUMMARY!

    AI Summary: "TensorFlow and PyTorch are AI development platforms"
    ✅ "What are examples of AI development platforms?" → Answer: "TensorFlow, PyTorch"

    AI Summary: "Computing resources with sufficient processing power are required"
    ✅ "What resources are required for AI training?" → Answer: "Computing resources"
    ❌ "What computing resources are you using?" → ASKS USER, NOT SUMMARY!

    General Summary: "Data sets are used for training and testing AI models"
    ✅ "What are data sets used for?" → Answer: "Training and testing AI models"
    ❌ "What data is relevant for your project?" → NO ANSWER IN SUMMARY!

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    VALIDATION: Before outputting, ask yourself:
    "Can someone answer this question by ONLY reading the summary above?"
    - YES → Output the question
    - NO → Find a different fact and try again
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    OUTPUT: One direct question (8-15 words):"""

    # ============================================================================
    # ✅ Technical Response (Summary-Grounded)
    # ============================================================================
    @staticmethod
    def dynamic_technical_response(context_text: str, user_input: str, next_question: str, session_state: dict = None) -> str:
        """
        Generate a context-aware, technical follow-up question grounded in the MongoDB summary.
        Keeps tone natural, progressive, and specific.
        """
        domain = (session_state or {}).get("domain", "technical discussion")

        core = f"""
You are a friendly but technically sharp mentor guiding the user through a {domain} discussion.

CONTEXT (Knowledge Source — from actual work summary):
{context_text[:2000] if context_text else "No summary available. Ask general technical questions."}

USER JUST SAID:
"{user_input}"

NEXT QUESTION PLAN:
"{next_question}"

GOAL:
- Continue a *technical standup* based on the CONTEXT above
- Ask about SPECIFIC tools, steps, or procedures mentioned in the context
- If the user answered well → move to the next specific operation from the context
- If the user was vague → ask a clarifying question about the same topic

STYLE RULES:
- Natural, human mentor tone — curious but concise
- One question only (12-20 words)
- Ask about SPECIFIC technical details from the context
- Avoid generic questions like "What area do you want to focus on?"
- Avoid meta language ("Let's talk about..." / "Can you tell me about...")
- Do not summarize or restate user answers
- Stay within the context provided

GOOD EXAMPLES (if context mentions SAP kernel upgrade):
- "After downloading the kernel, what's the verification step before applying it?"
- "When backing up the kernel directory, which files are most critical?"
- "What happens if you skip the sapcpe test after kernel extraction?"

BAD EXAMPLES:
- "What's your experience with kernel upgrades?" (too generic)
- "Tell me about SAP systems" (off-topic)
- "Let's discuss technical concepts" (meta language)

OUTPUT:
Write one short, technically specific question from the context (12-20 words).
"""
        return _append_boundaries(core)

    # ============================================================================
    # ✅ Follow-up Response - DIRECT QUIZ STYLE
    # ============================================================================
    @staticmethod
    def dynamic_followup_response(context_text: str, user_input: str,
                                previous_question: str, session_state: dict) -> str:
        """
        Generate direct follow-up questions in quiz style.
        """
        concept = session_state.get("concept", "the topic")
        questions_asked = session_state.get("questions_asked", 1)

        core = f"""
    You are conducting a technical knowledge check about: {concept}

    SUMMARY CONTENT:
    {context_text[:2000]}

    PREVIOUS QUESTION: "{previous_question}"
    USER'S ANSWER: "{user_input}"
    QUESTIONS ASKED: {questions_asked}/3

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ CRITICAL: THE ANSWER MUST BE IN THE SUMMARY CONTENT ABOVE ⚠️
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1 - IDENTIFY A FACT (not an instruction):
    ✅ FACTS (use these - they have answers):
    - "RFC enables communication" → FACT
    - "AI analyzes medical images" → FACT
    - "Port 3320 is used for target system" → FACT

    ❌ INSTRUCTIONS (ignore these - no answers):
    - "Identify the problem you want to solve" → INSTRUCTION
    - "Gather relevant data" → INSTRUCTION
    - "Define the problem statement" → INSTRUCTION

    STEP 2 - ASK ABOUT A DIFFERENT FACT THAN PREVIOUS QUESTION

    TASK:
    Generate ONE follow-up question about a DIFFERENT fact from the SAME SUMMARY CONTENT above.

    QUESTION STYLE (Match these exact patterns):
    - "What is [tool/command] used for?"
    - "How to [action]?"
    - "Which [tool/transaction] [does action]?"
    - "What command [does task]?"
    - "What is the meaning of [term]?"
    - "What happens when [scenario]?"
    - "What does [acronym] stand for?"
    - "What does [subject] analyze/enable/require?"

    STRICT RULES:
    1. Ask about a DIFFERENT detail than previous question
    2. THE ANSWER MUST BE IN SUMMARY CONTENT above
    3. Length: 10-18 words
    4. DIRECT style - no conversational phrases
    5. NO "you/your": "your project", "you are considering", "in your work"
    6. NO hypotheticals: "what would you", "how would you"
    7. Ask about specific tools, commands, transactions, ports, steps, facts

    GOOD EXAMPLES:

    SAP Summary:
    Previous Q: "What is RFC used for?"
    ✅ Follow-up: "Which transaction code monitors failed RFC calls?"

    Previous Q: "Which transaction applies support packages?"
    ✅ Follow-up: "What client is used for SPAM operations?"

    AI Summary:
    Previous Q: "What do AI algorithms analyze?"
    ✅ Follow-up: "Who does AI assist in diagnosing conditions?" → Answer: "Radiologists"

    Previous Q: "What platforms are used for AI development?"
    ✅ Follow-up: "What resources are required for AI training?" → Answer: "Computing resources"
    ❌ Follow-up: "What data is relevant for your project?" → NO ANSWER IN SUMMARY!

    Previous Q: "What are data sets used for?"
    ✅ Follow-up: "What are the two types of data sets needed?" → Answer: "Training and testing"
    ❌ Follow-up: "How would you split your data?" → ASKS USER OPINION!

    BAD EXAMPLES (too conversational - NEVER use):
    ❌ "How do you typically handle this in your environment?"
    ❌ "What's your experience with this concept?"
    ❌ "Can you explain more about your approach?"
    ❌ "What problem are you trying to solve?"
    ❌ "What tools are you considering?"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    VALIDATION: Before outputting, verify:
    "Can someone answer this by ONLY reading the SUMMARY CONTENT above?"
    - YES → Output the question
    - NO → Find a different fact and try again
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    NOW GENERATE:
    One direct follow-up question about "{concept}" from SUMMARY CONTENT (10-18 words, ANSWER IN CONTENT):"""

        return core

    # ============================================================================
    # ✅ Concept Transition - DIRECT QUIZ STYLE
    # ============================================================================
    @staticmethod
    def dynamic_concept_transition(current_concept: str, next_concept: str,
                                user_last_answer: str, next_concept_content: str) -> str:
        """
        Generate direct technical questions WITHOUT transition prefixes.
        """
        # Clean up concept titles (remove markdown formatting)
        next_concept_clean = next_concept.replace('**', '').replace('*', '').strip()
        if next_concept_clean.endswith(':'):
            next_concept_clean = next_concept_clean[:-1]

        core = f"""
    You are asking technical questions in a natural conversation flow.

    Previous topic: {current_concept}
    Current topic: {next_concept_clean}
    User's last answer: {user_last_answer}

    TOPIC CONTENT:
    {next_concept_content[:1500]}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ CRITICAL: THE ANSWER MUST BE IN THE TOPIC CONTENT ABOVE ⚠️
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1 - IDENTIFY A FACT (not an instruction):
    ✅ FACTS (use these - they have answers):
    - "RFC enables communication between SAP systems" → FACT
    - "AI algorithms analyze medical images" → FACT
    - "Transaction SM59 configures connections" → FACT
    - "TensorFlow is a development platform" → FACT

    ❌ INSTRUCTIONS (ignore these - no answers):
    - "Identify the specific problem you want to solve" → INSTRUCTION
    - "Gather relevant data" → INSTRUCTION
    - "Define the problem statement" → INSTRUCTION
    - "Choose the right model" → INSTRUCTION
    - Any sentence that tells you what to DO (imperative verb)

    STEP 2 - TURN THE FACT INTO A QUESTION:
    Pattern: Summary says "X does Y" → Ask "What does X do?" → Answer: "Y"

    TASK:
    Generate ONE direct technical question about "{next_concept_clean}" based on a FACT in the content above.

    CRITICAL RULES:
    1. THE ANSWER MUST BE STATED IN THE TOPIC CONTENT
    2. DO NOT use transition phrases like "Next topic", "Moving on", "Let's continue"
    3. Just ask the question directly
    4. NO "you/your" questions: "what do you want", "in your project", "your experience"
    5. NO hypotheticals: "what would you", "how would you"
    6. Keep it 10-15 words

    QUESTION PATTERNS (use these):
    - "What is [concept] used for?"
    - "How does [tool] work?"
    - "Which transaction code handles [task]?"
    - "What does [acronym] stand for?"
    - "What are the main components of [system]?"
    - "What does [subject] analyze/enable/require?"

    GOOD EXAMPLES:

    SAP Summary: "sending system waits for acknowledgment in synchronous RFC"
    ✅ "In synchronous RFC, does the sending system wait for acknowledgment?"

    SAP Summary: "three methods: Front End, Application Server, EPS Inbox"
    ✅ "What are the three methods for loading support packages?"

    SAP Summary: "transaction code ST11 for log trace files"
    ✅ "Which transaction code is used to access RFC error logs?"

    AI Summary: "AI algorithms analyze medical images to detect abnormalities"
    ✅ "What do AI algorithms analyze in healthcare?" → Answer: "Medical images"
    ❌ "What problem do you want AI to solve?" → NO ANSWER IN CONTENT!

    AI Summary: "TensorFlow and PyTorch are AI development platforms"
    ✅ "What are examples of AI development platforms?" → Answer: "TensorFlow, PyTorch"
    ❌ "What tools are you considering?" → ASKS USER, NOT CONTENT!

    AI Summary: "Computing resources with sufficient processing power are required"
    ✅ "What resources are required for AI training?" → Answer: "Computing resources"
    ❌ "What computing resources do you have?" → NO ANSWER IN CONTENT!

    BAD EXAMPLES (NEVER do this):
    ❌ "Next topic. What is..." (transition phrase)
    ❌ "Moving on. How does..." (transition phrase)
    ❌ "What do you want to achieve?" (no answer in content)
    ❌ "What is your approach to..." (asks user opinion)
    ❌ "How would you implement..." (hypothetical)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    VALIDATION: Before outputting, verify:
    "Can someone answer this by ONLY reading the TOPIC CONTENT above?"
    - YES → Output the question
    - NO → Find a different fact and try again
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    NOW GENERATE:
    Ask ONE direct question about "{next_concept_clean}" (10-15 words, NO transition prefix, ANSWER IN CONTENT):"""

        return core

    # ---------- Fragment evaluation ----------
    @staticmethod
    def dynamic_fragment_evaluation(concepts_covered: List[str], conversation_exchanges: List[Dict],
                                    session_stats: Dict) -> str:
        concepts_text = "\n".join([f"- {concept}" for concept in concepts_covered])

        conversation_summary = []
        for exchange in conversation_exchanges[-6:]:
            conversation_summary.append(
                f"Q: {exchange['ai_message'][:80]}...\n"
                f"A: {exchange['user_response'][:80]}...\n"
            )
        conversation_text = "\n".join(conversation_summary)

        core = f"""You're evaluating a technical standup.

    SESSION METRICS:
    - Topics covered: {session_stats['concepts_covered']}/{session_stats['total_concepts']} ({session_stats['coverage_percentage']}%)
    - Duration: {session_stats['duration_minutes']} minutes
    - Main questions: {session_stats['main_questions']}, Follow-ups: {session_stats['followup_questions']}

    TOPICS:
    {concepts_text}

    RECENT EXCHANGES:
    {conversation_text}

    TASK:
    Score ONLY these categories:
    - Communication (clarity, flow): 0–2
    - Confidence (tone, assertiveness): 0–2
    - Technical (topics covered & accuracy): 0–6

    Then write short feedback (≤120 words) that is concrete and helpful.

    OUTPUT FORMAT (exactly):
    COMMUNICATION: X/2
    CONFIDENCE: X/2
    TECHNICAL: X/6
    TOTAL: Y/10
    FEEDBACK: [short, human, specific feedback]"""
        return _append_boundaries(core)

    # ---------- Session completion ----------
    @staticmethod
    def dynamic_session_completion(conversation_summary: Dict, user_final_response: str = None) -> str:
        topics_discussed = conversation_summary.get('topics_covered', [])
        total_exchanges = conversation_summary.get('total_exchanges', 0)

        core = f"""You're ending a good standup chat with your teammate.

    **CHAT SUMMARY:**
    - Talked about: {len(topics_discussed)} different topics
    - Total questions: {total_exchanges}
    - Their final words: "{user_final_response}"

    **GOAL**: Give ONE short, natural closing line that includes a brief thanks and ends the session.

    **STYLE**
    - Very short (12–20 words), natural, human.
    - Must include "Thanks" or "Thank you".
    - No follow-up questions.
    - No bullets, no headings, no extra lines.

    **OUTPUT**
    Output exactly ONE sentence only, nothing else."""
        return _append_boundaries(core)

    # ---------- Clarification request ----------
    @staticmethod
    def dynamic_clarification_request(context: Dict) -> str:
        attempts = context.get('clarification_attempts', 0)
        core = f"""You need them to speak more clearly.

**SITUATION**: You've asked for clarity {attempts} times already.

**BE CREATIVE**: Ask for clarification in a different way each time. Don't use the same boring phrases.

Make it:
- Natural and friendly
- Different from previous attempts
- Not repetitive or annoying
- Understanding and patient

One creative sentence. Make it feel real."""
        return _append_boundaries(core)

    # ---------- Gentle conclusion response ----------
    @staticmethod
    def dynamic_conclusion_response(user_input: str, session_context: Dict) -> str:
        core = f"""They just said: "{user_input}"

You're wrapping up the chat about their work.

**BE CREATIVE**: Respond to what they said and end naturally. Don't use boring standard endings.

Make it:
- Personal to what they shared
- Appreciative of their time
- Natural like a real conversation ending
- Unique and genuine

Max 20 words. Be original every time."""
        return _append_boundaries(core)

    # ---------- OFF-TOPIC redirect ----------
    @staticmethod
    def boundary_offtopic_prompt(topic: str, subtask: str = "") -> str:
        ask = f"What progress since yesterday on {subtask or topic}?"
        core = f"""User is off-topic (e.g., talking about unrelated things like movies, sports, random analogies).
    TASK:
    - Do NOT follow the off-topic content.
    - Politely redirect in one short line (≤ {getattr(config, 'REDIRECT_MAX_WORDS', 18)} words).
    - Immediately ask ONE question about THIS topic: {ask}.

    STYLE:
    - Professional, concise, interview-focused.
    - Never expand on or question the off-topic subject.
    - Always bring the user back to the interview topic."""
        return _append_boundaries(core)

    # ---------- Generate main question from content ----------
    @staticmethod
    def generate_main_question_from_content(fragment_details: dict) -> str:
        """Generate question from specific fragment content."""
        title = fragment_details.get('title', '')
        content = fragment_details.get('content', '')
        key_terms = fragment_details.get('key_terms', [])

        return f"""
    You are asking a technical question from this training content:

    SECTION: {title}
    CONTENT:
    {content[:1200]}

    KEY TERMS: {', '.join(key_terms[:5]) if key_terms else 'None'}

    TASK: Generate ONE direct question about a SPECIFIC FACT from the CONTENT.

    RULES:
    1. Question MUST be answerable from CONTENT above
    2. Ask about FIRST key concept/definition
    3. Length: 8-15 words
    4. Use EXACT terms from content
    5. Pattern: "What is [X] used for?" / "What does [X] do?" / "Which [X] is used for [Y]?"

    EXAMPLES:
    Content: "RFCs enable communication between SAP systems"
    → "What do RFCs enable in SAP systems?"

    Content: "Transaction SM59 configures RFC connections"
    → "What is transaction SM59 used for?"

    Content: "There are four types: aRFC, sRFC, tRFC, qRFC"
    → "How many types of RFCs are there?"

    Generate ONE question about "{title}":
    """

    # ---------- Generate example question ----------
    @staticmethod
    def generate_example_question(fragment_details: dict, previous_question: str) -> str:
        """Generate follow-up asking for example."""
        title = fragment_details.get('title', '')

        return f"""
    User just answered a question about: {title}
    Previous question: "{previous_question}"

    The content has an example for this topic.

    TASK: Ask them to provide an example. Keep it SHORT (max 10 words).

    OUTPUT one of these (or similar):
    - "Can you give me an example of that?"
    - "Could you provide an example?"
    - "What would be a practical example?"

    Generate:
    """

    # ============================================================================
    # ✅ Web-based extended question generation
    # ============================================================================
    @staticmethod
    def generate_extended_web_question(topic: str, already_asked: List[str], summary_context: str) -> str:
        """
        Generate question that requires a DIFFERENT answer than all previous questions.
        """
        if already_asked and len(already_asked) > 0:
            questions_list = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(already_asked)])
            questions_count = len(already_asked)
        else:
            questions_list = "  None yet"
            questions_count = 0

        return f"""You are a technical interviewer. Generate ONE new question about: **{topic}**

    REFERENCE MATERIAL:
    {summary_context[:800] if summary_context else f"General knowledge about {topic}"}

    ════════════════════════════════════════════════════════════════════
    QUESTIONS ALREADY ASKED ({questions_count} total):
    {questions_list}
    ════════════════════════════════════════════════════════════════════

    ⚠️ CRITICAL RULE: Your question MUST require a DIFFERENT ANSWER!

    🚫 SAME ANSWER = DUPLICATE (NOT ALLOWED):

    These are ALL duplicates because they have the SAME answer:
    ❌ "risks of client administration" = "risks of client management" (administration = management)
    ❌ "best practices for securing" = "best practices for securing against unauthorized access"
    ❌ "implications of client settings" = "implications of configuration changes" (settings = configuration)
    ❌ "types of client copies" = "methods to transfer data" (same list: local, remote, export)
    ❌ "steps to do X" = "how to perform X" = "process for X" (all asking procedure)
    ❌ "what is X" = "purpose of X" = "X is used for" (all asking definition)

    ✅ DIFFERENT ANSWER = ALLOWED:

    These are DIFFERENT because answers are different:
    ✅ "what is X" vs "how to configure X" → Different!
    ✅ "types of X" vs "problems with X" → Different!
    ✅ "prerequisites" vs "best practices" → Different!
    ✅ "security considerations" vs "performance implications" → Different!

    ════════════════════════════════════════════════════════════════════

    INSTRUCTIONS:
    1. Read EVERY question above carefully
    2. Think: "What would be the ANSWER to my new question?"
    3. Check: "Does ANY previous question have this SAME answer?"
    4. If YES → Think of something else
    5. If NO → Generate it

    Generate ONE question (10-18 words) requiring a GENUINELY DIFFERENT answer:"""

    # ============================================================================
    # ✅ Smart answer validation
    # ============================================================================
    @staticmethod
    def smart_answer_validation_prompt(question: str, answer: str, summary_context: str = "") -> str:
        """
        Validate if an answer is correct using general knowledge.
        - IRRELEVANT = doesn't answer the question asked (wrong type OR off-topic)
        - INCORRECT = factually wrong but tries to answer the right thing
        - CORRECT/PARTIAL = addresses the question properly
        """
        return f"""You are evaluating a student's answer to a technical question.

    QUESTION: "{question}"
    STUDENT'S ANSWER: "{answer}"

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ MOST IMPORTANT CHECK: Does the answer ADDRESS what the question asks?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1 - IDENTIFY QUESTION TYPE:
    - "What is X?" / "Define X" → Expects DEFINITION/DESCRIPTION
    - "What are the types/kinds of X?" → Expects LIST OF CATEGORIES
    - "What is X USED FOR?" / "Purpose of X?" → Expects PURPOSE/FUNCTION
    - "What are ISSUES/PROBLEMS with X?" → Expects PROBLEMS/CHALLENGES
    - "What are PREREQUISITES for X?" → Expects REQUIREMENTS/SKILLS
    - "HOW to do X?" / "Steps for X?" → Expects PROCESS/STEPS
    - "WHY X?" → Expects REASONS/CAUSES
    - "WHEN to use X?" → Expects CONDITIONS/SCENARIOS

    STEP 2 - CHECK IF ANSWER MATCHES QUESTION TYPE:

    🚫 IRRELEVANT - Mark if ANY of these are true:
    - Answer is about a COMPLETELY DIFFERENT topic (off-topic)
    - Answer gives PREREQUISITES when asked for PURPOSE
    - Answer gives PURPOSE when asked for ISSUES/PROBLEMS
    - Answer gives DEFINITION when asked for STEPS/HOW-TO
    - Answer gives TYPES when asked for SPECIFIC EXAMPLE
    - Answer describes SKILLS when asked for FUNCTION/USE
    - Answer talks about WHAT something IS when asked WHAT IT'S USED FOR
    - Basically: ANSWER TYPE ≠ QUESTION TYPE

    ❌ INCORRECT - Mark ONLY if:
    - Answer TRIES to address the question correctly BUT contains factually wrong information
    - Example: Q: "What are standard SAP clients?" A: "100, 200, 300" (wrong numbers, but tried to answer)

    ⚠️ PARTIAL - Mark if:
    - Answer addresses the correct question type but is incomplete
    - Answer is too brief (under 15 words) but on the right track
    - Answer mentions some correct points but misses key ones

    ✅ CORRECT - Mark if:
    - Answer addresses EXACTLY what the question asked
    - Answer is factually accurate
    - Answer shows clear understanding

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    CRITICAL EXAMPLES:

    Question: "What are clients USED FOR in SAP?" (asks PURPOSE)
    Answer: "Clients require SAP basis knowledge, security skills, system understanding"
    → IRRELEVANT (gave PREREQUISITES/SKILLS, not PURPOSE. This doesn't answer what was asked!)

    Question: "What are clients USED FOR in SAP?" (asks PURPOSE)
    Answer: "Clients are used to separate development, testing, and production environments"
    → CORRECT (gives PURPOSE as asked)

    Question: "What are ISSUES during client administration?" (asks PROBLEMS)
    Answer: "Client administration includes client copy, data subsetting, transport"
    → IRRELEVANT (gave TYPES/TASKS, not ISSUES/PROBLEMS. Doesn't answer the question!)

    Question: "What are ISSUES during client administration?" (asks PROBLEMS)
    Answer: "Common issues are long runtime, locked tables, authorization errors"
    → CORRECT (gives PROBLEMS as asked)

    Question: "What are PREREQUISITES for SAP installation?" (asks REQUIREMENTS)
    Answer: "Prerequisites include disk space, correct OS, RAM, JDK"
    → CORRECT (gives REQUIREMENTS as asked)

    Question: "What are PREREQUISITES for SAP installation?" (asks REQUIREMENTS)
    Answer: "SAP is used for enterprise resource planning"
    → IRRELEVANT (gave PURPOSE/DEFINITION, not PREREQUISITES)

    Question: "What are the standard clients in SAP?" (asks SPECIFIC LIST)
    Answer: "000, 001, and 066"
    → CORRECT (correct list)

    Question: "What are the standard clients in SAP?" (asks SPECIFIC LIST)
    Answer: "100, 200, 300"
    → INCORRECT (tried to answer correctly but wrong numbers - factually wrong)

    Question: "What are the THREE types of data in SAP clients?" (asks for 3 items)
    Answer: "User master data and application data"
    → PARTIAL (only gave 2 of 3, incomplete but correct type of answer)

    Question: "What is SAP Basis?"
    Answer: "My favorite food is pizza"
    → IRRELEVANT (completely off-topic)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    NOW EVALUATE:
    Question: "{question}"
    Answer: "{answer}"

    DECISION PROCESS:
    1. What TYPE of information does the question ask for?
    2. What TYPE of information did the answer provide?
    3. Do they MATCH?
    - NO → IRRELEVANT (answer doesn't address what was asked)
    - YES → Check if factually correct:
        - Factually WRONG → INCORRECT
        - Factually correct but incomplete → PARTIAL
        - Factually correct and complete → CORRECT

    Respond with ONLY ONE word:
    - CORRECT
    - PARTIAL
    - INCORRECT
    - IRRELEVANT

    Your evaluation:"""

    # ============================================================================
    # ✅ Formal session closing
    # ============================================================================
    @staticmethod
    def formal_session_closing(session_context: Dict[str, Any]) -> str:
        """
        Generate a formal, interview-style closing message for the standup session.
        """
        name = session_context.get("name", "")
        topics_covered = session_context.get("topics_covered", [])
        total_questions = session_context.get("total_questions", 0)
        duration = session_context.get("duration_minutes", 15)

        topics_text = ", ".join(topics_covered[:5]) if topics_covered else "various technical topics"

        return f"""
Generate a FORMAL, PROFESSIONAL closing message for this technical standup session.

SESSION SUMMARY:
- Candidate: {name}
- Duration: {duration:.1f} minutes
- Topics Covered: {topics_text}
- Total Questions: {total_questions}

REQUIREMENTS:
1. Thank the candidate professionally
2. Mention this was a productive session
3. Indicate the session is now complete
4. Keep it formal but warm (like an interviewer ending an interview)
5. Maximum 25 words
6. Do NOT ask any follow-up questions

STYLE EXAMPLES:
- "Great session, {name}. Thank you for your time today. We've covered good ground. We'll connect again in our next standup."
- "That concludes our standup for today, {name}. Thank you for walking me through your work. See you in the next session."
- "Excellent, {name}. That's all for today's standup. I appreciate your responses. We'll reconnect in the next session."

Generate ONE formal closing message (max 25 words):"""

    # ---------- Silence response (hardcoded multi-level) ----------
    @staticmethod
    def dynamic_silence_response(session_context: Dict) -> str:
        """
        Return HARDCODED silence prompts with slight variations.
        No LLM generation - just return the text directly.
        """
        name = session_context.get("name", "")
        silence_count = session_context.get("silence_count", 1)

        import random

        # Level 1: Gentle check-in
        if silence_count == 1:
            options = [
                f"{name}, are you there? I'd love to hear from you.?",
                f"Hey {name},just checking — are you still with me?",
                f"{name}, are you there? I'm ready whenever you are?",
            ]
            return random.choice(options)

        # Level 2: Connection check
        elif silence_count == 2:
            options = [
                f"Hello? {name}? Can you hear me?",
                f"{name}, are you still there?",
                f"{name}, can you hear me? Just checking the connection.",
            ]
            return random.choice(options)

        # Level 3: Offering help
        elif silence_count == 3:
            options = [
                f"Are you still there, {name}? Let me know if you need me to repeat the question",
                f"{name}, I'm here when you're ready. Need me to repeat anything?",
                f"Still nothing from your side, {name}. Need me to repeat anything?",
            ]
            return random.choice(options)

        # Level 4: Giving space
        elif silence_count == 4:
            options = [
                f"{name}, I'm still here. Take your time - I'll wait.",
                f"No rush, {name}. I'll wait - just let me know when you're ready.",
                f"Still here, {name}. Take all the time you need.",
            ]
            return random.choice(options)

        # Level 5+: Graceful wrap-up
        else:
            options = [
                f"I notice you're not responding, {name}. Let's wrap up for now. We can continue this another time.",
                f"Seems like you're still silent,{name}. I'll wrap up for now. Catch you later!",
                f"Haven't heard from you, {name}. I'll end our session now. Thanks!",
            ]
            return random.choice(options)

    # ---------- VULGARITY handling ----------
    @staticmethod
    def boundary_vulgar_prompt(topic: str) -> str:
        core = f"User used inappropriate language. Give ONE short warning and restate the topic: {topic}. Do not repeat the language."
        return _append_boundaries(core)

    # ---------- Quick redirect helpers ----------
    @staticmethod
    def off_topic_redirect(topic: str, subtask: str = "") -> str:
        if subtask:
            return f"Let's keep this about {topic}. What's the status of {subtask}?"
        return f"Let's keep this about {topic}. What progress did you make since yesterday?"

    @staticmethod
    def off_topic_firm(topic: str, subtask: str = "") -> str:
        if subtask:
            return f"We need to stay on {topic}. What blockers are you facing on {subtask}?"
        return f"We need to stay on {topic}. Any blockers or progress since yesterday?"

    @staticmethod
    def off_topic_move_on(next_topic: str) -> str:
        return f"I'll move to the next item: {next_topic}. What changed since last update?"

    @staticmethod
    def vulgar_warning_1(topic: str) -> str:
        return f"Let's keep language respectful. Can you summarize your update on {topic}?"

    @staticmethod
    def vulgar_warning_2(topic: str) -> str:
        return f"This needs to stay respectful. Last chance—please share your update on {topic}."

    @staticmethod
    def end_due_to_vulgarity() -> str:
        return "I'm ending this standup due to repeated inappropriate language. We can resume when it's respectful."

    @staticmethod
    def refuse_nsfw_and_redirect(topic: str) -> str:
        return f"I can't discuss that. Let's focus on your {topic} update: progress, blockers, next steps?"

    @staticmethod
    def harassment_block_and_redirect(topic: str) -> str:
        return f"That language isn't okay here. Please share your concrete update on {topic}—progress, blockers, next steps."

    # ---------- Auto-advance helper ----------
    @staticmethod
    def dynamic_auto_advance(ctx: Dict[str, Any]) -> str:
        current_topic = ctx.get("current_topic", "your work")
        next_topic = ctx.get("next_topic", "the next item")
        name = ctx.get("name", "")
        core = f"""User remained silent or was unclear.

TASK:
- Move forward politely in ≤18 words.
Examples:
"Alright {name}, let's move on to {next_topic}."
"Let's continue with {next_topic}, {name}."
"""
        return _append_boundaries(core)

    # ---------- Hard cutoff closure helper ----------
    @staticmethod
    def dynamic_hardcutoff_closure(summary: Dict[str, Any]) -> str:
        topics = ", ".join(summary.get("topics_covered", [])) or "various topics"
        name = summary.get("name", "participant")
        core = f"""Time limit reached.

Topics covered: {topics}

TASK:
- Close politely in ≤18 words with thanks.
Examples:
"Thanks {name}, we're out of time but this was productive."
"Appreciate it {name}; we'll continue next time."
"""
        return _append_boundaries(core)

    # ============================================================================
    # COMPREHENSIVE EVALUATION PROMPTS (Daily Standup)
    # ============================================================================
    @staticmethod
    def comprehensive_evaluation_prompt(session_data: dict) -> str:
        """
        Generate a comprehensive evaluation prompt that uses general knowledge
        to evaluate answers, not just the summary content.
        """
        conversation_text = ""
        for idx, exchange in enumerate(session_data.get("conversation", []), 1):
            q = exchange.get("question", "")
            a = exchange.get("answer", "")
            response_type = exchange.get("response_type", "answered")
            concept = exchange.get("concept", "unknown")

            conversation_text += f"""
    Question {idx} (Concept: {concept}):
    Q: {q}
    A: {a}
    Response Type: {response_type}
    ---
    """

        stats = session_data.get("stats", {})

        prompt = f"""You are an expert technical interviewer evaluating a candidate's performance in a Daily Standup assessment.

    ## Session Information
    - Candidate Name: {session_data.get("student_name", "Unknown")}
    - Duration: {session_data.get("duration_minutes", 0):.1f} minutes
    - Total Questions Asked: {stats.get("total_questions", 0)}
    - Questions Answered: {stats.get("answered_count", 0)}
    - Questions Skipped: {stats.get("skipped_count", 0)}
    - Silent Responses: {stats.get("silent_count", 0)}
    - Irrelevant Answers: {stats.get("irrelevant_count", 0)}
    - Repeat Requests: {stats.get("repeat_requests_count", 0)}

    ## Conversation Log
    {conversation_text}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ CRITICAL EVALUATION RULE:
    Use YOUR GENERAL KNOWLEDGE to evaluate answers, NOT just training material!

    Accept answers that are:
    - Correct according to general/industry knowledge
    - Alternative valid definitions
    - Different wording but same meaning
    - Partially correct with key concepts right

    Only mark incorrect if FACTUALLY WRONG.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ## Your Task
    Analyze this standup session and provide a COMPREHENSIVE evaluation in the following JSON format:

    {{
        "overall_score": <number 0-100>,
        "technical_score": <number 0-100>,
        "communication_score": <number 0-100>,
        "attentiveness_score": <number 0-100>,

        "grade": "<A+/A/A-/B+/B/B-/C+/C/C-/D/F>",

        "summary": "<2-3 sentence overall summary>",

        "strengths": [
            "<strength 1>",
            "<strength 2>",
            "<strength 3>"
        ],

        "weaknesses": [
            "<weakness 1>",
            "<weakness 2>"
        ],

        "areas_for_improvement": [
            "<specific actionable improvement 1>",
            "<specific actionable improvement 2>",
            "<specific actionable improvement 3>"
        ],

        "question_analysis": [
            {{
                "question_number": 1,
                "question": "<the question>",
                "answer": "<the answer>",
                "concept": "<concept tested>",
                "evaluation": "correct|partial|incorrect|skipped|irrelevant|silent",
                "score": <0-10>,
                "feedback": "<specific feedback for this answer>",
                "used_alternative_definition": <true/false>
            }}
        ],

        "attentiveness_analysis": {{
            "engagement_level": "<High/Medium/Low>",
            "response_consistency": "<Consistent/Inconsistent>",
            "focus_areas": "<areas where candidate showed good focus>",
            "distraction_indicators": "<any signs of distraction or disengagement>"
        }},

        "recommendations": [
            "<recommendation 1>",
            "<recommendation 2>",
            "<recommendation 3>"
        ],

        "topics_mastered": ["<topic 1>", "<topic 2>"],
        "topics_to_review": ["<topic 1>", "<topic 2>"]
    }}

    ## Evaluation Guidelines:
    1. **Correct Answer**: Answer is factually correct (by ANY valid definition) - 8-10 points
    2. **Partial Answer**: Some understanding but incomplete - 4-7 points
    3. **Incorrect Answer**: Factually wrong response - 1-3 points
    4. **Skipped**: Candidate explicitly skipped - 0 points
    5. **Irrelevant**: Answer was off-topic - 0 points
    6. **Silent**: No response - 0 points

    REMEMBER: Accept alternative correct definitions! Different wording ≠ wrong answer.

    Respond ONLY with the JSON object, no additional text.
    """
        return prompt

    @staticmethod
    def per_question_evaluation_prompt(question: str, answer: str, concept: str, response_type: str) -> str:
        """
        Evaluate a single Q&A pair using BOTH summary knowledge AND general knowledge.
        Accepts correct answers even if they're not exactly in the summary.
        """
        prompt = f"""Evaluate this technical Q&A from a standup assessment:

    **Concept Being Tested:** {concept}
    **Question:** {question}
    **Candidate's Answer:** {answer}
    **Response Type:** {response_type}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️ IMPORTANT: Use GENERAL KNOWLEDGE to evaluate, not just training material!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    EVALUATION RULES:
    1. Accept answers that are CORRECT according to general/web knowledge
    2. Accept ALTERNATIVE definitions that are factually accurate
    3. Accept answers with DIFFERENT WORDING but same meaning
    4. Only mark incorrect if FACTUALLY WRONG
    5. Consider PARTIAL credit for incomplete but correct answers

    SCORING GUIDE:
    - 9-10: Excellent, comprehensive answer (correct by any valid definition)
    - 7-8: Good answer with minor gaps
    - 5-6: Acceptable, shows understanding
    - 3-4: Poor but shows some knowledge
    - 1-2: Very weak, mostly incorrect
    - 0: No answer/completely wrong/irrelevant

    Provide evaluation in this exact JSON format:
    {{
        "is_correct": <true/false>,
        "correctness_level": "<correct|partial|incorrect|skipped|irrelevant|silent>",
        "score": <0-10>,
        "key_points_covered": ["<point1>", "<point2>"],
        "key_points_missed": ["<point1>", "<point2>"],
        "feedback": "<specific constructive feedback, max 2 sentences>",
        "accepted_as_alternative": <true/false if answer is correct but uses different wording>
    }}

    EXAMPLES:

    Question: "What is a programming language?"
    Answer: "It's a way to give instructions to computers using specific syntax and rules"
    → score: 9, is_correct: true, accepted_as_alternative: true
    (Valid alternative definition)

    Question: "What does RFC stand for in SAP?"
    Answer: "Remote Function Call"
    → score: 10, is_correct: true
    (Correct answer)

    Question: "What is machine learning?"
    Answer: "Computers learning patterns from data without explicit programming"
    → score: 9, is_correct: true, accepted_as_alternative: true
    (Valid general knowledge definition)

    Respond ONLY with JSON.
    """
        return prompt

    @staticmethod
    def generate_strengths_weaknesses_prompt(evaluated_questions: list, stats: dict) -> str:
        """
        Generate a prompt to analyze strengths and weaknesses from evaluated questions.
        """
        questions_summary = ""
        for q in evaluated_questions:
            questions_summary += f"- {q.get('concept', 'Unknown')}: Score {q.get('score', 0)}/10 ({q.get('correctness_level', 'unknown')})\n"

        prompt = f"""Based on this standup performance summary, identify strengths and weaknesses:

    ## Performance by Concept:
    {questions_summary}

    ## Statistics:
    - Answered: {stats.get('answered_count', 0)}
    - Skipped: {stats.get('skipped_count', 0)}
    - Silent: {stats.get('silent_count', 0)}
    - Irrelevant: {stats.get('irrelevant_count', 0)}

    Provide analysis in JSON format:
    {{
        "strengths": [
            {{"area": "<strength area>", "evidence": "<why this is a strength>"}},
            {{"area": "<strength area>", "evidence": "<why this is a strength>"}}
        ],
        "weaknesses": [
            {{"area": "<weakness area>", "evidence": "<why this is a weakness>", "improvement_tip": "<how to improve>"}},
            {{"area": "<weakness area>", "evidence": "<why this is a weakness>", "improvement_tip": "<how to improve>"}}
        ],
        "top_performing_topics": ["<topic1>", "<topic2>"],
        "needs_improvement_topics": ["<topic1>", "<topic2>"]
    }}

    Respond ONLY with JSON.
    """
        return prompt


# Backward compatibility aliases for daily_standup:
Prompts = DailyStandupPrompts
prompts = DailyStandupPrompts()

# Module-level function wrappers for evaluation prompts
def comprehensive_evaluation_prompt(session_data: dict) -> str:
    """Module-level wrapper for comprehensive evaluation prompt."""
    return DailyStandupPrompts.comprehensive_evaluation_prompt(session_data)

def per_question_evaluation_prompt(question: str, answer: str, concept: str, response_type: str) -> str:
    """Module-level wrapper for per-question evaluation prompt."""
    return DailyStandupPrompts.per_question_evaluation_prompt(question, answer, concept, response_type)

def generate_strengths_weaknesses_prompt(evaluated_questions: list, stats: dict) -> str:
    """Module-level wrapper for strengths/weaknesses prompt."""
    return DailyStandupPrompts.generate_strengths_weaknesses_prompt(evaluated_questions, stats)


# =============================================================================
# ██╗    ██╗███████╗███████╗██╗  ██╗███████╗███╗   ██╗██████╗
# ██║    ██║██╔════╝██╔════╝██║ ██╔╝██╔════╝████╗  ██║██╔══██╗
# ██║ █╗ ██║█████╗  █████╗  █████╔╝ █████╗  ██╔██╗ ██║██║  ██║
# ██║███╗██║██╔══╝  ██╔══╝  ██╔═██╗ ██╔══╝  ██║╚██╗██║██║  ██║
# ╚███╔███╔╝███████╗███████╗██║  ██╗███████╗██║ ╚████║██████╔╝
#  ╚══╝╚══╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═════╝
# ███╗   ███╗ ██████╗  ██████╗██╗  ██╗████████╗███████╗███████╗████████╗
# ████╗ ████║██╔═══██╗██╔════╝██║ ██╔╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
# ██╔████╔██║██║   ██║██║     █████╔╝    ██║   █████╗  ███████╗   ██║
# ██║╚██╔╝██║██║   ██║██║     ██╔═██╗    ██║   ██╔══╝  ╚════██║   ██║
# ██║ ╚═╝ ██║╚██████╔╝╚██████╗██║  ██╗   ██║   ███████╗███████║   ██║
# ╚═╝     ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚══════╝   ╚═╝
# =============================================================================

class PromptTemplates:
    """Optimized prompt templates for AI question generation and evaluation"""

    @staticmethod
    def create_batch_questions_prompt(user_type: str, context: str, question_count: int = None) -> str:
        if question_count is None:
            question_count = config.QUESTIONS_PER_TEST
        if user_type == "dev":
            return PromptTemplates._dev_batch_prompt(context, question_count)
        else:
            return PromptTemplates._non_dev_batch_prompt(context, question_count)

    @staticmethod
    def _dev_batch_prompt(context: str, question_count: int) -> str:
        return f"""Generate {question_count} high-quality programming questions based on the provided context. Create practical, challenging questions that test real development skills and problem-solving abilities.

CONTEXT:
{context}

REQUIREMENTS:
- Generate exactly {question_count} questions numbered sequentially
- Mix question types: 40% practical coding, 30% system design, 30% debugging/optimization
- Progressive difficulty: start easier, increase complexity
- Each question must be complete and standalone
- Include clear requirements, constraints, and expected outcomes
- Base questions on concepts and technologies mentioned in the context
- Make questions realistic and industry-relevant

FORMAT each question exactly as shown:
=== QUESTION 1 ===
## Title: [Clear, descriptive title]
## Difficulty: [Easy/Medium/Hard]
## Type: [Practical/Algorithm/System Design/Debugging]
## Question:
[Complete question with detailed requirements, constraints, input/output examples, and any code snippets needed. Include specific technical requirements and success criteria.]

=== QUESTION 2 ===
## Title: [Clear, descriptive title]
## Difficulty: [Easy/Medium/Hard]
## Type: [Practical/Algorithm/System Design/Debugging]
## Question:
[Complete question with detailed requirements...]

Continue this exact pattern for all {question_count} questions.

IMPORTANT:
- Each question should test different aspects of development
- Include code examples where relevant
- Specify performance requirements when applicable
- Make questions challenging but solvable by a competent developer
- Ensure questions relate to the provided context

Generate all {question_count} questions now:"""

    @staticmethod
    def _non_dev_batch_prompt(context: str, question_count: int) -> str:
        return f"""Generate {question_count} high-quality multiple-choice questions based on the provided context. Focus on conceptual understanding, analytical thinking, and practical application of technical concepts for non-technical professionals.

CONTEXT:
{context}

REQUIREMENTS:
- Generate exactly {question_count} questions numbered sequentially
- Each question must have exactly 4 options (A, B, C, D) with only 1 correct answer
- Mix question types: 40% conceptual understanding, 30% analytical reasoning, 30% practical application
- Progressive difficulty: start with fundamental concepts, advance to complex analysis
- Create sophisticated distractors based on common misconceptions
- Test deep understanding rather than memorization
- Base questions on concepts and scenarios from the provided context

FORMAT each question exactly as shown:
=== QUESTION 1 ===
## Title: [Clear, descriptive title]
## Difficulty: [Easy/Medium/Hard]
## Type: [Conceptual/Analytical/Applied]
## Question:
[Clear, specific question that tests understanding of concepts from the context. Include scenario or case study if relevant.]
## Options:
A) [First option - could be correct or plausible distractor]
B) [Second option - could be correct or plausible distractor]
C) [Third option - could be correct or plausible distractor]
D) [Fourth option - could be correct or plausible distractor]

=== QUESTION 2 ===
## Title: [Clear, descriptive title]
## Difficulty: [Easy/Medium/Hard]
## Type: [Conceptual/Analytical/Applied]
## Question:
[Clear question testing different concept...]
## Options:
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

Continue this exact pattern for all {question_count} questions.

IMPORTANT:
- Only one option should be clearly correct for each question
- Distractors should be plausible but clearly wrong to someone who understands the concept
- Questions should test understanding, not just recall
- Relate all questions to concepts mentioned in the provided context
- Avoid trick questions or ambiguous wording

Generate all {question_count} questions now:"""

    @staticmethod
    def create_evaluation_prompt(user_type: str, qa_pairs: List[Dict[str, Any]]) -> str:
        qa_text = []
        for i, qa in enumerate(qa_pairs, 1):
            question = qa['question'][:300] + "..." if len(qa['question']) > 300 else qa['question']
            answer = qa['answer'][:200] + "..." if len(qa['answer']) > 200 else qa['answer']
            qa_text.append(f"QUESTION {i}:\n{question}\n\nSTUDENT ANSWER:\n{answer}")
        qa_content = "\n\n" + "="*50 + "\n\n".join(qa_text)
        if user_type == "dev":
            return PromptTemplates._dev_evaluation_prompt(qa_content, len(qa_pairs))
        else:
            return PromptTemplates._non_dev_evaluation_prompt(qa_content, len(qa_pairs))

    @staticmethod
    def _dev_evaluation_prompt(qa_content: str, question_count: int) -> str:
        return f"""Evaluate this developer assessment comprehensively. Analyze code quality, problem-solving approach, technical accuracy, and software engineering best practices.

ASSESSMENT CONTENT:
{qa_content}

EVALUATION CRITERIA:
- Code correctness and functionality (30%)
- Algorithm efficiency and optimization (25%)
- Code readability and structure (20%)
- Best practices and conventions (15%)
- Problem-solving approach and explanation (10%)

INSTRUCTIONS:
1. Score each question as 1 (acceptable/correct) or 0 (unacceptable/incorrect)
2. Be strict but fair - partial credit should round to 1 if approach is sound
3. Consider: Does the answer demonstrate competent programming skills?
4. Evaluate explanations and reasoning, not just code
5. Look for understanding of time/space complexity where relevant

REQUIRED OUTPUT FORMAT:
SCORES: [1,0,1,1,0]
FEEDBACK: [Question 1: Detailed feedback|Question 2: Detailed feedback|Question 3: Detailed feedback|Question 4: Detailed feedback|Question 5: Detailed feedback]

DETAILED ANALYSIS:
Provide comprehensive analysis covering:
- Overall programming competency level
- Strengths observed in coding approach
- Areas needing improvement
- Specific technical recommendations
- Assessment of problem-solving methodology

Score each of the {question_count} questions and provide detailed feedback. Be thorough and constructive."""

    @staticmethod
    def _non_dev_evaluation_prompt(qa_content: str, question_count: int) -> str:
        return f"""Evaluate this non-developer assessment comprehensively. Focus on conceptual understanding, analytical reasoning, and practical knowledge application.

ASSESSMENT CONTENT:
{qa_content}

EVALUATION CRITERIA:
- Conceptual accuracy and understanding (40%)
- Analytical reasoning quality (30%)
- Practical application knowledge (20%)
- Communication and explanation clarity (10%)

INSTRUCTIONS:
1. Score each question as 1 (correct) or 0 (incorrect)
2. For multiple choice: only exact correct answers get 1 point
3. Evaluate understanding demonstrated in any explanations provided
4. Consider partial understanding but be consistent with scoring
5. Look for evidence of genuine comprehension vs. guessing

REQUIRED OUTPUT FORMAT:
SCORES: [1,0,1,1,0]
FEEDBACK: [Question 1: Clear feedback on answer|Question 2: Clear feedback on answer|Question 3: Clear feedback on answer|Question 4: Clear feedback on answer|Question 5: Clear feedback on answer]

DETAILED ANALYSIS:
Provide comprehensive analysis covering:
- Overall conceptual understanding level
- Analytical thinking capabilities
- Knowledge gaps identified
- Recommendations for further learning
- Assessment of technical awareness

Score each of the {question_count} questions and provide specific feedback. Focus on understanding rather than memorization."""

    @staticmethod
    def optimize_context_prompt(context: str) -> str:
        return f"""Analyze and enhance this technical content to make it optimal for generating high-quality assessment questions.

ORIGINAL CONTEXT:
{context}

ENHANCEMENT REQUIREMENTS:
- Identify key technical concepts and learning objectives
- Extract practical scenarios and real-world applications
- Highlight different difficulty levels of concepts
- Organize information for question generation
- Ensure context supports both conceptual and practical questions

ENHANCED CONTEXT FORMAT:
## Key Concepts:
[List main technical concepts]

## Practical Applications:
[Real-world scenarios and use cases]

## Difficulty Progression:
- Beginner: [Fundamental concepts]
- Intermediate: [Applied knowledge]
- Advanced: [Complex analysis and synthesis]

## Question Opportunities:
[Specific areas suitable for different question types]

Provide the enhanced context optimized for question generation:"""


class PromptValidator:
    """Validation utilities for prompts and responses"""

    @staticmethod
    def validate_question_response(response: str, user_type: str, expected_count: int) -> Dict[str, Any]:
        validation = {
            "valid": True,
            "issues": [],
            "question_count": 0,
            "format_correct": True
        }
        question_markers = response.count("=== QUESTION")
        validation["question_count"] = question_markers

        if question_markers != expected_count:
            validation["valid"] = False
            validation["issues"].append(f"Expected {expected_count} questions, found {question_markers}")

        required_sections = ["## Title:", "## Difficulty:", "## Type:", "## Question:"]
        if user_type == "non_dev":
            required_sections.append("## Options:")

        for section in required_sections:
            if response.count(section) < expected_count:
                validation["valid"] = False
                validation["issues"].append(f"Missing {section} sections")

        if user_type == "non_dev":
            option_patterns = [f"{letter})" for letter in "ABCD"]
            for pattern in option_patterns:
                if response.count(pattern) < expected_count:
                    validation["format_correct"] = False
                    validation["issues"].append(f"Inconsistent option format: {pattern}")

        return validation

    @staticmethod
    def validate_evaluation_response(response: str, expected_count: int) -> Dict[str, Any]:
        validation = {
            "valid": True,
            "issues": [],
            "has_scores": False,
            "has_feedback": False,
            "score_count": 0
        }
        if "SCORES:" in response:
            validation["has_scores"] = True
            import re
            score_match = re.search(r'SCORES:\s*\[(.*?)\]', response)
            if score_match:
                scores = score_match.group(1).split(',')
                validation["score_count"] = len([s for s in scores if s.strip() in ['0', '1']])
                if validation["score_count"] != expected_count:
                    validation["valid"] = False
                    validation["issues"].append(f"Expected {expected_count} scores, found {validation['score_count']}")
        else:
            validation["valid"] = False
            validation["issues"].append("Missing SCORES section")

        if "FEEDBACK:" in response:
            validation["has_feedback"] = True
        else:
            validation["valid"] = False
            validation["issues"].append("Missing FEEDBACK section")

        return validation


# =============================================================================
# ██╗    ██╗███████╗███████╗██╗  ██╗██╗  ██╗   ██╗
# ██║    ██║██╔════╝██╔════╝██║ ██╔╝██║  ╚██╗ ██╔╝
# ██║ █╗ ██║█████╗  █████╗  █████╔╝ ██║   ╚████╔╝
# ██║███╗██║██╔══╝  ██╔══╝  ██╔═██╗ ██║    ╚██╔╝
# ╚███╔███╔╝███████╗███████╗██║  ██╗███████╗ ██║
#  ╚══╝╚══╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═╝
# ██╗███╗   ██╗████████╗███████╗██████╗ ██╗   ██╗██╗███████╗██╗    ██╗
# ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██║   ██║██║██╔════╝██║    ██║
# ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██║   ██║██║█████╗  ██║ █╗ ██║
# ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗╚██╗ ██╔╝██║██╔══╝  ██║███╗██║
# ██║██║ ╚████║   ██║   ███████╗██║  ██║ ╚████╔╝ ██║███████╗╚███╔███╔╝
# ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝
# =============================================================================

SYSTEM_CONTEXT_BASE = """You are a professional AI interviewer conducting a Weekly Interview session. Your goal is to simulate a real-world interview with natural conversation flow.

PERSONALITY:
- Professional, calm, supportive
- Genuinely interested in responses
- Patient and encouraging
- Natural conversational flow

STYLE:
- Ask ONE clear question at a time
- Listen actively and respond to what was said
- Build contextual follow-ups
- Keep responses concise (2-3 sentences max)"""


# ---------- Dynamic Question Generation Prompts ----------

def build_communication_question_prompt(
    student_name: str,
    conversation_history: str,
    topics_already_covered: List[str],
    questions_already_asked: List[str],
    is_first_question: bool = False
) -> str:
    """Build prompt for generating dynamic communication questions"""

    topics_str = ", ".join(topics_already_covered) if topics_already_covered else "none yet"
    questions_str = "\n".join([f"- {q}" for q in questions_already_asked[-5:]]) if questions_already_asked else "none yet"

    if is_first_question:
        return f"""Generate a friendly, casual FIRST question to start a conversation with {student_name}.

This is the COMMUNICATION round - we want to get to know them personally, NOT test technical skills.

RULES:
- Ask about their interests, hobbies, favorites, or personality
- Be warm and welcoming
- Make them feel comfortable
- DO NOT ask technical questions
- Keep it casual and friendly

GOOD TOPICS: favorite places, hobbies, interests, music, movies, travel, food, weekend activities, personality traits

Generate ONE casual, friendly opening question (MAX 12 words):"""

    return f"""Generate a NEW casual conversation question for the COMMUNICATION round.

CONVERSATION SO FAR:
{conversation_history if conversation_history else "Just started"}

TOPICS ALREADY COVERED: {topics_str}
QUESTIONS ALREADY ASKED:
{questions_str}

RULES:
1. Ask about something DIFFERENT from topics already covered
2. Keep it casual and friendly - this is NOT a technical round
3. DO NOT ask about work, coding, programming, or technical topics
4. Ask about personal interests, hobbies, favorites, experiences, or personality
5. Generate a UNIQUE question not similar to ones already asked
6. Make it natural and conversational

GOOD TOPICS TO EXPLORE:
- Favorite places, cities, travel destinations
- Hobbies and interests
- Music, movies, books
- Food and cuisine
- Weekend activities
- Personal goals and dreams
- Memorable experiences
- How they relax or unwind

Generate ONE new, casual question (MAX 12 words):"""


def build_communication_followup_prompt(
    user_response: str,
    original_question: str,
    current_topic: str,
    followup_count: int
) -> str:
    """Build prompt for generating natural follow-up on same topic"""

    return f"""The candidate just answered a question. Generate a NATURAL follow-up that shows interest.

ORIGINAL QUESTION: "{original_question}"
TOPIC: {current_topic}
CANDIDATE'S RESPONSE: "{user_response}"
FOLLOW-UPS ALREADY ASKED ON THIS TOPIC: {followup_count}

YOUR TASK:
Generate ONE natural follow-up question that:
1. Shows GENUINE interest in what they said
2. Digs deeper into the SAME topic (don't change topics)
3. References something SPECIFIC they mentioned
4. Is casual and friendly, not formal
5. Is short (under 15 words)

EXAMPLE GOOD FOLLOW-UPS:
- If they mentioned a place: "What do you like most about it?"
- If they mentioned an activity: "How did you get into that?"
- If they mentioned people: "That sounds fun! Do you do that often?"
- If they shared a feeling: "What makes it so special to you?"

BAD FOLLOW-UPS (avoid):
- Changing to a completely different topic
- Asking technical questions
- Being too formal or interview-like
- Generic questions that ignore what they said

Generate ONE short, natural follow-up question that BUILDS ON what they said:"""


def build_technical_question_prompt(
    content_context: str,
    previous_questions: List[str],
    user_last_response: str,
    difficulty: str,
    conversation_history: str
) -> str:
    """Build prompt for generating dynamic technical questions from summaries"""

    prev_q_text = "\n".join([f"- {q}" for q in previous_questions[-5:]]) if previous_questions else "None yet"

    # If no content context, provide fallback technical topics
    if not content_context or len(content_context) < 50:
        content_context = "General software development, coding, debugging, databases, APIs, web development"

    return f"""Generate ONE TECHNICAL interview question based on the candidate's work.

CANDIDATE'S TECHNICAL WORK:
{content_context[:2000]}

QUESTIONS ALREADY ASKED (DO NOT REPEAT):
{prev_q_text}

DIFFICULTY: {difficulty}

⚠️ CRITICAL RULES - MUST FOLLOW:
1. Ask ONLY about technical topics: programming, coding, databases, APIs, frameworks, debugging
2. Base question on technologies mentioned in their work above
3. DO NOT REPEAT previous questions
4. Match difficulty: easy=basics, medium=how/why, hard=optimization

❌ FORBIDDEN - NEVER ASK ABOUT:
- Movies, TV shows, books, music
- Hobbies, free time, relaxation
- Favorites, preferences, feelings
- Travel, places, food
- Personal life, friends, family
- Anything NOT related to their technical work

✅ GOOD TECHNICAL QUESTIONS:
- "How did you handle authentication in your project?"
- "What database did you use and why?"
- "How would you optimize this query?"
- "Explain your approach to error handling."

Generate ONE TECHNICAL question (MAX 15 words only):"""


def build_hr_question_prompt(
    previously_asked_questions: List[str],
    conversation_history: str,
    student_context: str
) -> str:
    """Build prompt for generating unique HR/behavioral questions"""

    prev_q_text = "\n".join([f"- {q}" for q in previously_asked_questions[-15:]]) if previously_asked_questions else "None"

    return f"""Generate ONE unique behavioral/HR interview question.

QUESTIONS ALREADY ASKED (DO NOT REPEAT OR ASK SIMILAR):
{prev_q_text}

RECENT CONVERSATION:
{conversation_history[-400:] if conversation_history else 'Starting HR round'}

CANDIDATE BACKGROUND:
{student_context[:300] if student_context else 'General candidate'}

QUESTION GENERATION RULES:
1. Generate a COMPLETELY NEW question not similar to any in the list
2. Focus on behavioral situations, leadership, teamwork, or personal growth
3. Ask for SPECIFIC examples (use STAR method style)
4. Make it professional but conversational
5. Avoid generic yes/no questions

QUESTION CATEGORIES TO VARY:
- Leadership: "Tell me about a time you led/took initiative..."
- Teamwork: "Describe a situation where you collaborated/worked with others..."
- Challenges: "Share an experience where you faced a difficult situation..."
- Conflict: "Tell me about a time you had a disagreement..."
- Growth: "What feedback have you received that helped you improve..."
- Motivation: "What drives you / what are you passionate about..."
- Ethics: "What would you do if..."
- Strengths/Weaknesses: Different angles on self-awareness
- Goals: Career aspirations and personal development
- Failure: Learning from mistakes

AVOID:
- Questions already in the list above
- Questions too similar to asked ones
- Generic questions without asking for examples
- Technical questions (this is HR round)

Generate ONE short HR/behavioral question (MAX 15 words only):"""


# ---------- Round Transition Messages ----------

ROUND_TRANSITION_TO_TECHNICAL = """Great conversation! I've enjoyed getting to know you better.

Now let's move on to the Technical round. For the next 20 minutes, I'll ask you some questions based on your recent work and technical knowledge.

Don't worry if you don't know something - just share your thought process. Ready?"""

ROUND_TRANSITION_TO_HR = """Excellent work on the technical questions! You handled that well.

For our final round, we'll spend about 15 minutes on some HR and behavioral questions. I'd love to hear about your experiences and how you handle different situations.

Shall we continue?"""

INTERVIEW_COMPLETION_MESSAGE = """That brings us to the end of our interview session!

Thank you so much for your time and thoughtful responses. You did a great job engaging with the questions.

I'll now generate your detailed feedback covering all three rounds. Give me just a moment..."""


# ---------- Standard Phrase Collections ----------

ACKNOWLEDGMENT_PHRASES = [
    "That's a good point.",
    "I see what you mean.",
    "Interesting perspective.",
    "Thank you for sharing that.",
    "That's helpful to understand.",
    "Good explanation.",
    "That makes sense.",
]

TRANSITION_PHRASES = [
    "Building on that,",
    "Following up on what you mentioned,",
    "That brings up an interesting question:",
    "Related to that,",
    "Given your experience with that,",
]

ENCOURAGEMENT_PHRASES = [
    "That's exactly the kind of thinking we're looking for.",
    "Great explanation.",
    "You've clearly thought about this.",
    "Good problem-solving approach.",
    "Your reasoning is sound.",
]

CLARIFICATION_PROMPTS = [
    "Could you elaborate on that a bit more?",
    "Can you give me a specific example?",
    "What was your reasoning there?",
    "Could you walk me through your thought process?",
]

GENTLE_REDIRECT_PROMPTS = [
    "That's helpful context. Let me ask about",
    "Good to know. Moving on to",
    "Thanks for sharing. Let's explore",
]

SILENCE_GENTLE_PROMPTS = [
    "Take your time, there's no rush.",
    "Would you like me to rephrase the question?",
    "Feel free to think out loud if that helps.",
    "No pressure - take a moment if you need.",
]


# ---------- Evaluation & Scoring Prompt Templates ----------

WI_EVALUATION_PROMPT_TEMPLATE = """Evaluate interview for {student_name}.
Duration: {duration} minutes | Rounds: {stages_completed}

TRANSCRIPT:
{conversation_log}

BACKGROUND:
{content_context}

Generate evaluation with QUESTION-BY-QUESTION feedback in this EXACT format:

=== OVERALL SUMMARY ===
(2-3 sentences about overall performance)

=== COMMUNICATION ROUND ===
For EACH question in communication round:

Q1: [AI's question]
A1: [User's answer]
Feedback: [1 sentence - what was good or needs improvement]

Q2: [AI's question]
A2: [User's answer]
Feedback: [1 sentence feedback]

(continue for all communication questions...)

Round Summary: [1 sentence overall communication assessment]

=== TECHNICAL ROUND ===
For EACH question in technical round:

Q1: [AI's question]
A1: [User's answer]
Feedback: [Was answer correct? What was missing? 1-2 sentences]

Q2: [AI's question]
A2: [User's answer]
Feedback: [1-2 sentences]

(continue for all technical questions...)

Round Summary: [1 sentence overall technical assessment]

=== HR ROUND ===
For EACH question in HR round:

Q1: [AI's question]
A1: [User's answer]
Feedback: [Did they give specific example? Quality of response? 1-2 sentences]

Q2: [AI's question]
A2: [User's answer]
Feedback: [1-2 sentences]

(continue for all HR questions...)

Round Summary: [1 sentence overall HR assessment]

=== KEY STRENGTHS ===
1. [strength 1]
2. [strength 2]
3. [strength 3]

=== AREAS TO IMPROVE ===
1. [improvement 1]
2. [improvement 2]
3. [improvement 3]

=== TIPS FOR NEXT TIME ===
1. [actionable tip 1]
2. [actionable tip 2]

IMPORTANT: Include EVERY question and answer from the transcript. If user's answer was unclear or garbage text, mention that in feedback."""


WI_SCORING_PROMPT_TEMPLATE = """Score this interview (1-10 scale):

COMMUNICATION (25%): Clarity, fluency, confidence
TECHNICAL (30%): Knowledge, problem-solving, explanations
LEADERSHIP (15%): Initiative, collaboration, decision-making
BEHAVIOUR (15%): Professionalism, ethics, maturity
CONFIDENCE (15%): Self-awareness, composure, growth mindset

OUTPUT FORMAT:
COMMUNICATION: X/10
TECHNICAL: X/10
LEADERSHIP: X/10
BEHAVIOUR: X/10
CONFIDENCE: X/10
WEIGHTED_OVERALL: X/10"""


WI_SILENCE_PROMPT_TEMPLATE = """The candidate has been silent.
STAGE: {stage}
LAST QUESTION: "{last_question}"
SILENCE PROMPTS GIVEN: {silence_count}

Generate a gentle, encouraging prompt (max 15 words):"""


WI_CONVERSATION_PROMPT_TEMPLATE = """Stage: {stage}
Time Elapsed: {time_elapsed} minutes
Questions Asked: {questions_asked}
Response: "{user_response}"

History:
{conversation_history}

Answer Quality: {answer_quality}

Generate a natural response that acknowledges what they said and asks ONE follow-up question."""


WI_INTRODUCTION_PROMPT_TEMPLATE = """Generate a warm introduction for {student_name}'s interview.
Explain the 3 rounds: Communication (10min), Technical (20min), HR (15min).
Ask how they're doing today."""


# ---------- Backward-compatible aliases (so existing code referencing the old names still works) ----------
EVALUATION_PROMPT_TEMPLATE = WI_EVALUATION_PROMPT_TEMPLATE
SCORING_PROMPT_TEMPLATE = WI_SCORING_PROMPT_TEMPLATE
SILENCE_PROMPT_TEMPLATE = WI_SILENCE_PROMPT_TEMPLATE
CONVERSATION_PROMPT_TEMPLATE = WI_CONVERSATION_PROMPT_TEMPLATE
INTRODUCTION_PROMPT_TEMPLATE = WI_INTRODUCTION_PROMPT_TEMPLATE


# =============================================================================
# WEEKLY INTERVIEW HELPER FUNCTIONS
# =============================================================================

def build_introduction_prompt(student_name: str) -> str:
    return WI_INTRODUCTION_PROMPT_TEMPLATE.format(student_name=student_name)


def build_stage_prompt(stage: str, content_context: str = "") -> str:
    """Build stage-specific system prompt"""
    base = SYSTEM_CONTEXT_BASE

    if stage == "communication":
        base += "\n\nCURRENT STAGE: Communication Round - casual conversation, NO technical questions."
    elif stage == "technical":
        base += f"\n\nCURRENT STAGE: Technical Round\nCANDIDATE'S WORK:\n{content_context[:1500]}"
    elif stage == "hr":
        base += "\n\nCURRENT STAGE: HR/Behavioral Round - focus on experiences and situational questions."

    return base


def build_conversation_prompt(
    stage: str, user_response: str, content_context: str, conversation_history: str,
    round_duration: int = 10, time_elapsed: float = 0, questions_asked: int = 0,
    answer_quality: str = "neutral"
) -> str:
    return WI_CONVERSATION_PROMPT_TEMPLATE.format(
        stage=stage,
        time_elapsed=round(time_elapsed, 1),
        questions_asked=questions_asked,
        user_response=user_response,
        conversation_history=conversation_history[-1000:],
        answer_quality=answer_quality
    )


def build_silence_prompt(stage: str, last_question: str, silence_count: int) -> str:
    return WI_SILENCE_PROMPT_TEMPLATE.format(
        stage=stage,
        last_question=last_question,
        silence_count=silence_count
    )


def build_evaluation_prompt(
    student_name: str, duration: float, stages_completed: list,
    conversation_log: str, content_context: str
) -> str:
    return WI_EVALUATION_PROMPT_TEMPLATE.format(
        student_name=student_name,
        duration=f"{duration:.1f}",
        stages_completed=", ".join(stages_completed),
        conversation_log=conversation_log,
        content_context=content_context[:800]
    )


def get_round_transition_message(next_stage: str) -> str:
    transitions = {
        "technical": ROUND_TRANSITION_TO_TECHNICAL,
        "hr": ROUND_TRANSITION_TO_HR,
        "complete": INTERVIEW_COMPLETION_MESSAGE
    }
    return transitions.get(next_stage, "Let's continue.")


def validate_prompts() -> bool:
    """Validate required prompts exist"""
    required = [SYSTEM_CONTEXT_BASE, WI_EVALUATION_PROMPT_TEMPLATE, WI_SCORING_PROMPT_TEMPLATE]
    for p in required:
        if not p or len(p.strip()) < 50:
            raise ValueError("Invalid prompt detected")
    return True


# Validate on import
validate_prompts()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # --- Daily Standup ---
    "DailyStandupPrompts", "Prompts", "prompts",
    "comprehensive_evaluation_prompt",
    "per_question_evaluation_prompt",
    "generate_strengths_weaknesses_prompt",

    # --- Response phrase collections ---
    "WRONG_ANSWER_RESPONSES", "SILENCE_ENCOURAGEMENT_RESPONSES", "CANT_ANSWER_RESPONSES",

    # --- Weekend Mocktest ---
    "PromptTemplates", "PromptValidator",

    # --- Weekly Interview: System & Templates ---
    "SYSTEM_CONTEXT_BASE",
    "EVALUATION_PROMPT_TEMPLATE", "SCORING_PROMPT_TEMPLATE",
    "CONVERSATION_PROMPT_TEMPLATE", "SILENCE_PROMPT_TEMPLATE", "INTRODUCTION_PROMPT_TEMPLATE",
    "WI_EVALUATION_PROMPT_TEMPLATE", "WI_SCORING_PROMPT_TEMPLATE",
    "WI_CONVERSATION_PROMPT_TEMPLATE", "WI_SILENCE_PROMPT_TEMPLATE", "WI_INTRODUCTION_PROMPT_TEMPLATE",
    "ROUND_TRANSITION_TO_TECHNICAL", "ROUND_TRANSITION_TO_HR", "INTERVIEW_COMPLETION_MESSAGE",

    # --- Weekly Interview: Phrase collections ---
    "ACKNOWLEDGMENT_PHRASES", "TRANSITION_PHRASES", "ENCOURAGEMENT_PHRASES",
    "CLARIFICATION_PROMPTS", "GENTLE_REDIRECT_PROMPTS", "SILENCE_GENTLE_PROMPTS",

    # --- Weekly Interview: Dynamic generation ---
    "build_communication_question_prompt", "build_communication_followup_prompt",
    "build_technical_question_prompt", "build_hr_question_prompt",

    # --- Weekly Interview: Helper functions ---
    "build_introduction_prompt", "build_stage_prompt", "build_conversation_prompt",
    "build_evaluation_prompt", "build_silence_prompt", "get_round_transition_message",
    "validate_prompts",
]