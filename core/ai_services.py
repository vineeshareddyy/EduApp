# Edu-app/core/ai_services.py
# Unified AI services for: daily_standup, weekly_interview, weekend_mocktest
# - Keeps weekend_mocktest API names intact (AIService, get_ai_service)
# - Namespaces overlapping classes for daily_standup (DS_*) and weekly_interview (WI_*)
# - No functionality removed
# ✅ FIXED: Added conversation_log field to DS_SessionData for REPEAT and IRRELEVANT features
# ✅ ADDED: Deterministic question shuffling per student+session
# ✅ MERGED: Weekly Interview rewritten with template-based questions, time-based rounds,
#            hallucination detection, HR from MongoDB categories, comprehensive evaluation

import os
import time
import logging
import asyncio
import re
import uuid
import json
import random
import hashlib
import tempfile
import subprocess
import io
import wave
from typing import List, AsyncGenerator, Tuple, Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
# ---- External clients (both sync & async variants) ----
import openai as openai_sync
from groq import Groq, AsyncGroq
from openai import AsyncOpenAI

from .config import config
from .prompts import (
    # daily_standup prompt helper
    prompts as ds_prompts,
    # weekly_interview prompt helpers (used by WI evaluation & question generation)
    build_evaluation_prompt, SCORING_PROMPT_TEMPLATE,
    build_technical_question_prompt, build_hr_question_prompt,
    build_communication_question_prompt, build_communication_followup_prompt,
    WRONG_ANSWER_RESPONSES, SILENCE_ENCOURAGEMENT_RESPONSES,
    # weekend_mocktest templates
    PromptTemplates,
)

logger = logging.getLogger(__name__)


# =============================================================================
# =============================================================================
#  SECTION 1:  DAILY STANDUP  (DS_*)
# =============================================================================
# =============================================================================

# ---- Utilities used by DS_FragmentManager / DS_SummaryManager ----

def _ds_parse_summary_into_fragments(summary: str) -> Dict[str, str]:
    """Daily-standup original fragment parser (kept identical)."""
    if not summary or not summary.strip():
        return {"General": summary or "No content available"}
    lines = summary.strip().split('\n')
    section_pattern = re.compile(r'^\s*(\d+)\.\s+(.+)')
    fragments = {}
    current_section = None
    current_content = []
    for line in lines:
        match = section_pattern.match(line)
        if match:
            if current_section and current_content:
                fragments[current_section] = '\n'.join(current_content).strip()
            section_num = match.group(1)
            section_title = match.group(2).strip()
            current_section = f"{section_num}. {section_title}"
            current_content = [line]
        else:
            if current_section:
                current_content.append(line)
            else:
                fragments["Introduction"] = (fragments.get("Introduction", "") + '\n' + line).strip()
    if current_section and current_content:
        fragments[current_section] = '\n'.join(current_content).strip()
    if not fragments:
        fragments["General"] = summary
    logger.info(f"[DS] Parsed summary into {len(fragments)} fragments: {list(fragments.keys())}")
    return fragments


class DS_SessionStage(Enum):
    GREETING = "greeting"
    TECHNICAL = "technical"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class DS_ConversationExchange:
    timestamp: float
    stage: DS_SessionStage
    ai_message: str
    user_response: str
    transcript_quality: float = 0.0
    chunk_id: Optional[int] = None
    concept: Optional[str] = None
    is_followup: bool = False


@dataclass
class DS_SessionData:
    session_id: str
    test_id: str
    student_id: int
    student_name: str
    session_key: str
    created_at: float
    last_activity: float
    current_stage: DS_SessionStage
    exchanges: List[DS_ConversationExchange] = field(default_factory=list)
    conversation_window: deque = field(default_factory=lambda: deque(maxlen=config.CONVERSATION_WINDOW_SIZE))
    conversation_log: List[Dict[str, Any]] = field(default_factory=list)  # ✅ For repeat/irrelevant features
    greeting_count: int = 0
    is_active: bool = True
    websocket: Optional[Any] = field(default=None)
    summary_manager: Optional[Any] = field(default=None)
    clarification_attempts: int = 0

    # Fragment-based attributes
    fragments: Dict[str, str] = field(default_factory=dict)
    fragment_keys: List[str] = field(default_factory=list)
    concept_question_counts: Dict[str, int] = field(default_factory=dict)
    questions_per_concept: int = 2
    current_concept: str = ""
    question_index: int = 0
    followup_questions: int = 0

    # Silence interruption flags
    silence_tts_active: bool = False
    cancel_silence_tts: bool = False
    is_user_speaking_live: bool = False
    normal_tts_active: bool = False
    silence_cooldown_until: float = 0.0

    def add_exchange(self, ai_message: str, user_response: str, quality: float = 0.0,
                     concept: Optional[str] = None, is_followup: bool = False,
                     validation_result: Optional[str] = None):
        """Add exchange to conversation log with concept tracking"""
        ex = DS_ConversationExchange(
            timestamp=time.time(),
            stage=self.current_stage,
            ai_message=ai_message,
            user_response=user_response,
            transcript_quality=quality,
            chunk_id=None,
            concept=concept,
            is_followup=is_followup
        )
        self.exchanges.append(ex)
        self.conversation_window.append(ex)

        conversation_entry = {
            "timestamp": time.time(),
            "stage": self.current_stage.value,
            "ai_message": ai_message,
            "user_response": user_response,
            "quality": quality,
            "concept": concept,
            "validation_result": validation_result,
            "is_followup": is_followup
        }
        self.conversation_log.append(conversation_entry)
        self.last_activity = time.time()
        logger.info(f"✅ Exchange added - Concept: '{concept}', Is followup: {is_followup}")


@dataclass
class DS_SummaryChunk:
    id: int
    content: str
    base_questions: List[str]
    current_question_count: int = 0
    completed: bool = False
    follow_up_questions: List[str] = field(default_factory=list)


class DS_SharedClientManager:
    """Daily-standup original (sync OpenAI + Groq, threadpool)"""
    def __init__(self):
        self._groq_client = None
        self._openai_client = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.THREAD_POOL_MAX_WORKERS)

    @property
    def groq_client(self) -> Groq:
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise Exception("GROQ_API_KEY not found in environment variables")
            self._groq_client = Groq(api_key=api_key)
            logger.info("[DS] Groq client initialized")
        return self._groq_client

    @property
    def openai_client(self) -> openai_sync.OpenAI:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            self._openai_client = openai_sync.OpenAI(api_key=api_key)
            logger.info("[DS] OpenAI (sync) client initialized")
        return self._openai_client

    @property
    def executor(self):
        return self._executor

    async def close_connections(self):
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("[DS] AI client connections closed")


# global DS shared clients
ds_shared_clients = DS_SharedClientManager()


class DS_SummaryManager:
    """
    IMPROVED: Line-by-line summary parsing with example detection.
    ✅ ADDED: Deterministic question shuffling per student+session
    """

    def __init__(self, shared_clients, session_data=None):
        self.shared_clients = shared_clients
        self.session_data = session_data
        self.fragments = []
        self.current_fragment_index = 0
        self.questions_asked_on_current = 0
        self.current_topic = ""
        self.exchange_log = []
        self.total_questions_asked = 0

    def initialize_fragments(self, summary_text: str) -> bool:
        """Parse summary into ordered content units with example detection."""
        try:
            if not summary_text or len(summary_text.strip()) < 50:
                return False

            self.fragments = self._parse_summary_structured(summary_text)

            # ✅ DETERMINISTIC SHUFFLE PER STUDENT + SESSION
            if self.session_data:
                seed_str = f"{self.session_data.student_id}:{self.session_data.session_id}"
                self.fragments = self._deterministic_shuffle(self.fragments, seed_str)

            logger.info(
                "[DS] Final fragment order for session %s (student %s): %s",
                self.session_data.session_id if self.session_data else "N/A",
                self.session_data.student_id if self.session_data else "N/A",
                [f['title'] for f in self.fragments]
            )

            if not self.fragments:
                return False

            first_line = summary_text.strip().split('\n')[0]
            self.current_topic = first_line[:50].replace('#', '').strip()

            if self.session_data:
                self.session_data.fragment_keys = list(range(len(self.fragments)))
                self.session_data.current_concept = self.fragments[0]['title']

            logger.info(f"[DS] Parsed summary into {len(self.fragments)} fragments: {[f['title'][:30] for f in self.fragments]}")
            logger.info(f"[DS] Initialized {len(self.fragments)} fragments, target 1/concept")
            return True
        except Exception as e:
            logger.error(f"[DS] Fragment init error: {e}")
            return False

    def _parse_summary_structured(self, text: str) -> list:
        """Parse summary maintaining structure and detecting examples."""
        fragments = []
        lines = text.strip().split('\n')
        current = {
            'title': 'Introduction',
            'content': '',
            'has_example': False,
            'example_content': '',
            'key_terms': []
        }
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            is_header = (
                line_stripped.startswith('#') or
                re.match(r'^\d+\.?\d*\.?\s+\w', line_stripped) or
                (line_stripped.endswith(':') and len(line_stripped.split()) <= 6)
            )
            if is_header:
                if current['content'].strip():
                    current['has_example'] = self._check_for_example(current['content'])
                    current['key_terms'] = self._extract_terms(current['content'])
                    fragments.append(current.copy())
                title = re.sub(r'^#+\s*', '', line_stripped)
                title = re.sub(r'^\d+\.?\d*\.?\s*', '', title)
                title = title.rstrip(':')
                current = {
                    'title': title,
                    'content': '',
                    'has_example': False,
                    'example_content': '',
                    'key_terms': []
                }
            else:
                current['content'] += line_stripped + '\n'
                if self._is_example_line(line_stripped):
                    current['example_content'] += line_stripped + '\n'
        if current['content'].strip() or current['title']:
            current['has_example'] = self._check_for_example(current['content'])
            current['key_terms'] = self._extract_terms(current['content'])
            fragments.append(current)
        return fragments

    def _check_for_example(self, content: str) -> bool:
        patterns = [
            r'example\s*[:\-–]', r'for example', r'e\.g\.', r'such as:',
            r'Example –', r'Generated Example', r'#### Example'
        ]
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    def _is_example_line(self, line: str) -> bool:
        lower = line.lower()
        return 'example' in lower or 'e.g.' in lower

    def _extract_terms(self, content: str) -> list:
        terms = []
        terms.extend(re.findall(r'\b[A-Z]{2,4}\d{2,3}\b', content))
        terms.extend(re.findall(r'\(([A-Z]{2,6})\)', content))
        terms.extend(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content))
        return list(set(terms))[:10]

    def get_active_fragment(self) -> tuple:
        if not self.fragments or self.current_fragment_index >= len(self.fragments):
            return ("", "")
        frag = self.fragments[self.current_fragment_index]
        return (frag['title'], frag['content'])

    def get_current_fragment_details(self) -> dict:
        if not self.fragments or self.current_fragment_index >= len(self.fragments):
            return {}
        frag = self.fragments[self.current_fragment_index]
        return {
            'title': frag['title'],
            'content': frag['content'],
            'has_example': frag.get('has_example', False),
            'example_content': frag.get('example_content', ''),
            'key_terms': frag.get('key_terms', []),
            'index': self.current_fragment_index,
            'total': len(self.fragments)
        }

    def should_ask_for_example(self) -> bool:
        if not self.fragments or self.current_fragment_index >= len(self.fragments):
            return False
        frag = self.fragments[self.current_fragment_index]
        return frag.get('has_example', False) and self.questions_asked_on_current == 1

    def advance_fragment(self) -> bool:
        self.current_fragment_index += 1
        self.questions_asked_on_current = 0
        if self.current_fragment_index >= len(self.fragments):
            logger.info("[DS] All concepts have been covered - no more fragments to advance to")
            return False
        new_frag = self.fragments[self.current_fragment_index]
        logger.info(f"[DS] Advanced to concept: '{new_frag['title']}' (questions: 0/1)")
        if self.session_data:
            self.session_data.current_concept = new_frag['title']
        return True

    def _deterministic_shuffle(self, items: list, seed_str: str) -> list:
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (10**8)
        rng = random.Random(seed)
        shuffled = items.copy()
        rng.shuffle(shuffled)
        return shuffled

    def add_question(self, question: str, concept: str, is_followup: bool = False):
        self.questions_asked_on_current += 1
        self.total_questions_asked += 1
        self.exchange_log.append({
            'question': question,
            'concept': concept,
            'is_followup': is_followup,
            'fragment_index': self.current_fragment_index
        })
        logger.info(f"✅ Exchange added - Concept: '{concept}', Is followup: {is_followup}")

    def add_answer(self, answer: str):
        if self.exchange_log:
            self.exchange_log[-1]['answer'] = answer

    def should_continue_test(self) -> bool:
        return self.current_fragment_index < len(self.fragments)


class DS_OptimizedAudioProcessor:
    """Daily-standup fast STT using Groq sync client via threadpool"""
    def __init__(self, client_manager: DS_SharedClientManager):
        self.client_manager = client_manager

    @property
    def groq_client(self) -> Groq:
        return self.client_manager.groq_client

    async def transcribe_audio_fast(self, audio_data: bytes) -> Tuple[str, float]:
        try:
            audio_size = len(audio_data)
            logger.info(f"[DS] Transcribing {audio_size} bytes")
            if audio_size < 50:
                raise Exception(f"Audio data too small ({audio_size} bytes)")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor, self._sync_transcribe, audio_data
            )
        except Exception as e:
            logger.error(f"[DS] Transcription error: {e}")
            raise Exception(f"Transcription failed: {e}")

    def _sync_transcribe(self, audio_data: bytes) -> Tuple[str, float]:
        try:
            temp_file = config.TEMP_DIR / f"audio_{int(time.time()*1e6)}.webm"
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            with open(temp_file, "rb") as fh:
                result = self.groq_client.audio.transcriptions.create(
                    file=(temp_file.name, fh.read()),
                    model=config.GROQ_TRANSCRIPTION_MODEL,
                    response_format="verbose_json",
                    prompt="Please transcribe clearly, even if short."
                )
            try:
                os.remove(temp_file)
            except:
                pass
            transcript = result.text.strip() if getattr(result, "text", "") else ""
            if not transcript:
                return "", 0.0
            quality = min(len(transcript) / 30, 1.0)
            if hasattr(result, "segments") and result.segments:
                confs = [seg.get("confidence", 0.8) for seg in result.segments[:3]]
                if confs:
                    quality = (quality + sum(confs) / len(confs)) / 2
            return transcript, quality
        except Exception as e:
            if "format" in str(e).lower():
                raise Exception("Audio format not supported")
            elif "timeout" in str(e).lower():
                raise Exception("Transcription timeout")
            raise Exception(f"Groq transcription failed: {e}")


class DS_OptimizedConversationManager:
    """Daily-standup conversation management (single OpenAI call per step)"""
    def __init__(self, client_manager: DS_SharedClientManager):
        self.client_manager = client_manager

    @property
    def openai_client(self):
        return self.client_manager.openai_client

    def _sync_openai_call(self, prompt: str) -> str:
        try:
            resp = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.OPENAI_TEMPERATURE,
                max_tokens=config.OPENAI_MAX_TOKENS
            )
            result = resp.choices[0].message.content.strip()
            if not result:
                raise Exception("OpenAI returned empty response")
            return result
        except Exception as e:
            logger.error(f"[DS] OpenAI call failed: {e}")
            raise Exception(f"OpenAI API failed: {e}")

    # =========================================================================
    # LLM-based answer evaluation
    # =========================================================================
    def _evaluate_single_answer(self, question: str, answer: str, concept: str = "") -> dict:
        """Use LLM to evaluate if an answer is correct, partial, or incorrect."""
        try:
            if not answer or answer.strip() in ["", "[SKIP]", "[SILENT]", "[IRRELEVANT]", "[AUTO_ADVANCE]", "[SESSION_ENDED]", "(Session ended - no answer)"]:
                return {"evaluation": "no_response", "score": 0, "feedback": "No response provided"}

            answer_lower = answer.lower().strip()
            answer_words = answer.split()
            non_answers = [
                "no", "yes", "okay", "ok", "thank you", "thanks", "beware",
                "maybe", "possibly", "i think", "or no", "hello", "hi",
                "sure", "yeah", "nope", "hmm", "um", "uh", "i don't know",
                "not sure", "idk", "dunno", "no idea", "pass"
            ]
            if len(answer_words) <= 3:
                for non_ans in non_answers:
                    if answer_lower == non_ans or answer_lower.rstrip('.!?') == non_ans:
                        return {"evaluation": "incorrect", "score": 1, "feedback": "Response does not answer the technical question"}
                if len(answer_words) <= 2 and not any(c.isupper() for c in answer[1:] if len(answer) > 1):
                    technical_indicators = ['client', 'sap', 'system', 'data', 'server', 'code', 'transaction', 'table', 'field', 'module', 'function', 'process', '000', '001', '066']
                    has_technical = any(ind in answer_lower for ind in technical_indicators)
                    if not has_technical:
                        return {"evaluation": "incorrect", "score": 1, "feedback": "Response too brief and lacks technical content"}

            eval_prompt = f"""You are evaluating an answer to a technical interview question.

QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

TOPIC/CONCEPT: {concept if concept and concept != "unknown" else "Technical knowledge"}

EVALUATION CRITERIA:
- CORRECT (Score 7-10): Answer is accurate, relevant, and demonstrates understanding of the concept
- PARTIAL (Score 4-6): Answer shows some understanding but is incomplete, vague, or has minor errors
- INCORRECT (Score 0-3): Answer is wrong, completely off-topic, irrelevant, or doesn't address the question at all

IMPORTANT RULES:
1. Single word responses like "No", "Yes", "Thank you", "Beware", "Okay" that don't explain anything technical are INCORRECT (Score 1-2)
2. Vague or non-technical responses that don't address the question are INCORRECT
3. The answer must actually address what the question asked
4. Responses that show confusion or don't attempt to answer are INCORRECT
5. Partial credit requires at least SOME relevant technical content

Analyze the answer and provide your evaluation in this EXACT format:
EVALUATION: [CORRECT/PARTIAL/INCORRECT]
SCORE: [0-10]
FEEDBACK: [One sentence explaining why]"""

            response = self._sync_openai_call(eval_prompt)
            evaluation = "incorrect"
            score = 2
            feedback = "Could not properly evaluate response"
            response_upper = response.upper()

            if "EVALUATION: CORRECT" in response_upper or "EVALUATION:CORRECT" in response_upper:
                evaluation = "correct"
            elif "EVALUATION: PARTIAL" in response_upper or "EVALUATION:PARTIAL" in response_upper:
                evaluation = "partial"
            elif "EVALUATION: INCORRECT" in response_upper or "EVALUATION:INCORRECT" in response_upper:
                evaluation = "incorrect"

            score_match = re.search(r'SCORE:\s*(\d+)', response)
            if score_match:
                score = max(0, min(10, int(score_match.group(1))))
            else:
                score = 9 if evaluation == "correct" else (5 if evaluation == "partial" else 2)

            feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip().split('\n')[0]
            else:
                if evaluation == "correct":
                    feedback = "Excellent answer demonstrating clear understanding"
                elif evaluation == "partial":
                    feedback = "Partially correct - answer could be more complete"
                else:
                    feedback = "Answer does not adequately address the question"

            logger.info(f"📝 LLM Evaluation: Q='{question[:50]}...' A='{answer[:30]}...' → {evaluation.upper()} ({score}/10)")
            return {"evaluation": evaluation, "score": score, "feedback": feedback}

        except Exception as e:
            logger.warning(f"Answer evaluation failed: {e}")
            if len(answer.split()) <= 3:
                return {"evaluation": "incorrect", "score": 2, "feedback": "Response too brief to be a valid technical answer"}
            return {"evaluation": "partial", "score": 4, "feedback": "Could not fully evaluate - marked as partial"}

    # =========================================================================
    # generate_fast_response
    # =========================================================================
    async def generate_fast_response(self, session_data: DS_SessionData, user_input: str) -> str:
        try:
            if session_data.current_stage == DS_SessionStage.GREETING:
                ctx = {
                    "recent_exchanges": [
                        f"AI: {ex.ai_message}, User: {ex.user_response}"
                        for ex in list(session_data.conversation_window)[-2:]
                    ]
                }
                prompt = ds_prompts.dynamic_greeting_response(user_input, session_data.greeting_count, ctx)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(ds_shared_clients.executor, self._sync_openai_call, prompt)

            if session_data.current_stage == DS_SessionStage.TECHNICAL:
                fm: DS_SummaryManager = session_data.summary_manager
                if not fm:
                    raise Exception("Fragment manager not initialized")
                if not fm.should_continue_test():
                    session_data.current_stage = DS_SessionStage.COMPLETE
                    conversation_summary = {
                        "topics_covered": [],
                        "total_exchanges": len(session_data.exchanges),
                        "name": session_data.student_name,
                    }
                    prompt = ds_prompts.dynamic_session_completion(conversation_summary)
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(ds_shared_clients.executor, self._sync_openai_call, prompt)

                current_concept_title, current_concept_content = fm.get_active_fragment()
                last_q = session_data.exchanges[-1].ai_message if session_data.exchanges else ""
                questions_for_concept = fm.questions_asked_on_current

                prompt = ds_prompts.dynamic_followup_response(
                    context_text=current_concept_content[:2000],
                    user_input=user_input,
                    previous_question=last_q,
                    session_state={
                        "domain": current_concept_title,
                        "questions_asked": questions_for_concept,
                        "concept": current_concept_title
                    }
                )
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(ds_shared_clients.executor, self._sync_openai_call, prompt)

                lines = response.strip().split('\n')
                understanding = "NO"
                concept = current_concept_title
                actual_response = response
                for line in lines:
                    if line.upper().startswith("UNDERSTANDING:"):
                        understanding = line.split(":", 1)[1].strip().upper()
                    elif line.upper().startswith("CONCEPT:"):
                        concept = line.split(":", 1)[1].strip()
                    elif line.upper().startswith("QUESTION:"):
                        actual_response = line.split(":", 1)[1].strip()

                if understanding == "YES":
                    next_concept_title, _ = fm.get_active_fragment()
                    fm.add_question(actual_response, next_concept_title, False)
                else:
                    fm.add_question(actual_response, current_concept_title, True)
                return actual_response

            session_context = {
                'key_topics': list(set(ex.chunk_id for ex in session_data.exchanges if ex.chunk_id))[:3],
                'total_exchanges': len(session_data.exchanges)
            }
            prompt = ds_prompts.dynamic_conclusion_response(user_input, session_context)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(ds_shared_clients.executor, self._sync_openai_call, prompt)

        except Exception as e:
            logger.error(f"[DS] Response generation error: {e}")
            raise Exception(f"AI response generation failed: {e}")

    # =========================================================================
    # generate_fast_evaluation  (LLM-based)
    # =========================================================================
    async def generate_fast_evaluation(self, session_data) -> Tuple[str, float, dict]:
        """Comprehensive evaluation that calculates REAL scores based on actual performance."""
        try:
            conversation_log = getattr(session_data, "conversation_log", [])
            if not conversation_log:
                raise Exception("No conversation data found for evaluation")

            logger.info(f"📊 Starting comprehensive evaluation with {len(conversation_log)} entries")

            paired_exchanges = []
            stats = {
                'duration_minutes': round((time.time() - session_data.created_at) / 60, 1),
                'total_technical_questions': 0, 'answered_count': 0,
                'correct_count': 0, 'partial_count': 0, 'incorrect_count': 0,
                'skipped_count': 0, 'silent_count': 0, 'irrelevant_count': 0,
                'repeat_requests_count': 0, 'greeting_exchanges': 0,
                'concepts_covered': set(), 'concepts_strong': set(), 'concepts_weak': set(),
            }

            for idx in range(len(conversation_log)):
                entry = conversation_log[idx]
                ai_message = entry.get("ai_message", "")
                stage = entry.get("stage", "unknown")
                concept = entry.get("concept", "unknown")
                is_followup = entry.get("is_followup", False)
                quality = entry.get("quality", 0.0)

                if not ai_message or len(ai_message.strip()) < 10:
                    continue
                if stage == "greeting":
                    stats['greeting_exchanges'] += 1
                    continue

                ai_msg_lower = ai_message.lower()
                silence_prompt_phrases = [
                    "are you there", "still with me", "can you hear",
                    "are you still", "hello?", "you there", "are you ready",
                    "just checking", "i'd love to hear", "take your time",
                    "need me to repeat", "i'm here when", "let me know"
                ]
                if any(phrase in ai_msg_lower for phrase in silence_prompt_phrases):
                    continue

                user_answer = ""
                answer_quality = 0.0
                if idx + 1 < len(conversation_log):
                    next_entry = conversation_log[idx + 1]
                    user_answer = next_entry.get("user_response", "")
                    answer_quality = next_entry.get("quality", 0.0)
                else:
                    user_answer = "(Session ended - no answer)"

                if user_answer == "(session_start)":
                    continue

                response_type = "answered"
                evaluation_result = "pending"
                score_for_question = 0
                answer_feedback = ""

                if not user_answer or user_answer.strip() == "":
                    response_type = "no_response"; evaluation_result = "no_response"; score_for_question = 0; answer_feedback = "No response provided"
                elif user_answer == "(Session ended - no answer)":
                    response_type = "session_ended"; evaluation_result = "session_ended"; score_for_question = 0; answer_feedback = "Session ended before response"
                elif user_answer == "[USER_SILENT]":
                    response_type = "silent"; evaluation_result = "silent"; score_for_question = 0; stats['silent_count'] += 1; answer_feedback = "No response provided - practice articulating answers"
                elif user_answer == "[SKIP]":
                    response_type = "skipped"; evaluation_result = "skipped"; score_for_question = 0; stats['skipped_count'] += 1; answer_feedback = "Question was skipped - attempt to answer even if unsure"
                elif user_answer == "[IRRELEVANT]":
                    response_type = "irrelevant"; evaluation_result = "irrelevant"; score_for_question = 0; stats['irrelevant_count'] += 1; answer_feedback = "Response was off-topic - focus on the specific question asked"
                elif user_answer == "[AUTO_ADVANCE]":
                    response_type = "auto_advance"; evaluation_result = "no_response"; score_for_question = 0; answer_feedback = "Auto-advanced due to no response"
                else:
                    lower_answer = user_answer.lower()
                    if any(p in lower_answer for p in ["repeat", "again", "what did you", "didn't hear", "pardon", "can you repeat"]):
                        response_type = "repeat_request"; evaluation_result = "repeat_request"; score_for_question = 0; stats['repeat_requests_count'] += 1; answer_feedback = "Asked to repeat - try to catch the question first time"
                    else:
                        response_type = "answered"; stats['answered_count'] += 1
                        llm_eval = self._evaluate_single_answer(question=ai_message, answer=user_answer, concept=concept)
                        evaluation_result = llm_eval["evaluation"]; score_for_question = llm_eval["score"]; answer_feedback = llm_eval["feedback"]
                        if evaluation_result == "correct":
                            stats['correct_count'] += 1; stats['concepts_strong'].add(concept)
                        elif evaluation_result == "partial":
                            stats['partial_count'] += 1
                        else:
                            stats['incorrect_count'] += 1; stats['concepts_weak'].add(concept)
                        stats['concepts_covered'].add(concept)

                stats['total_technical_questions'] += 1
                paired_exchanges.append({
                    "question_number": stats['total_technical_questions'],
                    "question": ai_message, "answer": user_answer if response_type == "answered" else f"[{response_type.upper()}]",
                    "response_type": response_type, "concept": concept,
                    "quality_score": answer_quality, "evaluation": evaluation_result,
                    "score": score_for_question, "is_followup": is_followup, "feedback": answer_feedback
                })

            logger.info(f"📊 Paired {len(paired_exchanges)} technical Q&A exchanges")

            # --- Calculate scores ---
            total_questions = stats['total_technical_questions'] or 1
            answered = stats['answered_count']; correct = stats['correct_count']
            partial = stats['partial_count']; incorrect = stats['incorrect_count']

            if answered > 0:
                quality_points = (correct * 1.0) + (partial * 0.5)
                technical_score = round((quality_points / answered) * 100, 1)
            else:
                technical_score = 0

            final_comm_score = getattr(session_data, 'final_communication_score', None)
            if final_comm_score and isinstance(final_comm_score, dict):
                communication_score = final_comm_score.get('total_score', 50)
            elif total_questions > 0:
                response_attempts = answered + stats['irrelevant_count']
                willingness_rate = response_attempts / total_questions if total_questions > 0 else 0
                willingness_score = willingness_rate * 30
                relevance_rate = answered / response_attempts if response_attempts > 0 else 0
                relevance_score = relevance_rate * 30
                response_times = getattr(session_data, 'response_times', [])
                if response_times:
                    avg_time = sum(response_times) / len(response_times)
                    responsiveness_score = 25 if avg_time < 3 else (20 if avg_time < 5 else (15 if avg_time < 8 else (10 if avg_time < 12 else 5)))
                else:
                    responsiveness_score = 15
                clarity_penalty = min(stats['repeat_requests_count'] * 3, 15)
                clarity_score = 15 - clarity_penalty
                communication_score = max(0, min(100, round(willingness_score + relevance_score + responsiveness_score + clarity_score, 1)))
            else:
                communication_score = 50

            if total_questions > 0:
                problem_responses = stats['silent_count'] + stats['irrelevant_count'] + stats['skipped_count']
                engaged_responses = max(0, total_questions - problem_responses)
                engagement_rate = engaged_responses / total_questions
                base_attentiveness = engagement_rate * 100
                repeat_penalty = min(stats['repeat_requests_count'] * 2, 10)
                attentiveness_score = round(max(0, base_attentiveness - repeat_penalty), 1)
            else:
                attentiveness_score = 50

            overall_score = round((technical_score * 0.50) + (communication_score * 0.25) + (attentiveness_score * 0.25), 1)

            if overall_score >= 80: grade = "A"
            elif overall_score >= 65: grade = "B"
            elif overall_score >= 50: grade = "C"
            elif overall_score >= 35: grade = "D"
            else: grade = "F"

            # --- Strengths / Weaknesses ---
            strengths = []; weaknesses = []; areas_for_improvement = []
            if correct >= answered * 0.7 and answered > 0:
                strengths.append("Strong technical knowledge demonstrated across most questions")
            elif correct >= answered * 0.5 and answered > 0:
                strengths.append("Good understanding of core technical concepts")
            if stats['silent_count'] == 0 and stats['irrelevant_count'] == 0:
                strengths.append("Excellent focus and attentiveness throughout the session")
            elif stats['silent_count'] <= 2:
                strengths.append("Generally attentive with minimal distractions")
            if answered >= total_questions * 0.8:
                strengths.append("High engagement - attempted to answer most questions")
            if stats['repeat_requests_count'] <= 1:
                strengths.append("Good listening skills - understood questions clearly")
            if len(stats['concepts_strong']) > 0:
                clean_topics = [t.replace('**', '').replace('*', '').strip()[:30] for t in list(stats['concepts_strong'])[:3] if t and t != 'unknown']
                if clean_topics:
                    strengths.append(f"Demonstrated strong knowledge in: {', '.join(clean_topics)}")
            if communication_score >= 80:
                strengths.append("Clear and effective communication of technical concepts")
            if not strengths:
                strengths.append("Participated actively in the session" if answered > 0 else "Completed the session")

            if stats['incorrect_count'] >= 3:
                weaknesses.append(f"Multiple incorrect answers ({stats['incorrect_count']} times)")
                areas_for_improvement.append("Review fundamental concepts and practice explaining them clearly")
            if stats['silent_count'] >= 5:
                weaknesses.append(f"Frequent unresponsiveness ({stats['silent_count']} silent periods)")
                areas_for_improvement.append("Practice staying engaged and responding promptly to questions")
            elif stats['silent_count'] >= 3:
                weaknesses.append(f"Multiple silent periods ({stats['silent_count']} times)")
                areas_for_improvement.append("Work on maintaining focus throughout technical discussions")
            if stats['irrelevant_count'] >= 3:
                weaknesses.append(f"Multiple off-topic responses ({stats['irrelevant_count']} times)")
                areas_for_improvement.append("Focus on understanding the question before answering")
            if stats['skipped_count'] >= 3:
                weaknesses.append(f"Skipped several questions ({stats['skipped_count']} times)")
                areas_for_improvement.append("Review fundamental concepts to build confidence in answering")
            if technical_score < 50:
                weaknesses.append("Technical knowledge needs significant improvement")
                areas_for_improvement.append("Deep dive into the core concepts covered in this session")
            elif technical_score < 70:
                weaknesses.append("Some gaps in technical understanding")
                areas_for_improvement.append("Review and practice the topics where answers were incorrect")
            if stats['repeat_requests_count'] >= 3:
                weaknesses.append("Frequently asked for questions to be repeated")
                areas_for_improvement.append("Practice active listening during technical discussions")
            weak_concepts = list(stats['concepts_weak'])[:3]
            if weak_concepts:
                clean_weak = [c.replace('**', '').replace('*', '').strip()[:30] for c in weak_concepts if c and c != 'unknown']
                if clean_weak:
                    areas_for_improvement.append(f"Focus on reviewing: {', '.join(clean_weak)}")
            if not areas_for_improvement:
                if technical_score < 80:
                    areas_for_improvement.append("Continue practicing technical explanations")
                areas_for_improvement.append("Regular review of key concepts will strengthen retention")

            # --- Question analysis ---
            question_analysis = []
            for ex in paired_exchanges:
                fb = ex.get('feedback', '')
                if not fb:
                    fb_map = {
                        'correct': "Excellent answer demonstrating clear understanding",
                        'partial': "Partially correct - answer could be more complete",
                        'incorrect': "Answer was not accurate - review this concept",
                        'skipped': "Question was skipped - attempt to answer even if unsure",
                        'silent': "No response provided - practice articulating answers",
                        'irrelevant': "Response was off-topic - focus on the specific question asked",
                        'repeat_request': "Asked to repeat - try to catch the question first time",
                    }
                    fb = fb_map.get(ex['evaluation'], f"Response type: {ex['response_type']}")
                question_analysis.append({
                    "question_number": ex['question_number'],
                    "question": ex['question'][:300],
                    "answer": (ex['answer'][:300] if ex['answer'] else "[No answer]"),
                    "concept": (ex['concept'].replace('**', '').replace('*', '').strip()[:50] if ex['concept'] else "General"),
                    "evaluation": ex['evaluation'], "score": ex['score'], "feedback": fb
                })

            # --- Attentiveness analysis ---
            engagement_level = "High" if attentiveness_score >= 80 else ("Medium" if attentiveness_score >= 50 else "Low")
            response_consistency = "Consistent" if stats['silent_count'] <= 2 and stats['irrelevant_count'] <= 1 else "Inconsistent"
            focus_areas = "Technical questions" if answered > 0 else "Needs improvement"
            distraction_indicators = []
            if stats['silent_count'] > 2: distraction_indicators.append(f"{stats['silent_count']} silent periods")
            if stats['irrelevant_count'] > 0: distraction_indicators.append(f"{stats['irrelevant_count']} off-topic responses")
            if stats['repeat_requests_count'] > 2: distraction_indicators.append(f"{stats['repeat_requests_count']} repeat requests")
            attentiveness_analysis = {
                "engagement_level": engagement_level, "response_consistency": response_consistency,
                "focus_areas": focus_areas,
                "distraction_indicators": ", ".join(distraction_indicators) if distraction_indicators else "None detected"
            }

            topics_mastered = [t.replace('**', '').replace('*', '').strip()[:40] for t in list(stats['concepts_strong'])[:5] if t and t != 'unknown']
            topics_to_review = [t.replace('**', '').replace('*', '').strip()[:40] for t in list(stats['concepts_weak'])[:5] if t and t != 'unknown']

            performance_desc = "excellent" if overall_score >= 80 else ("good" if overall_score >= 65 else ("satisfactory" if overall_score >= 50 else ("below expectations" if overall_score >= 35 else "unsatisfactory")))
            summary = f"Candidate answered {answered} out of {total_questions} technical questions. Of those answered, {correct} were correct, {partial} were partial, and {incorrect} were incorrect. Overall performance was {performance_desc} with a technical score of {technical_score}/100."

            detailed_evaluation = {
                "overall_score": overall_score, "technical_score": technical_score,
                "communication_score": communication_score, "attentiveness_score": attentiveness_score,
                "grade": grade, "summary": summary,
                "strengths": strengths, "weaknesses": weaknesses,
                "areas_for_improvement": areas_for_improvement,
                "question_analysis": question_analysis,
                "attentiveness_analysis": attentiveness_analysis,
                "recommendations": areas_for_improvement[:5],
                "topics_mastered": topics_mastered, "topics_to_review": topics_to_review,
                "raw_stats": {
                    "total_questions": total_questions, "answered_count": answered,
                    "correct_count": correct, "partial_count": partial, "incorrect_count": incorrect,
                    "skipped_count": stats['skipped_count'], "silent_count": stats['silent_count'],
                    "irrelevant_count": stats['irrelevant_count'],
                    "repeat_requests_count": stats['repeat_requests_count'],
                    "duration_minutes": stats['duration_minutes'],
                    "greeting_exchanges": stats['greeting_exchanges']
                },
                "session_info": {
                    "session_id": session_data.session_id, "test_id": session_data.test_id,
                    "student_id": session_data.student_id, "student_name": session_data.student_name,
                    "duration_minutes": stats['duration_minutes']
                }
            }

            evaluation_text = self._format_evaluation_text(detailed_evaluation)
            logger.info(f"✅ Evaluation complete: Score={overall_score}, Grade={grade}")
            return evaluation_text, overall_score, detailed_evaluation

        except Exception as e:
            logger.error(f"[DS] Evaluation error: {e}")
            import traceback; traceback.print_exc()
            return "Evaluation could not be completed due to an error.", 50.0, {
                "error": str(e), "overall_score": 50, "technical_score": 50,
                "communication_score": 50, "attentiveness_score": 50,
                "grade": "C", "summary": "Evaluation encountered an error"
            }

    def _format_evaluation_text(self, evaluation: dict) -> str:
        text_parts = []
        text_parts.append(f"=== DAILY STANDUP EVALUATION REPORT ===\n")
        text_parts.append(f"Overall Score: {evaluation.get('overall_score', 0)}/100 (Grade: {evaluation.get('grade', 'N/A')})")
        text_parts.append(f"\nSummary: {evaluation.get('summary', 'No summary available.')}\n")
        text_parts.append("--- SCORE BREAKDOWN ---")
        text_parts.append(f"Technical Knowledge: {evaluation.get('technical_score', 0)}/100")
        text_parts.append(f"Communication: {evaluation.get('communication_score', 0)}/100")
        text_parts.append(f"Attentiveness: {evaluation.get('attentiveness_score', 0)}/100\n")
        for label, key, marker in [("STRENGTHS", "strengths", "✓"), ("AREAS OF CONCERN", "weaknesses", "✗")]:
            items = evaluation.get(key, [])
            if items:
                text_parts.append(f"--- {label} ---")
                for s in items:
                    text_parts.append(f"{marker} {s}")
                text_parts.append("")
        recommendations = evaluation.get('recommendations', [])
        if recommendations:
            text_parts.append("--- RECOMMENDATIONS ---")
            for i, r in enumerate(recommendations, 1):
                text_parts.append(f"{i}. {r}")
            text_parts.append("")
        mastered = evaluation.get('topics_mastered', [])
        to_review = evaluation.get('topics_to_review', [])
        if mastered: text_parts.append(f"Topics Mastered: {', '.join(mastered)}")
        if to_review: text_parts.append(f"Topics to Review: {', '.join(to_review)}")
        return "\n".join(text_parts)

    def _create_fallback_evaluation(self, stats: dict, conversation: list) -> dict:
        total = stats.get('total_questions', 1) or 1
        answered = stats.get('answered_count', 0)
        answer_rate = (answered / total) * 100 if total > 0 else 0
        base_score = min(100, max(0, answer_rate))
        penalty = (stats.get('skipped_count', 0) * 5 + stats.get('silent_count', 0) * 3 + stats.get('irrelevant_count', 0) * 7)
        final_score = max(30, base_score - penalty)
        if final_score >= 80: grade = "A"
        elif final_score >= 65: grade = "B"
        elif final_score >= 50: grade = "C"
        elif final_score >= 35: grade = "D"
        else: grade = "F"
        return {
            "overall_score": round(final_score, 1), "technical_score": round(final_score, 1),
            "communication_score": round(min(100, answer_rate + 10), 1),
            "attentiveness_score": round(max(0, 100 - (stats.get('silent_count', 0) + stats.get('irrelevant_count', 0)) * 10), 1),
            "grade": grade,
            "summary": f"Candidate answered {answered} out of {total} technical questions.",
            "strengths": ["Participated in the session"], "weaknesses": [],
            "areas_for_improvement": ["Review core concepts"],
            "question_analysis": [],
            "attentiveness_analysis": {"engagement_level": "Medium", "response_consistency": "Consistent", "focus_areas": "Technical questions", "distraction_indicators": "None detected"},
            "recommendations": ["Continue practicing"], "topics_mastered": [], "topics_to_review": []
        }


# =============================================================================
# =============================================================================
#  SECTION 2:  WEEKLY INTERVIEW  (WI_*)
# =============================================================================
# =============================================================================

# ---- Round durations: Communication 10 min, Technical 25 min, HR 10 min ----
ROUND_DURATIONS = {
    "introduction": 60,       # 1 minute
    "communication": 300,     # 10 minutes
    "technical": 1500,        # 25 minutes
    "hr": 600,                # 10 minutes
}

# ---------------------------------------------------------------------------
# 40 QUESTION TEMPLATES - Works for ANY subject (SAP, Python, Java, etc.)
# {tech} = technology/topic from user's MongoDB summary
# {project} = project context from summary
# ---------------------------------------------------------------------------

# TECHNICAL QUESTIONS (25 templates)
TECHNICAL_QUESTION_TEMPLATES = [
    # Basic Understanding (Q1-Q5)
    "Can you explain what {tech} is and how you've used it in your work?",
    "What are the key components or features of {tech} that you worked with?",
    "How does {tech} fit into the overall architecture of your projects?",
    "Walk me through the basic workflow when working with {tech}.",
    "What's the purpose of {tech} and why is it important in your domain?",
    # Practical Experience (Q6-Q10)
    "Describe a specific project where you implemented {tech}.",
    "What was your day-to-day work with {tech} like?",
    "How did you configure or set up {tech} in your environment?",
    "What tools, commands, or transactions did you use when working with {tech}?",
    "Can you give me an example of how you used {tech} to solve a real business problem?",
    # Problem Solving (Q11-Q15)
    "What was the most challenging issue you faced with {tech} and how did you resolve it?",
    "Describe a bug or error you encountered in {tech} and your debugging approach.",
    "How do you troubleshoot problems when {tech} isn't working correctly?",
    "Tell me about a time when {tech} failed unexpectedly. How did you handle it?",
    "What's the most complex problem you solved using {tech}?",
    # Best Practices (Q16-Q20)
    "What best practices do you follow when working with {tech}?",
    "How do you ensure quality and avoid errors when implementing {tech}?",
    "What documentation or standards do you follow for {tech}?",
    "How do you test your work with {tech} before deploying to production?",
    "What common mistakes should be avoided when working with {tech}?",
    # Advanced & Integration (Q21-Q25)
    "How does {tech} integrate with other systems or components you've worked with?",
    "What performance considerations do you keep in mind when using {tech}?",
    "How do you handle security aspects when working with {tech}?",
    "What improvements or optimizations have you made to {tech} processes?",
    "How do you train or guide others on using {tech}?",
]

# BEHAVIORAL QUESTIONS (15 templates)
TECHNICAL_BEHAVIORAL_QUESTIONS = [
    # Problem Solving & Challenges (Q26-Q30)
    "Tell me about a challenging problem you solved while working on {tech}.",
    "Describe a situation where you had to learn {tech} quickly under pressure.",
    "Tell me about a time when your {tech} implementation didn't go as planned. What did you do?",
    "Describe a difficult decision you had to make regarding {tech}.",
    "Tell me about a time you identified and fixed a critical issue in {tech}.",
    # Teamwork & Communication (Q31-Q35)
    "Describe a time when you had to explain {tech} concepts to someone non-technical.",
    "Tell me about a project where you collaborated with others on {tech}.",
    "How did you handle a disagreement with a colleague about {tech} implementation?",
    "Describe a time when you received feedback on your {tech} work. How did you respond?",
    "Tell me about a time you helped a team member who was struggling with {tech}.",
    # Initiative & Growth (Q36-Q40)
    "Tell me about a time you took initiative to improve a {tech} process.",
    "Describe how you stay updated with new developments in {tech}.",
    "Tell me about a time you went beyond your responsibilities for a {tech} project.",
    "Describe a {tech} skill you developed on your own. How did you learn it?",
    "Tell me about a time you proposed a new approach or solution for {tech}.",
]

# HR/SOFT SKILL QUESTIONS (15 templates)
HR_QUESTIONS_POOL = [
    "Describe a time when you took the lead on a project.",
    "Tell me about a situation where you motivated your team during a difficult time.",
    "How do you prioritize tasks when you have multiple deadlines?",
    "Describe a time when you had to make a decision without all the information you needed.",
    "Tell me about a time you took ownership of a mistake and fixed it.",
    "How do you handle sudden changes in project requirements?",
    "Describe a time when you had to adapt to a new technology or process quickly.",
    "Tell me about a failure you experienced and what you learned from it.",
    "How do you handle criticism about your work?",
    "Where do you see yourself professionally in 5 years?",
    "How do you maintain work-life balance during demanding projects?",
    "Describe your ideal work environment.",
    "What motivates you to do your best work?",
    "How do you handle stress when facing tight deadlines?",
    "Tell me about a time you went above and beyond for a project or client.",
]

GENERIC_HR_QUESTIONS = [
    "What motivates you to do your best work?",
    "How do you handle stress when facing tight deadlines?",
    "Where do you see yourself professionally in 5 years?",
    "How do you handle criticism about your work?",
    "Tell me about a time you went above and beyond for a project.",
    "What values are important to you in a workplace?",
    "How do you approach learning new skills?",
    "Describe your ideal work environment.",
    "What are your greatest professional strengths?",
    "How do you prioritize tasks when you have multiple deadlines?"
]

# GENERIC FALLBACK QUESTIONS
GENERIC_TECHNICAL_QUESTIONS = [
    "Can you describe your typical day at work?",
    "What technical skills are you most proud of?",
    "Tell me about a project you're particularly proud of.",
    "How do you approach learning new technologies?",
    "What's the most interesting technical problem you've solved recently?",
    "How do you stay current with industry trends?",
    "Describe your experience with system troubleshooting.",
    "What development or administration tools are you most comfortable with?",
    "How do you document your work?",
    "What's your approach to testing and quality assurance?",
]

GENERIC_BEHAVIORAL_QUESTIONS = [
    "Tell me about a time you overcame a significant challenge at work.",
    "Describe a situation where you had to work with a difficult team member.",
    "Tell me about a time you had to meet a very tight deadline.",
    "Describe a project that didn't go as planned and how you handled it.",
    "Tell me about a time you received constructive criticism.",
    "How do you approach debugging a complex issue?",
    "Tell me about a project where you had to collaborate with others.",
    "Describe a time you had to explain technical concepts to non-technical people.",
    "How do you handle disagreements about technical decisions?",
    "Tell me about a time you improved an existing process.",
]

# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------
COMMUNICATION_TRANSITIONS = [
    "That's interesting! ", "Nice! ", "Great to know! ", "Thanks for sharing! ",
    "That sounds wonderful! ", "How lovely! ", "That's cool! ", "Awesome! ",
    "That's really nice! ", "Wonderful! ", "Oh, that's great! ", "I like that! ",
    "Sounds fun! ", "That's fantastic! ", "How interesting! ", "Good to know! ",
]

FOLLOWUP_ACKS = ["Oh interesting!", "That's nice!", "I see!", "That sounds great!", "Nice!", "Lovely!", "Oh really?", "That's cool!", "Wow!", "Fascinating!"]
TECHNICAL_GOOD_ACKS = ["Good explanation!", "That's correct!", "Nice approach!", "Well explained!", "Good point!", "Exactly right!", "Great understanding!", "Well done!", "Perfect!", "Excellent!"]
TECHNICAL_NEUTRAL_ACKS = ["I see.", "Okay.", "Alright.", "Got it.", "Understood.", "Fair enough."]

DONT_KNOW_RESPONSES = [
    "That's okay! Let me ask you something different.",
    "No problem at all! Here's another question.",
    "It's fine! Let's try a different one.",
    "No worries! Let me change the topic.",
    "That's alright! Moving to something else.",
]
WEAK_RESPONSE_ACKS = [
    "I see. Let me ask you something else.",
    "Okay, let's try a different question.",
    "Alright, let me move to another topic.",
    "Got it. Here's a different one.",
    "Understood. Let me ask something else.",
]
SKIP_RESPONSES = ["Sure! Let's move on.", "No problem, next one.", "Of course! Here's another.", "Got it, moving forward."]
REPEAT_RESPONSES = ["Of course! The question was:", "Sure, let me repeat:", "No problem! Here it is again:"]
HR_ACKS = ["Thank you for sharing.", "That's a good point.", "I appreciate that.", "Interesting.", "Good to know."]


# ---------------------------------------------------------------------------
# WI Enums / Dataclasses
# ---------------------------------------------------------------------------

class WI_InterviewStage(Enum):
    INTRODUCTION = "introduction"
    COMMUNICATION = "communication"
    TECHNICAL = "technical"
    HR = "hr"
    COMPLETE = "complete"


@dataclass
class WI_ConversationExchange:
    timestamp: float
    stage: WI_InterviewStage
    ai_message: str
    user_response: str = ""
    transcript_quality: float = 0.0
    concept: str = ""
    is_followup: bool = False
    answer_quality: str = "neutral"
    topic_category: str = ""
    expected_keywords: List[str] = field(default_factory=list)
    technical_accuracy: Optional[float] = None
    question_type: str = "general"


@dataclass
class WI_ConversationState:
    current_topic: str = ""
    last_question: str = ""
    last_user_response: str = ""
    last_pure_question: str = ""
    followups_on_topic: int = 0
    max_followups: int = 2
    topics_discussed: List[str] = field(default_factory=list)
    used_transitions: List[str] = field(default_factory=list)
    extracted_topics: List[str] = field(default_factory=list)
    user_mentioned_tech: List[str] = field(default_factory=list)


@dataclass
class WI_InterviewSession:
    session_id: str
    test_id: str
    student_id: int
    student_name: str
    session_key: str
    created_at: float
    last_activity: float
    current_stage: WI_InterviewStage = WI_InterviewStage.INTRODUCTION
    is_active: bool = True
    websocket: Optional[Any] = None
    content_context: str = ""
    fragment_keys: List[str] = field(default_factory=list)
    current_concept: Optional[str] = None
    fragment_manager: Optional[Any] = None
    exchanges: List[WI_ConversationExchange] = field(default_factory=list)
    round_start_times: Dict[str, float] = field(default_factory=dict)
    questions_per_round: Dict[str, int] = field(default_factory=lambda: {"introduction": 0, "communication": 0, "technical": 0, "hr": 0})
    concept_question_counts: Dict[str, int] = field(default_factory=dict)
    followup_questions: int = 0
    silence_prompt_count: int = 0
    current_difficulty: str = "medium"
    last_answer_quality: str = "neutral"
    conversation_state: WI_ConversationState = field(default_factory=WI_ConversationState)
    questions_asked: List[str] = field(default_factory=list)
    communication_topics_covered: List[str] = field(default_factory=list)
    technical_topics_covered: List[str] = field(default_factory=list)
    hr_topics_covered: List[str] = field(default_factory=list)
    introduction_completed: bool = False
    behavioral_questions_in_technical: int = 0
    last_was_repeat: bool = False

    silent_topics: List[str] = field(default_factory=list)
    topic_attempt_count: Dict[str, int] = field(default_factory=dict)
    used_behavioral_questions: List[str] = field(default_factory=list)
    used_hr_questions: List[str] = field(default_factory=list)
    previously_asked_hr_questions: List[str] = field(default_factory=list)
    technical_question_count: int = 0
    behavioral_question_count: int = 0

    # SEQUENTIAL TRACKING
    current_tech_index: int = 0
    current_hr_index: int = 0
    current_topic_index: int = 0
    tech_question_types_used: Dict[str, List[str]] = field(default_factory=dict)

    # Extracted from summaries
    extracted_technologies: List[str] = field(default_factory=list)
    extracted_topics_for_questions: List[str] = field(default_factory=list)
    extracted_projects: List[str] = field(default_factory=list)
    extracted_challenges: List[str] = field(default_factory=list)
    extracted_team_info: List[str] = field(default_factory=list)

    # For evaluation accuracy
    technical_answers: List[Dict[str, Any]] = field(default_factory=list)
    correct_answers: int = 0
    partial_answers: int = 0
    wrong_answers: int = 0

    is_finalized: bool = False
    last_hr_question_time: float = 0.0

    def __post_init__(self):
        self.interview_start_time = self.created_at
        logger.info(f"[WI] Session initialized. Interview start time: {self.interview_start_time}")

    def start_round(self, stage: WI_InterviewStage):
        current_time = time.time()
        logger.info(f"[WI] ===== STARTING ROUND: {stage.value} =====")
        self.round_start_times[stage.value] = current_time
        self.current_stage = stage
        self.conversation_state = WI_ConversationState()

    def get_round_elapsed_time(self) -> float:
        current_stage_value = self.current_stage.value
        current_time = time.time()
        if current_stage_value not in self.round_start_times:
            logger.warning(f"[WI] ⚠️ Round {current_stage_value} has no start time! Setting now.")
            self.round_start_times[current_stage_value] = current_time
            return 0.0
        return current_time - self.round_start_times[current_stage_value]

    def get_round_elapsed_minutes(self) -> float:
        return self.get_round_elapsed_time() / 60

    def get_total_interview_time_minutes(self) -> float:
        if not hasattr(self, 'interview_start_time') or self.interview_start_time is None:
            self.interview_start_time = self.created_at
        return (time.time() - self.interview_start_time) / 60

    def get_questions_in_current_round(self) -> int:
        return self.questions_per_round.get(self.current_stage.value, 0)

    def add_exchange(self, ai_message: str, user_response: str = "", quality: float = 0.0,
                     concept: str = "", is_followup: bool = False, answer_quality: str = "neutral",
                     expected_keywords: List[str] = None, technical_accuracy: float = None,
                     question_type: str = "general"):
        ex = WI_ConversationExchange(
            timestamp=time.time(), stage=self.current_stage, ai_message=ai_message,
            user_response=user_response, transcript_quality=quality, concept=concept,
            is_followup=is_followup, answer_quality=answer_quality,
            expected_keywords=expected_keywords or [], technical_accuracy=technical_accuracy,
            question_type=question_type
        )
        self.exchanges.append(ex)
        self.questions_per_round[self.current_stage.value] = self.questions_per_round.get(self.current_stage.value, 0) + 1
        self.questions_asked.append(ai_message)
        if '?' in ai_message:
            parts = ai_message.split('?')
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i].strip()
                if len(part) > 10:
                    for sep in ['. ', '! ', '\n']:
                        if sep in part: part = part.split(sep)[-1].strip()
                    self.conversation_state.last_pure_question = part + '?'
                    break
        else:
            self.conversation_state.last_pure_question = ai_message

    def update_last_response(self, user_response: str, quality: float,
                             answer_quality: str = "neutral", technical_accuracy: float = None):
        if self.exchanges:
            self.exchanges[-1].user_response = user_response
            self.exchanges[-1].answer_quality = answer_quality
            self.exchanges[-1].technical_accuracy = technical_accuracy
            if technical_accuracy is not None:
                if technical_accuracy >= 0.7:
                    self.correct_answers += 1
                elif technical_accuracy >= 0.4:
                    self.partial_answers += 1
                else:
                    self.wrong_answers += 1
        self.last_answer_quality = answer_quality

    def get_stage_conversation_history(self, stage: WI_InterviewStage, limit: int = 10) -> str:
        exs = [e for e in self.exchanges if e.stage == stage][-limit:]
        return "\n".join([f"Q: {e.ai_message}\nA: {e.user_response}" for e in exs if e.user_response])

    def get_questions_asked_in_round(self, stage: WI_InterviewStage) -> List[str]:
        return [e.ai_message for e in self.exchanges if e.stage == stage]

    def get_last_user_response(self) -> str:
        for ex in reversed(self.exchanges):
            if ex.user_response:
                return ex.user_response
        return ""
    
    def get_conversation_by_round(self):
        result = {"communication": [], "technical": [], "hr": []}
        for ex in self.exchanges:
            exchange_data = {
                "question": ex.ai_message,
                "answer": ex.user_response or "[NO RESPONSE]",
                "timestamp": ex.timestamp,
                "answer_quality": ex.answer_quality,
                "is_followup": ex.is_followup,
                "technical_accuracy": ex.technical_accuracy
            }
            if ex.stage == WI_InterviewStage.COMMUNICATION:
                result["communication"].append(exchange_data)
            elif ex.stage == WI_InterviewStage.TECHNICAL:
                result["technical"].append(exchange_data)
            elif ex.stage == WI_InterviewStage.HR:
                result["hr"].append(exchange_data)
        return result


# ---------------------------------------------------------------------------
# WI Client Manager & Fragment Manager
# ---------------------------------------------------------------------------

class WI_SharedClientManager:
    def __init__(self):
        self.openai_client: Optional[AsyncOpenAI] = None
        self.groq_client: Optional[AsyncGroq] = None
        self.executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_MAX_WORKERS)
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise Exception("OPENAI_API_KEY not found in environment")
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise Exception("GROQ_API_KEY not found in environment")
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.groq_client = AsyncGroq(api_key=groq_key)
        self._initialized = True
        logger.info("[WI] AI clients initialized")

    async def close_connections(self):
        if self.openai_client:
            await self.openai_client.close()
        if self.groq_client:
            await self.groq_client.close()
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("[WI] AI clients closed")


# global WI shared clients
wi_shared_clients = WI_SharedClientManager()


class WI_EnhancedInterviewFragmentManager:
    def __init__(self, client_manager, session):
        self.client_manager = client_manager
        self.session = session

    def initialize_fragments(self, summaries) -> bool:
        if not summaries:
            return False
        self.session.content_context = "\n".join([s.get("summary", "") for s in summaries])
        self._extract_summary_info(self.session.content_context)
        self.session.start_round(WI_InterviewStage.INTRODUCTION)
        return True

    def _extract_summary_info(self, content: str):
        """Extract DETAILED topics from summaries for personalized questions."""
        content_lower = content.lower()

        sap_keywords = ["sap", "abap", "fiori", "hana", "s/4hana", "s4hana", "mm", "sd", "fico", "pp", "wm", "ewm", "ariba", "successfactors", "bw", "btp", "t-code", "tcode", "transaction", "idoc", "bapi", "rfc", "smartforms", "sapscript", "odata", "client administration", "scc4", "sccl", "scc3", "basis"]
        developer_keywords = ["python", "javascript", "react", "node", "fastapi", "django", "flask", "mongodb", "mysql", "postgresql", "docker", "kubernetes", "aws", "azure", "java", "spring", "typescript", "angular", "vue", "express", "api", "rest", "graphql"]

        sap_matches = [k for k in sap_keywords if k in content_lower]
        dev_matches = [k for k in developer_keywords if k in content_lower]

        # Extract specific topics from summary content
        self.session.extracted_topics_for_questions = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith('#') or line.endswith(':') or
                any(word in line.lower() for word in ['understanding', 'creating', 'configuring', 'implementing', 'troubleshooting', 'best practices', 'types of', 'step-by-step'])
            ):
                topic = line.strip('#').strip('0123456789.').strip(':').strip()
                if 5 < len(topic) < 100:
                    self.session.extracted_topics_for_questions.append(topic)

        concept_patterns = [
            r"(?:about|understand|learn)\s+(.+?)(?:\.|,|and|$)",
            r"(?:creating|configuring|implementing)\s+(.+?)(?:\.|,|and|$)",
            r"(?:using|with)\s+([A-Z][a-zA-Z0-9\s]+)(?:\.|,|and|$)",
            r"(?:T-code|transaction)\s+([A-Z0-9]+)",
        ]
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if 3 < len(match) < 50:
                    self.session.extracted_topics_for_questions.append(match.strip())

        seen = set()
        unique_topics = []
        for topic in self.session.extracted_topics_for_questions:
            topic_lower = topic.lower()
            if topic_lower not in seen and len(topic) > 5:
                seen.add(topic_lower)
                unique_topics.append(topic)
        self.session.extracted_topics_for_questions = unique_topics[:20]

        if len(sap_matches) > len(dev_matches):
            self.session.extracted_technologies = list(set(sap_matches))[:15]
            logger.info(f"[WI] Detected SAP track - Technologies: {self.session.extracted_technologies}")
        elif len(dev_matches) > 0:
            self.session.extracted_technologies = list(set(dev_matches))[:15]
            logger.info(f"[WI] Detected Developer track - Technologies: {self.session.extracted_technologies}")
        else:
            self.session.extracted_technologies = []
            logger.info("[WI] No specific tech detected")

        project_patterns = [r"worked on (.+?)(?:\.|,|and)", r"built (.+?)(?:\.|,|and)", r"developed (.+?)(?:\.|,|and)", r"implemented (.+?)(?:\.|,|and)", r"created (.+?)(?:\.|,|and)", r"configured (.+?)(?:\.|,|and)", r"managed (.+?)(?:\.|,|and)"]
        projects = []
        for pattern in project_patterns:
            projects.extend(re.findall(pattern, content_lower))
        self.session.extracted_projects = list(set(projects))[:10]

        challenge_patterns = [r"challenge.*?was (.+?)(?:\.|,)", r"difficult.*?(.+?)(?:\.|,)", r"problem.*?(.+?)(?:\.|,)", r"issue.*?was (.+?)(?:\.|,)", r"troubleshoot.*?(.+?)(?:\.|,)"]
        challenges = []
        for pattern in challenge_patterns:
            challenges.extend(re.findall(pattern, content_lower))
        self.session.extracted_challenges = list(set(challenges))[:5]

        if any(word in content_lower for word in ["team", "collaborate", "together", "group", "lead"]):
            self.session.extracted_team_info = ["worked in team"]

        logger.info(f"[WI] Extracted Topics for Questions: {self.session.extracted_topics_for_questions[:5]}")
        logger.info(f"[WI] Extracted Technologies: {self.session.extracted_technologies[:5]}")
        logger.info(f"[WI] Extracted Projects: {self.session.extracted_projects[:3]}")

    def should_continue_round(self, stage) -> bool:
        if stage == WI_InterviewStage.INTRODUCTION:
            return not self.session.introduction_completed
        duration = ROUND_DURATIONS.get(stage.value, 600)
        return self.session.get_round_elapsed_time() < duration

    def get_round_time_remaining(self) -> float:
        duration = ROUND_DURATIONS.get(self.session.current_stage.value, 600)
        return max(0, duration - self.session.get_round_elapsed_time())

    def add_question(self, question, concept, is_followup=False):
        pass

# =============================================================================
# HUMAN VOICE DETECTION + AUDIO PREPROCESSING + DEVICE HEALTH MONITOR
# (Headphone/Bluetooth support)
# =============================================================================

class HumanVoiceDetector:
    """Detects human voice and rejects non-human sounds (TV, fan, traffic, music)."""
    VOICE_FREQ_LOW = 60
    VOICE_FREQ_HIGH = 4000
    VOICE_ENERGY_THRESHOLD = 0.005
    VOICE_RATIO_THRESHOLD = 0.20
    ZCR_LOW = 0.01
    ZCR_HIGH = 0.45
    MIN_CONFIDENCE = 0.20

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def audio_bytes_to_numpy(self, audio_data):
        try:
            try:
                with io.BytesIO(audio_data) as audio_io:
                    with wave.open(audio_io, 'rb') as wav:
                        self.sample_rate = wav.getframerate()
                        n_channels = wav.getnchannels()
                        sampwidth = wav.getsampwidth()
                        frames = wav.readframes(wav.getnframes())
                        if sampwidth == 2: samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sampwidth == 4: samples = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else: samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                        if n_channels > 1: samples = samples.reshape(-1, n_channels).mean(axis=1)
                        return samples
            except Exception:
                pass
            try:
                target_sr = 16000
                result = subprocess.run(
                    ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', str(target_sr), '-ac', '1', 'pipe:1'],
                    input=audio_data, capture_output=True, timeout=10
                )
                if result.returncode == 0 and len(result.stdout) > 0:
                    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
                    self.sample_rate = target_sr
                    return samples
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
            return None
        except Exception as e:
            logger.error(f"[VAD] Audio conversion failed: {e}")
            return None

    def _spectral_voice_ratio(self, samples):
        try:
            window = np.hanning(len(samples))
            fft_result = np.abs(np.fft.rfft(samples * window))
            freqs = np.fft.rfftfreq(len(samples), 1.0 / self.sample_rate)
            total_energy = np.sum(fft_result ** 2)
            if total_energy < 1e-10: return 0.0
            voice_mask = (freqs >= self.VOICE_FREQ_LOW) & (freqs <= self.VOICE_FREQ_HIGH)
            return np.sum(fft_result[voice_mask] ** 2) / total_energy
        except Exception: return 0.0

    def _zero_crossing_rate(self, samples):
        try:
            if len(samples) < 2: return 0.0
            signs = np.sign(samples)
            return np.sum(np.abs(np.diff(signs)) > 0) / len(samples)
        except Exception: return 0.0

    def _speech_pattern_score(self, samples, frame_size=1024):
        try:
            n_frames = len(samples) // frame_size
            if n_frames < 3: return 0.5
            frame_energies = np.array([np.sqrt(np.mean(samples[i*frame_size:(i+1)*frame_size]**2)) for i in range(n_frames)])
            max_energy = np.max(frame_energies)
            if max_energy < 1e-6: return 0.0
            frame_energies /= max_energy
            energy_std = np.std(frame_energies)
            energy_mean = np.mean(frame_energies)
            voiced = frame_energies > (energy_mean * 0.5)
            transition_rate = np.sum(np.abs(np.diff(voiced.astype(int)))) / n_frames
            score = 0.0
            if 0.1 <= transition_rate <= 0.5: score += 0.5
            elif transition_rate < 0.1: score += 0.1
            else: score += 0.2
            if 0.15 <= energy_std <= 0.45: score += 0.5
            elif energy_std < 0.15: score += 0.1
            else: score += 0.2
            return score
        except Exception: return 0.5

    def is_human_voice(self, audio_data):
        samples = self.audio_bytes_to_numpy(audio_data)
        if samples is None or len(samples) < 1000: return False, 0.0, {"error": "too_short"}
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if rms < self.VOICE_ENERGY_THRESHOLD: return False, 0.0, {"rms": rms, "reason": "silence"}
        voice_ratio = self._spectral_voice_ratio(samples)
        zcr = self._zero_crossing_rate(samples)
        pattern = self._speech_pattern_score(samples)
        vr_score = min(voice_ratio / 0.6, 1.0) * 0.35 if voice_ratio >= self.VOICE_RATIO_THRESHOLD else (voice_ratio * 0.2)
        zcr_score = 0.0
        if self.ZCR_LOW <= zcr <= self.ZCR_HIGH:
            center = (self.ZCR_LOW + self.ZCR_HIGH) / 2
            deviation = abs(zcr - center) / (self.ZCR_HIGH - self.ZCR_LOW)
            zcr_score = (1.0 - deviation) * 0.25
        elif zcr < self.ZCR_LOW * 3:
            zcr_score = 0.08
        pat_score = pattern * 0.40
        confidence = vr_score + zcr_score + pat_score
        is_voice = confidence >= self.MIN_CONFIDENCE
        logger.info(f"[VAD] is_voice={is_voice} conf={confidence:.2f} [ratio={voice_ratio:.2f} zcr={zcr:.3f} pattern={pattern:.2f} rms={rms:.4f}]")
        return is_voice, confidence, {"rms": round(rms, 4), "voice_ratio": round(voice_ratio, 3), "confidence": round(confidence, 3), "is_voice": is_voice}


class AudioPreprocessor:
    """Gently cleans audio before Whisper: trim silence + normalize. No spectral manipulation."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self._vad = HumanVoiceDetector(sample_rate)

    def _normalize(self, samples):
        max_val = np.max(np.abs(samples))
        return samples * (0.8 / max_val) if max_val > 1e-6 else samples

    def _trim_silence(self, samples, threshold=0.003, pad=3200):
        above = np.where(np.abs(samples) > threshold)[0]
        if len(above) == 0: return samples
        start = max(0, above[0] - pad)
        end = min(len(samples), above[-1] + pad)
        if (end - start) < len(samples) * 0.5: return samples
        return samples[start:end]

    def _to_wav_bytes(self, samples):
        pcm = (samples * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(self.sample_rate)
            wav.writeframes(pcm.tobytes())
        return buf.getvalue()

    def preprocess(self, audio_data):
        try:
            samples = self._vad.audio_bytes_to_numpy(audio_data)
            if samples is None: return audio_data
            orig_len = len(samples)
            samples = self._trim_silence(samples)
            samples = self._normalize(samples)
            logger.info(f"[AUDIO] Preprocessed: {orig_len} -> {len(samples)} samples")
            return self._to_wav_bytes(samples)
        except Exception as e:
            logger.error(f"[AUDIO] Preprocessing failed: {e}"); return audio_data


class AudioDeviceHealthMonitor:
    """Detects Bluetooth/headphone disconnect. Keeps interview alive."""
    GRACE_PERIOD = 10
    MAX_BAD_BEFORE_WARN = 3
    def __init__(self): self.last_good_time = None; self.consecutive_bad = 0; self.disconnect_detected = False
    def check_audio_health(self, audio_data):
        try:
            if not audio_data or len(audio_data) < 50: self.consecutive_bad += 1; return self._decide("empty_audio")
            vad = HumanVoiceDetector(); samples = vad.audio_bytes_to_numpy(audio_data)
            if samples is None: self.consecutive_bad += 1; return self._decide("unreadable")
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if rms < 0.0005: self.consecutive_bad += 1; return self._decide("dead_silence")
            if rms > 0.9: self.consecutive_bad += 1; return self._decide("static")
            self.consecutive_bad = 0; self.disconnect_detected = False; self.last_good_time = time.time()
            return {"healthy": True, "action": "continue"}
        except Exception as e:
            logger.error(f"[DEVICE] Health check error: {e}"); return {"healthy": True, "action": "continue"}
    def _decide(self, issue):
        if self.consecutive_bad >= self.MAX_BAD_BEFORE_WARN:
            self.disconnect_detected = True
            if self.last_good_time:
                elapsed = time.time() - self.last_good_time
                if elapsed < self.GRACE_PERIOD:
                    return {"healthy": False, "action": "wait_reconnect", "issue": issue, "message": f"Audio device may have disconnected. Waiting {int(self.GRACE_PERIOD - elapsed)}s..."}
            return {"healthy": False, "action": "warn_user", "issue": issue, "message": "Audio device disconnected. Please check your headphones/microphone."}
        return {"healthy": True, "action": "continue", "issue": issue}
    def reset(self): self.last_good_time = time.time(); self.consecutive_bad = 0; self.disconnect_detected = False

# ---------------------------------------------------------------------------
# WI Audio Processor (with hallucination detection)
# ---------------------------------------------------------------------------

class WI_OptimizedAudioProcessor:
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self.voice_detector = HumanVoiceDetector()
        self.audio_preprocessor = AudioPreprocessor()
        self.device_monitor = AudioDeviceHealthMonitor()
        self.HALLUCINATION_PHRASES = [
            # Whisper prompt echoes
            "the speaker is answering questions about their",
            "interview response",
            "the speaker is answering",
            "answering questions about their work",
            "work experience, technical skills",
            "technical skills, and projects",
            # Standard Whisper hallucinations
            "thank you for watching", "thanks for watching", "please subscribe",
            "like and subscribe", "see you in the next", "bye bye", "goodbye",
            "thank you for listening", "the end", "music", "applause", "laughter",
            "silence", "inaudible", "unintelligible", "foreign",
            "speaking foreign language", "don't forget to subscribe", "hit the bell",
            "leave a comment", "check out my", "link in description", "sponsored by",
        ]

    def _decode_to_wav(self, audio_data: bytes) -> bytes:
        """Decode any audio format to WAV PCM. Runs ffmpeg once."""
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            return audio_data
        try:
            result = subprocess.run(
                ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
                input=audio_data, capture_output=True, timeout=10
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                return result.stdout
            return None
        except: return None

    async def transcribe_audio_fast(self, audio_data: bytes) -> Tuple[str, float]:
        await self.client_manager.initialize()
        if len(audio_data) < 2000: return "", 0.0

        decoded_wav = self._decode_to_wav(audio_data)
        if decoded_wav is None: return "", 0.0

        # Device Health Check
        device_health = self.device_monitor.check_audio_health(decoded_wav)
        if not device_health["healthy"]:
            if device_health["action"] == "warn_user": return "__DEVICE_DISCONNECTED__", 0.0
            elif device_health["action"] == "wait_reconnect": return "__DEVICE_RECONNECTING__", 0.0

        # Human Voice Detection
        is_voice, vad_confidence, _ = self.voice_detector.is_human_voice(decoded_wav)
        if not is_voice: return "", 0.0

        # Preprocess Audio
        processed_audio = self.audio_preprocessor.preprocess(decoded_wav)

        # Transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(processed_audio); temp_path = tf.name
        try:
            with open(temp_path, "rb") as f: audio_bytes = f.read()
            tr = await self.client_manager.groq_client.audio.transcriptions.create(
                file=(temp_path, audio_bytes), model="whisper-large-v3-turbo", language="en",
                prompt="um, uh, like, okay, so, yeah, right, actually, basically"
            )
            raw_text = tr.text.strip() if hasattr(tr, 'text') else ""
            if not raw_text: return "", 0.0
            cleaned_text = self._remove_hallucinations(raw_text)
            confidence = (self._calculate_confidence(cleaned_text) + vad_confidence) / 2
            if confidence < 0.3: return "", confidence
            final_text = self._final_cleanup(cleaned_text)
            if len(final_text.split()) < 2: return "", 0.2
            self.device_monitor.consecutive_bad = 0
            self.device_monitor.disconnect_detected = False
            return final_text, confidence
        except Exception as e:
            logger.error(f"[WI] Transcription error: {e}"); return "", 0.0
        finally:
            try: os.unlink(temp_path)
            except: pass

    def _remove_hallucinations(self, text: str) -> str:
        if not text:
            return ""
        result = text.lower()
        for phrase in self.HALLUCINATION_PHRASES:
            result = result.replace(phrase, "")
        cleaned = ""
        for char in result:
            if char.isascii() or char in ".,?!'\"- ":
                cleaned += char
        cleaned = re.sub(r'[.]{2,}', '.', cleaned)
        cleaned = re.sub(r'[,]{2,}', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        words = cleaned.split()
        if len(words) > 3:
            deduped = []; repeat_count = 0; last_word = ""
            for word in words:
                if word.lower() == last_word.lower():
                    repeat_count += 1
                    if repeat_count <= 1:
                        deduped.append(word)
                else:
                    repeat_count = 0
                    deduped.append(word)
                last_word = word
            cleaned = " ".join(deduped)
        return cleaned.strip()

    def _calculate_confidence(self, text: str) -> float:
        if not text:
            return 0.0
        words = text.split()
        word_count = len(words)
        if word_count < 2:
            return 0.1
        real_speech_indicators = {
            'i', 'we', 'my', 'our', 'the', 'this', 'that', 'is', 'are', 'was', 'were',
            'have', 'has', 'had', 'do', 'did', 'work', 'worked', 'use', 'used',
            'project', 'system', 'data', 'client', 'team', 'experience', 'years',
            'developed', 'created', 'managed', 'handled', 'implemented', 'configured',
            'learned', 'know', 'think', 'believe', 'like', 'want', 'need',
            'yes', 'no', 'because', 'so', 'and', 'but', 'or', 'for', 'with'
        }
        text_lower = text.lower()
        indicator_count = sum(1 for word in real_speech_indicators if word in text_lower)
        indicator_score = min(indicator_count / 5, 1.0)
        length_score = min(word_count / 10, 1.0)
        gibberish_penalty = 0.0
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.5:
            gibberish_penalty += 0.3
        if re.search(r'[a-z]{10,}', text_lower):
            gibberish_penalty += 0.2
        return max(0.0, min(1.0, (indicator_score * 0.5 + length_score * 0.5) - gibberish_penalty))

    def _final_cleanup(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        if text and text[-1] not in '.?!':
            text += '.'
        return text


# ---------------------------------------------------------------------------
# WI Conversation Manager  (main logic)
# ---------------------------------------------------------------------------

class WI_OptimizedConversationManager:
    def __init__(self, client_manager):
        self.client_manager = client_manager

    # --- Utility helpers ---

    def _detect_user_intent(self, user_response: str) -> str:
        r = user_response.lower().strip()
        if any(p in r for p in ["skip", "next question", "move on", "next one", "pass"]):
            return "skip"
        if any(p in r for p in ["repeat", "say again", "can you repeat", "what was the question"]):
            return "repeat"
        if any(p in r for p in ["i don't know", "i'm not sure", "no idea", "can't answer", "don't remember"]):
            return "dont_know"
        return "normal"

    def _is_gibberish(self, text: str) -> bool:
        if not text:
            return True
        ascii_chars = sum(1 for c in text if c.isascii())
        if len(text) > 0 and (ascii_chars / len(text)) < 0.8:
            return True
        words = text.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return True
        nonsense_patterns = [r'(.)\1{4,}', r'\b(\w+)\s+\1\s+\1\s+\1']
        for pattern in nonsense_patterns:
            if re.search(pattern, text.lower()):
                return True
        hallucinations = ["thank you for watching", "please subscribe", "like and subscribe", "see you next time", "bye bye bye", "youtube", "mcdonald"]
        text_lower = text.lower()
        if any(h in text_lower for h in hallucinations):
            return True
        return False

    def _assess_answer_quality(self, user_response: str) -> str:
        if not user_response:
            return "silence"
        if self._is_gibberish(user_response):
            logger.warning(f"[WI] Detected gibberish: {user_response[:100]}...")
            return "gibberish"
        intent = self._detect_user_intent(user_response)
        if intent != "normal":
            return "skip" if intent == "skip" else ("repeat" if intent == "repeat" else "cant_answer")
        words = len(user_response.split())
        if words <= 3:
            return "weak"
        strong = ["because", "therefore", "for example", "specifically", "implemented", "experience", "i think", "used", "worked", "built", "designed", "configured", "created", "developed", "managed", "handled"]
        if words >= 20 and any(k in user_response.lower() for k in strong):
            return "strong"
        return "neutral" if words >= 10 else "weak"

    async def _evaluate_technical_accuracy(self, session, question: str, answer: str, expected_keywords: List[str]) -> float:
        if not answer or len(answer.split()) < 3:
            return 0.0
        await self.client_manager.initialize()
        prompt = f"""Evaluate this technical interview answer.

Question: {question}
Answer: {answer}
Context (user's work): {session.content_context[:500] if session.content_context else 'General'}

Rate accuracy from 0.0 to 1.0:
- 1.0 = Correct, detailed, shows understanding
- 0.7 = Mostly correct, some details
- 0.5 = Partially correct, missing key points
- 0.3 = Vague or mostly incorrect
- 0.0 = Wrong or no real answer

Reply with ONLY a number between 0.0 and 1.0"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=10
            )
            score_text = resp.choices[0].message.content.strip()
            score = float(re.search(r"(\d+\.?\d*)", score_text).group(1))
            return min(max(score, 0.0), 1.0)
        except:
            answer_lower = answer.lower()
            if expected_keywords:
                matches = sum(1 for k in expected_keywords if k.lower() in answer_lower)
                return min(matches / len(expected_keywords), 1.0)
            return 0.5 if len(answer.split()) > 10 else 0.3

    def _extract_topics_from_response(self, response: str, session=None) -> List[str]:
        response_lower = response.lower()
        if session and session.extracted_technologies:
            return [t for t in session.extracted_technologies if t in response_lower]
        all_tech = ["python", "javascript", "react", "node", "api", "database", "mongodb", "mysql", "docker", "aws", "frontend", "backend", "testing", "debugging", "git", "sap", "abap", "fiori", "hana", "mm", "sd", "fico"]
        return [t for t in all_tech if t in response_lower]

    def _get_unique_transition(self, session) -> str:
        used = session.conversation_state.used_transitions
        available = [t for t in COMMUNICATION_TRANSITIONS if t not in used] or COMMUNICATION_TRANSITIONS
        t = random.choice(available)
        session.conversation_state.used_transitions.append(t)
        if len(session.conversation_state.used_transitions) > 10:
            session.conversation_state.used_transitions = session.conversation_state.used_transitions[-10:]
        return t

    def _should_followup(self, session, quality) -> bool:
        if quality in ["weak", "cant_answer", "silence", "skip", "repeat"]:
            return False
        if session.conversation_state.followups_on_topic >= 2:
            return False
        return random.random() < (0.6 if quality == "strong" else 0.4)

    def _extract_question_from_response(self, ai_message):
        if not ai_message:
            return "Could you please repeat your answer?"
        cleaned = ai_message.strip()
        prefixes_to_remove = ["Of course! The question was:", "Sure, let me repeat:", "No problem! Here it is again:", "Let me repeat that:", "Here's the question again:"]
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        if '?' in cleaned:
            parts = cleaned.split('?')
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i].strip()
                if len(part) > 10:
                    for sep in ['. ', '! ', '\n']:
                        if sep in part:
                            part = part.split(sep)[-1].strip()
                    return part + '?'
        return cleaned

    def _adjust_difficulty(self, session, quality):
        if session.current_stage != WI_InterviewStage.TECHNICAL:
            return
        if quality == "strong":
            session.current_difficulty = "hard" if session.current_difficulty == "medium" else "medium"
        elif quality in ["weak", "cant_answer"]:
            session.current_difficulty = "easy"

    def _is_similar_question(self, q1: str, q2: str) -> bool:
        q1_clean = q1.lower().strip().rstrip('?').strip()
        q2_clean = q2.lower().strip().rstrip('?').strip()
        if q1_clean == q2_clean:
            return True
        common_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'your', 'you', 'can', 'do', 'did', 'does', 'tell', 'me', 'about', 'describe', 'explain'}
        words1 = set(q1_clean.split()) - common_words
        words2 = set(q2_clean.split()) - common_words
        if len(words1) == 0 or len(words2) == 0:
            return False
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        return overlap / min_len > 0.4

    def _normalize_question(self, question: str) -> str:
        if not question:
            return ""
        q = question.lower().strip().rstrip('?').strip()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'your', 'you', 'can', 'do', 'did', 'does', 'tell', 'me', 'about', 'describe', 'explain', 'please', 'could', 'would', 'should', 'to', 'in', 'on', 'for', 'with'}
        words = [w for w in q.split() if w not in stop_words and len(w) > 2]
        return ' '.join(sorted(words))

    def _get_encouragement(self) -> str:
        return random.choice([
            "That's a great explanation!", "Excellent point!", "Well explained!",
            "Good answer!", "That's exactly right!",
            "Nice! You clearly have good experience with this.",
            "Great insight!", "That's impressive!",
        ])

    # --- Question generators ---

    async def _generate_communication_question(self, session, is_first=False) -> str:
        await self.client_manager.initialize()
        asked = session.get_questions_asked_in_round(WI_InterviewStage.COMMUNICATION)
        topics = [
            "weekend plans", "favorite food", "travel dreams", "morning routine",
            "favorite movie or show", "music preferences", "childhood memories",
            "dream vacation", "favorite season", "cooking or eating out",
            "pets or animals", "sports or fitness", "books or reading",
            "family traditions", "city or countryside", "coffee or tea",
            "early bird or night owl", "relaxation methods", "learning something new",
            "favorite holiday", "hometown memories", "friends and social life",
            "dream job as a child", "favorite game", "weather preferences"
        ]
        used_topics = session.communication_topics_covered
        available = [t for t in topics if t not in used_topics] or topics
        chosen_topic = random.choice(available)
        session.communication_topics_covered.append(chosen_topic)
        prompt = f"""Generate ONE friendly casual question about: {chosen_topic}
Keep it natural like a human conversation.
Already asked (DO NOT repeat): {asked[-5:]}
MAX 12 words. Just the question."""
        resp = await self.client_manager.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9, max_tokens=30)
        q = resp.choices[0].message.content.strip()
        q_lower = q.lower()
        for asked_q in asked:
            if self._is_similar_question(q_lower, asked_q.lower()):
                q = random.choice([
                    f"What do you think about {chosen_topic}?",
                    f"Tell me about your {chosen_topic}?",
                    f"How do you feel about {chosen_topic}?",
                ])
                break
        return q if '?' in q else q + "?"

    async def _generate_dynamic_ack(self, context: str, tone: str = "friendly") -> str:
        await self.client_manager.initialize()
        prompts = {
            "weak": "Generate ONE short understanding response when someone gives unclear answer. Like 'I see, let me try another question' or 'Okay, let's move on'. MAX 8 words.",
            "good": "Generate ONE short positive acknowledgment like 'That's nice!' or 'Good to know!' MAX 5 words.",
            "technical_good": "Generate ONE short praise for good technical answer like 'Well explained!' or 'Good point!' MAX 5 words.",
            "technical_weak": "Generate ONE short understanding response for unclear technical answer. MAX 8 words.",
            "cant_answer": "Generate ONE short supportive response when someone can't answer, like 'No problem, let's try something else'. MAX 10 words.",
            "transition": "Generate ONE short transition phrase like 'Interesting!' or 'Nice!' MAX 3 words.",
            "hr": "Generate ONE short professional acknowledgment like 'Thank you for sharing' or 'Good point'. MAX 5 words.",
        }
        prompt = prompts.get(tone, prompts["good"])
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9, max_tokens=20)
            ack = resp.choices[0].message.content.strip().replace('"', '').replace("'", "")
            if not ack.endswith(('!', '.', '?')):
                ack += '!'
            return ack
        except:
            fallbacks = {"weak": "I see. Let me ask something else.", "good": "Nice!", "technical_good": "Good explanation!", "technical_weak": "Okay, let's try another one.", "cant_answer": "No problem! Let's move on.", "transition": "Interesting!", "hr": "Thank you."}
            return fallbacks.get(tone, "Okay!")

    async def _generate_communication_followup(self, session, user_response: str) -> str:
        await self.client_manager.initialize()
        prompt = f"""User said: "{user_response[:100]}"
Generate a short follow-up question. MAX 12 words."""
        resp = await self.client_manager.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=30)
        q = resp.choices[0].message.content.strip()
        return q if '?' in q else q + "?"

    async def _generate_technical_question(self, session, user_response: str = "", include_behavioral: bool = False) -> Tuple[str, List[str]]:
        """Generate technical question using 40 TEMPLATES based on MongoDB summary."""
        await self.client_manager.initialize()
        if not hasattr(session, 'total_technical_questions_generated'):
            session.total_technical_questions_generated = 0
        session.total_technical_questions_generated += 1
        if not hasattr(session, 'used_technical_templates'):
            session.used_technical_templates = []
        if not hasattr(session, 'used_behavioral_templates'):
            session.used_behavioral_templates = []
        if not hasattr(session, 'current_template_index'):
            session.current_template_index = 0

        all_asked_questions = list(session.questions_asked)

        # Analyze user's last response
        response_quality = "none"; should_followup = False; prefix = ""
        if user_response:
            response_lower = user_response.lower().strip()
            word_count = len(response_lower.split())
            bad_indicators = ["thank you", "skip", "next", "i don't know", "no idea", "can't answer", "pass", "move on", "bye", "i can't", "don't understand", "not sure", "no clue", "don't remember", "hello", "hi", "okay", "ok", "yes", "no"]
            words = response_lower.split(); unique_words = set(words)
            is_repetitive = len(words) > 3 and len(unique_words) < len(words) * 0.4
            tech_keywords = ['sap', 'client', 'transaction', 't-code', 'config', 'system', 'data', 'user', 'table', 'module', 'basis', 'abap', 'fiori', 'report', 'program', 'function', 'process', 'implement', 'configure', 'setup', 'install', 'error', 'issue', 'problem', 'solution', 'project', 'team', 'work', 'experience', 'used', 'created', 'developed', 'managed', 'handled', 'deployed']
            has_tech_content = any(kw in response_lower for kw in tech_keywords)
            irrelevant = ['mcdonald', 'youtube', 'google', 'phone', 'rupee', 'otp', 'video', 'movie', 'song', 'food', 'hospital', 'cookie']
            has_irrelevant = any(irr in response_lower for irr in irrelevant)
            is_bad_answer = (word_count < 8 or is_repetitive or has_irrelevant or any(indicator == response_lower.strip() for indicator in bad_indicators) or (word_count < 15 and not has_tech_content))
            if is_bad_answer:
                response_quality = "bad"
                prefix = "I think you might not be familiar with that topic. No worries, let me ask you something different. "
                if session.exchanges:
                    last_q = session.exchanges[-1].ai_message.lower()
                    for tech in (session.extracted_technologies or []):
                        if tech.lower() in last_q and tech not in session.silent_topics:
                            session.silent_topics.append(tech)
                            logger.info(f"[WI] Skipping topic '{tech}' - user doesn't know it")
                            break
            elif word_count >= 20 and has_tech_content:
                response_quality = "good"; should_followup = True
                prefix = self._get_encouragement() + " "

        if should_followup and user_response:
            follow_up = await self._generate_followup_from_answer(session, user_response, all_asked_questions)
            if follow_up:
                return f"{prefix}{follow_up}", ["followup"]

        technologies = [t for t in (session.extracted_technologies or []) if t not in session.silent_topics]
        if not technologies:
            technologies = ["your work experience", "your daily tasks", "your technical skills"]

        total_qs = session.technical_question_count + session.behavioral_question_count
        should_be_behavioral = (include_behavioral and total_qs > 0 and total_qs % 4 == 3 and len(session.used_behavioral_templates) < len(TECHNICAL_BEHAVIORAL_QUESTIONS))
        if should_be_behavioral:
            session.behavioral_question_count += 1
            return await self._generate_behavioral_from_template(session, technologies, all_asked_questions, prefix)

        session.technical_question_count += 1
        tech_idx = session.current_tech_index % len(technologies)
        chosen_tech = technologies[tech_idx]
        session.current_tech_index += 1

        question = None
        for i, template in enumerate(TECHNICAL_QUESTION_TEMPLATES):
            template_key = (i, chosen_tech.lower())
            if template_key not in session.used_technical_templates:
                question = template.format(tech=chosen_tech)
                if not any(self._is_similar_question(question.lower(), aq.lower()) for aq in all_asked_questions):
                    session.used_technical_templates.append(template_key)
                    break
                else:
                    question = None

        if not question:
            session.current_template_index = 0
            for tech in technologies:
                if tech != chosen_tech:
                    for i, template in enumerate(TECHNICAL_QUESTION_TEMPLATES):
                        template_key = (i, tech.lower())
                        if template_key not in session.used_technical_templates:
                            question = template.format(tech=tech)
                            chosen_tech = tech
                            if not any(self._is_similar_question(question.lower(), aq.lower()) for aq in all_asked_questions):
                                session.used_technical_templates.append(template_key)
                                break
                            else:
                                question = None
                    if question:
                        break

        if not question:
            question = await self._generate_dynamic_question_from_summary(session, chosen_tech, all_asked_questions)

        full_question = f"{prefix}{question}" if prefix else question
        if chosen_tech not in session.technical_topics_covered:
            session.technical_topics_covered.append(chosen_tech)
        return full_question, [chosen_tech]

    async def _generate_behavioral_from_template(self, session, technologies: List[str], all_asked: List[str], prefix: str = "") -> Tuple[str, List[str]]:
        tech_idx = session.current_tech_index % len(technologies)
        chosen_tech = technologies[tech_idx]
        question = None
        for i, template in enumerate(TECHNICAL_BEHAVIORAL_QUESTIONS):
            if i not in session.used_behavioral_templates:
                question = template.format(tech=chosen_tech, project=chosen_tech)
                if not any(self._is_similar_question(question.lower(), aq.lower()) for aq in all_asked):
                    session.used_behavioral_templates.append(i)
                    break
                else:
                    question = None
        if not question:
            question = f"Tell me about a challenging experience you had while working with {chosen_tech}."
        full_question = f"{prefix}{question}" if prefix else question
        return full_question, [chosen_tech]

    async def _generate_dynamic_question_from_summary(self, session, tech: str, all_asked: List[str]) -> str:
        await self.client_manager.initialize()
        summary = session.content_context or "General technical work"
        prompt = f"""Generate ONE unique technical interview question.

CANDIDATE'S WORK SUMMARY:
{summary[:1500]}

TOPIC: {tech}

ALREADY ASKED (DO NOT REPEAT):
{chr(10).join(all_asked[-10:])}

Generate a specific question about their practical experience.
MAX 20 words. Just the question:"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=50)
            question = resp.choices[0].message.content.strip().strip('"').strip("'")
            if not question.endswith('?'):
                question += '?'
            return question
        except Exception as e:
            logger.error(f"Error generating dynamic question: {e}")
            return f"Tell me more about your experience with {tech}?"

    async def _generate_followup_from_answer(self, session, user_response: str, all_asked: List[str]) -> Optional[str]:
        await self.client_manager.initialize()
        prompt = f"""The candidate gave this good answer: "{user_response[:300]}"

Generate ONE short follow-up question to dig deeper into what they mentioned.
Ask about: Specific details they mentioned, Challenges they faced, How they solved problems, Results or outcomes

ALREADY ASKED (DO NOT REPEAT):
{chr(10).join(all_asked[-5:])}

MAX 15 words. Just the question:"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=40)
            question = resp.choices[0].message.content.strip()
            if not question.endswith('?'):
                question += '?'
            is_duplicate = any(self._is_similar_question(question.lower(), aq.lower()) for aq in all_asked)
            if not is_duplicate:
                return question
        except:
            pass
        return None

    async def _generate_technical_behavioral_question(self, session) -> Tuple[str, List[str]]:
        """Generate behavioral question for technical round - SEQUENTIAL selection."""
        all_asked = list(session.questions_asked)
        if not hasattr(session, 'asked_question_hashes'):
            session.asked_question_hashes = set()
            for q in all_asked:
                session.asked_question_hashes.add(self._normalize_question(q))
        primary_tech = session.extracted_technologies[0] if session.extracted_technologies else "your technical work"
        project_context = session.extracted_projects[0] if session.extracted_projects else "your projects"
        if not hasattr(session, 'behavioral_question_idx'):
            session.behavioral_question_idx = 0
        while session.behavioral_question_idx < len(TECHNICAL_BEHAVIORAL_QUESTIONS):
            template = TECHNICAL_BEHAVIORAL_QUESTIONS[session.behavioral_question_idx]
            session.behavioral_question_idx += 1
            try:
                question = template.format(tech=primary_tech, project=project_context)
            except:
                question = template.replace("{tech}", primary_tech).replace("{project}", project_context)
            q_hash = self._normalize_question(question)
            if q_hash not in session.asked_question_hashes:
                session.asked_question_hashes.add(q_hash)
                session.used_behavioral_questions.append(question)
                return question, ["behavioral"]
        if not hasattr(session, 'generic_behavioral_idx'):
            session.generic_behavioral_idx = 0
        while session.generic_behavioral_idx < len(GENERIC_BEHAVIORAL_QUESTIONS):
            question = GENERIC_BEHAVIORAL_QUESTIONS[session.generic_behavioral_idx]
            session.generic_behavioral_idx += 1
            q_hash = self._normalize_question(question)
            if q_hash not in session.asked_question_hashes:
                session.asked_question_hashes.add(q_hash)
                session.used_behavioral_questions.append(question)
                return question, ["behavioral"]
        await self.client_manager.initialize()
        question_num = len(session.used_behavioral_questions) + 1
        prompt = f"""Generate ONE unique behavioral interview question.
Candidate works with: {primary_tech}
ALREADY ASKED - DO NOT REPEAT:
{chr(10).join(session.used_behavioral_questions[-10:])}
Ask about: challenges, learning, teamwork, problem-solving
MAX 15 words. Just the question."""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9, max_tokens=40)
            question = resp.choices[0].message.content.strip()
            if not question.endswith('?'):
                question += '?'
            q_hash = self._normalize_question(question)
            if q_hash in session.asked_question_hashes:
                question = f"Share an experience where you overcame a challenge. (Q#{question_num})"
        except:
            question = f"Tell me about a learning experience in your career. (Q#{question_num})"
        session.asked_question_hashes.add(self._normalize_question(question))
        session.used_behavioral_questions.append(question)
        return question, ["behavioral"]

    async def _generate_hr_question(self, session, db_manager=None) -> Tuple[str, List[str]]:
        """Generate HR question from MongoDB collection with category distribution."""
        if not hasattr(session, 'asked_question_hashes'):
            session.asked_question_hashes = set()
            for q in session.questions_asked:
                session.asked_question_hashes.add(self._normalize_question(q))
        if not hasattr(session, 'hr_category_counts'):
            session.hr_category_counts = {'introduction': 0, 'behavioral': 0, 'leadership': 0, 'logical_thinking': 0}
        if not hasattr(session, 'hr_questions_by_category'):
            session.hr_questions_by_category = {}
        CATEGORY_LIMITS = {'introduction': 2, 'behavioral': 3, 'leadership': 3, 'logical_thinking': 2}

        if not session.previously_asked_hr_questions and db_manager:
            try:
                session.previously_asked_hr_questions = await db_manager.get_hr_questions_asked(session.student_id, limit=200)
                logger.info(f"[HR] Loaded {len(session.previously_asked_hr_questions)} previously asked HR questions")
                for q in session.previously_asked_hr_questions:
                    session.asked_question_hashes.add(self._normalize_question(q))
            except Exception as e:
                logger.warning(f"[HR] Could not load previous HR questions: {e}")
                session.previously_asked_hr_questions = []

        if not session.hr_questions_by_category:
            if db_manager:
                try:
                    await self._load_hr_questions_by_category(session, db_manager)
                except Exception as e:
                    logger.warning(f"[HR] Could not load from MongoDB: {e}")
            if not session.hr_questions_by_category:
                logger.warning("[HR] Using fallback questions")
                session.hr_questions_by_category = {'introduction': GENERIC_HR_QUESTIONS[:5], 'behavioral': HR_QUESTIONS_POOL[:5], 'leadership': HR_QUESTIONS_POOL[5:10], 'logical_thinking': HR_QUESTIONS_POOL[10:15]}

        total_hr_asked = sum(session.hr_category_counts.values())
        logger.info(f"[HR] Total HR questions asked so far: {total_hr_asked}")
        category_order = ['introduction', 'behavioral', 'leadership', 'logical_thinking']
        target_category = None
        for category in category_order:
            if session.hr_category_counts[category] < CATEGORY_LIMITS[category]:
                target_category = category
                break
        if target_category is None:
            logger.info("[HR] All category limits reached - HR round complete")
            return "Thank you! That concludes our HR round. You did great!", ["hr_complete"]

        logger.info(f"[HR] Asking from category: {target_category} (current: {session.hr_category_counts[target_category]}/{CATEGORY_LIMITS[target_category]})")
        category_questions = session.hr_questions_by_category.get(target_category, [])
        if not category_questions:
            for fallback_cat in category_order:
                if fallback_cat != target_category and session.hr_questions_by_category.get(fallback_cat):
                    category_questions = session.hr_questions_by_category[fallback_cat]
                    target_category = fallback_cat
                    break

        all_asked = set(session.used_hr_questions) | set(session.previously_asked_hr_questions)
        selected_question = None
        shuffled = category_questions.copy()
        random.shuffle(shuffled)
        for question in shuffled:
            q_normalized = self._normalize_question(question)
            if q_normalized not in session.asked_question_hashes:
                is_similar = any(self._is_similar_question(question.lower(), aq.lower()) for aq in all_asked)
                if not is_similar:
                    selected_question = question
                    break
        if not selected_question and category_questions:
            selected_question = random.choice(category_questions)
        if not selected_question:
            fallback_questions = {'introduction': "What motivated you to choose your career path?", 'behavioral': "Tell me about a challenging situation you faced at work.", 'leadership': "Describe a time when you took initiative on a project.", 'logical_thinking': "How do you approach solving complex problems?"}
            selected_question = fallback_questions.get(target_category, "What are your career goals?")

        session.asked_question_hashes.add(self._normalize_question(selected_question))
        session.used_hr_questions.append(selected_question)
        session.hr_category_counts[target_category] += 1

        if db_manager:
            try:
                await db_manager.store_hr_question_asked(student_id=session.student_id, question=selected_question, session_id=session.session_id)
            except Exception as e:
                logger.warning(f"[HR] Could not store question: {e}")

        logger.info(f"[HR] Selected [{target_category.upper()}] ({session.hr_category_counts[target_category]}/{CATEGORY_LIMITS[target_category]}): {selected_question[:60]}...")
        return selected_question, ["hr", target_category]

    async def _load_hr_questions_by_category(self, session, db_manager):
        """Load HR questions from MongoDB and organize by category."""
        try:
            from pymongo import MongoClient
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client["ml_notes"]
            collection = db["HR&Managerial_Interview_Questions"]
            logger.info("[HR] Loading questions from MongoDB by category...")
            doc = collection.find_one({"candidate_type": "fresher"})
            if not doc:
                doc = collection.find_one({})
            if not doc:
                logger.error("[HR] Collection is empty!")
                client.close()
                return
            session.hr_questions_by_category = {'introduction': [], 'behavioral': [], 'leadership': [], 'logical_thinking': []}
            for category in ['introduction', 'behavioral', 'leadership', 'logical_thinking']:
                if category in doc and isinstance(doc[category], dict):
                    category_data = doc[category]
                    if "questions" in category_data and isinstance(category_data["questions"], list):
                        questions = []
                        for q_obj in category_data["questions"]:
                            if isinstance(q_obj, dict) and "text" in q_obj:
                                q_text = str(q_obj["text"]).strip()
                                if len(q_text) > 10:
                                    questions.append(q_text)
                        session.hr_questions_by_category[category] = questions
                        logger.info(f"[HR] Loaded {len(questions)} questions from '{category}'")
            client.close()
            total = sum(len(qs) for qs in session.hr_questions_by_category.values())
            logger.info(f"[HR] ✅ Total questions loaded: {total}")
        except Exception as e:
            logger.error(f"[HR] Error loading questions by category: {e}")
            import traceback; traceback.print_exc()
            raise

    async def _generate_smart_followup(self, session, user_response: str, current_stage: WI_InterviewStage) -> str:
        await self.client_manager.initialize()
        prompt = f"""User said: "{user_response[:80]}"
Generate a short follow-up question. MAX 12 words."""
        resp = await self.client_manager.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=30)
        q = resp.choices[0].message.content.strip()
        return q if '?' in q else q + "?"

    async def _pace_hr_question(self, session):
        """Ensure at least 60 seconds between HR questions."""
        if session.last_hr_question_time > 0:
            elapsed = time.time() - session.last_hr_question_time
            if elapsed < 60:
                wait_time = 60 - elapsed
                logger.info(f"[WI] HR Pacing: waiting {wait_time:.1f}s before next question")
                await asyncio.sleep(wait_time)
        session.last_hr_question_time = time.time()

    # =========================================================================
    # MAIN RESPONSE GENERATION
    # =========================================================================

    async def generate_first_question(self, session) -> str:
        return await self.generate_introduction(session)

    async def generate_introduction(self, session):
        return f"""Hello {session.student_name}! Welcome to your weekly interview session. I'm excited to chat with you today!\n\nWe'll have three rounds:\n• First, a Communication round (about 5 minutes) where we'll have a casual conversation and get to know each other.\n• Then, a Technical round (about 25 minutes) where we'll discuss your recent work and technical knowledge.\n• Finally, an HR round (about 10 minutes) with some behavioral questions.\n\nSo, how are you doing today? Ready to get started?"""

    async def generate_silence_response(self, session) -> str:
        session.silence_prompt_count += 1
        return random.choice(["Take your time.", "I'm here when you're ready.", "Would you like me to repeat?", "No rush, think about it."])

    async def generate_fast_response(self, session, user_response: str, db_manager=None) -> str:
        await self.client_manager.initialize()
        quality = self._assess_answer_quality(user_response)
        logger.info(f"[WI] Quality: {quality}, Stage: {session.current_stage.value}")
        if quality != "silence":
            session.silence_prompt_count = 0
        session.conversation_state.last_user_response = user_response
        mentioned_tech = self._extract_topics_from_response(user_response, session)
        session.conversation_state.user_mentioned_tech.extend(mentioned_tech)

        # Handle REPEAT
        if quality == "repeat":
            if session.exchanges:
                if session.conversation_state.last_pure_question:
                    original_question = session.conversation_state.last_pure_question
                else:
                    last_ai_msg = session.exchanges[-1].ai_message
                    original_question = self._extract_question_from_response(last_ai_msg)
                repeat_response = f"{random.choice(REPEAT_RESPONSES)} {original_question}"
                session.last_was_repeat = True
                return repeat_response
            return "Let me start with a question!"
        session.last_was_repeat = False

        # Introduction -> Communication
        if session.current_stage == WI_InterviewStage.INTRODUCTION:
            session.introduction_completed = True
            session.start_round(WI_InterviewStage.COMMUNICATION)
            q = await self._generate_communication_question(session, True)
            return f"Great to hear! Let's get to know you. {q}"

        elapsed = session.get_round_elapsed_minutes()
        total_elapsed = session.get_total_interview_time_minutes()
        questions_in_round = session.get_questions_in_current_round()

        logger.info(f"[WI] ╔══ TIME CHECK: Stage={session.current_stage.value} Elapsed={elapsed:.2f}min Total={total_elapsed:.2f}min Qs={questions_in_round} ══╝")

        # --- TIME-BASED TRANSITIONS ---
        if session.current_stage == WI_InterviewStage.COMMUNICATION and elapsed >= 10:
            logger.info(f"[WI] ⏰ TRANSITIONING: Communication -> Technical")
            session.start_round(WI_InterviewStage.TECHNICAL)
            q, keywords = await self._generate_technical_question(session)
            session.add_exchange(q, expected_keywords=keywords, question_type="technical")
            return f"Nice chatting! Now let's discuss your technical work. {q}"
        elif session.current_stage == WI_InterviewStage.TECHNICAL and elapsed >= 25:
            logger.info(f"[WI] ⏰ TRANSITIONING: Technical -> HR")
            session.start_round(WI_InterviewStage.HR)
            session.last_hr_question_time = time.time()
            q, keywords = await self._generate_hr_question(session, db_manager)
            session.add_exchange(q, expected_keywords=keywords, question_type="hr")
            return f"Great technical discussion! Now some behavioral questions. {q}"
        elif session.current_stage == WI_InterviewStage.HR and elapsed >= 10:
            logger.info(f"[WI] ⏰ TRANSITIONING: HR -> Complete")
            session.current_stage = WI_InterviewStage.COMPLETE
            return "Thank you! Great interview. Let me generate your detailed feedback..."

        # === COMMUNICATION ROUND ===
        if session.current_stage == WI_InterviewStage.COMMUNICATION:
            if quality == "skip":
                q = await self._generate_communication_question(session)
                session.add_exchange(q, question_type="communication")
                ack = await self._generate_dynamic_ack("skip", "transition")
                return f"{ack} {q}"
            if quality == "silence":
                return await self.generate_silence_response(session)
            if quality == "gibberish":
                return "I'm sorry, I didn't catch that clearly. Could you please repeat your answer?"
            if quality == "cant_answer":
                q = await self._generate_communication_question(session)
                session.add_exchange(q, question_type="communication")
                ack = await self._generate_dynamic_ack("cant answer", "cant_answer")
                return f"{ack} {q}"
            if quality == "weak":
                q = await self._generate_communication_question(session)
                session.add_exchange(q, question_type="communication")
                ack = await self._generate_dynamic_ack("weak response", "weak")
                return f"{ack} {q}"
            if self._should_followup(session, quality):
                session.conversation_state.followups_on_topic += 1
                q = await self._generate_communication_followup(session, user_response)
                session.add_exchange(q, question_type="communication", is_followup=True)
                ack = await self._generate_dynamic_ack("good response", "good")
                return f"{ack} {q}"
            q = await self._generate_communication_question(session)
            session.add_exchange(q, question_type="communication")
            session.conversation_state.followups_on_topic = 0
            ack = await self._generate_dynamic_ack("transition", "transition")
            return f"{ack} {q}"

        # === TECHNICAL ROUND ===
        if session.current_stage == WI_InterviewStage.TECHNICAL:
            if session.exchanges and session.exchanges[-1].question_type == "technical":
                last_ex = session.exchanges[-1]
                accuracy = await self._evaluate_technical_accuracy(session, last_ex.ai_message, user_response, last_ex.expected_keywords)
                session.update_last_response(user_response, 0.8, quality, accuracy)
                logger.info(f"[WI] Technical accuracy: {accuracy:.2f}")
            self._adjust_difficulty(session, quality)
            if quality == "skip":
                q, keywords = await self._generate_technical_question(session, "", True)
                session.add_exchange(q, expected_keywords=keywords, question_type="technical")
                ack = await self._generate_dynamic_ack("skip", "transition")
                return f"{ack} {q}"
            if quality == "gibberish":
                return "I'm sorry, I didn't catch that clearly. Could you please repeat your answer?"
            if quality == "silence":
                if session.exchanges:
                    last_q = session.exchanges[-1].ai_message.lower()
                    for tech in session.extracted_technologies:
                        if tech.lower() in last_q:
                            session.topic_attempt_count[tech] = session.topic_attempt_count.get(tech, 0) + 1
                            if session.topic_attempt_count[tech] >= 2 and tech not in session.silent_topics:
                                session.silent_topics.append(tech)
                                logger.info(f"[WI] Marking topic '{tech}' as silent - will skip in future")
                            break
                session.silence_prompt_count += 1
                if session.silence_prompt_count >= 2:
                    session.silence_prompt_count = 0
                    q, keywords = await self._generate_technical_question(session, "", True)
                    session.add_exchange(q, expected_keywords=keywords, question_type="technical")
                    return f"Let's try something different. {q}"
                return await self.generate_silence_response(session)
            if quality == "cant_answer":
                if session.exchanges:
                    last_q = session.exchanges[-1].ai_message.lower()
                    for tech in session.extracted_technologies:
                        if tech.lower() in last_q:
                            session.topic_attempt_count[tech] = session.topic_attempt_count.get(tech, 0) + 1
                            if session.topic_attempt_count[tech] >= 2 and tech not in session.silent_topics:
                                session.silent_topics.append(tech)
                            break
                session.current_difficulty = "easy"
                q, keywords = await self._generate_technical_question(session, "", True)
                session.add_exchange(q, expected_keywords=keywords, question_type="technical")
                ack = await self._generate_dynamic_ack("cant answer technical", "cant_answer")
                return f"{ack} {q}"
            if quality == "weak":
                session.current_difficulty = "easy"
                q, keywords = await self._generate_technical_question(session, "", True)
                session.add_exchange(q, expected_keywords=keywords, question_type="technical")
                ack = await self._generate_dynamic_ack("weak technical", "technical_weak")
                return f"{ack} {q}"
            if quality == "strong" and random.random() < 0.3:
                q = await self._generate_smart_followup(session, user_response, WI_InterviewStage.TECHNICAL)
                session.add_exchange(q, question_type="technical", is_followup=True)
                ack = await self._generate_dynamic_ack("good technical", "technical_good")
                return f"{ack} {q}"
            q, keywords = await self._generate_technical_question(session, user_response, True)
            session.add_exchange(q, expected_keywords=keywords, question_type="technical")
            ack = await self._generate_dynamic_ack("technical", "technical_good" if quality == "strong" else "transition")
            return f"{ack} {q}"

        # === HR ROUND ===
        if session.current_stage == WI_InterviewStage.HR:
            if session.exchanges and session.exchanges[-1].question_type == "hr":
                last_ex = session.exchanges[-1]
                accuracy = await self._evaluate_technical_accuracy(session, last_ex.ai_message, user_response, last_ex.expected_keywords)
                session.update_last_response(user_response, 0.8, quality, accuracy)
            if quality == "skip":
                await self._pace_hr_question(session)
                q, keywords = await self._generate_hr_question(session, db_manager)
                session.add_exchange(q, expected_keywords=keywords, question_type="hr")
                ack = await self._generate_dynamic_ack("skip", "transition")
                return f"{ack} {q}"
            if quality == "gibberish":
                return "I'm sorry, I didn't catch that clearly. Could you please repeat your answer?"
            if quality == "silence":
                session.silence_prompt_count += 1
                if session.silence_prompt_count >= 2:
                    session.silence_prompt_count = 0
                    await self._pace_hr_question(session)
                    q, keywords = await self._generate_hr_question(session, db_manager)
                    session.add_exchange(q, expected_keywords=keywords, question_type="hr")
                    return f"Let's try a different question. {q}"
                return await self.generate_silence_response(session)
            if quality == "cant_answer":
                await self._pace_hr_question(session)
                q, keywords = await self._generate_hr_question(session, db_manager)
                session.add_exchange(q, expected_keywords=keywords, question_type="hr")
                ack = await self._generate_dynamic_ack("cant answer hr", "cant_answer")
                return f"{ack} {q}"
            if quality == "weak":
                await self._pace_hr_question(session)
                q, keywords = await self._generate_hr_question(session, db_manager)
                session.add_exchange(q, expected_keywords=keywords, question_type="hr")
                ack = await self._generate_dynamic_ack("weak hr", "weak")
                return f"{ack} {q}"
            if quality == "strong" and random.random() < 0.25:
                await self._pace_hr_question(session)
                q = await self._generate_smart_followup(session, user_response, WI_InterviewStage.HR)
                session.add_exchange(q, question_type="hr", is_followup=True)
                ack = await self._generate_dynamic_ack("good hr", "hr")
                return f"{ack} {q}"
            await self._pace_hr_question(session)
            q, keywords = await self._generate_hr_question(session, db_manager)
            if keywords and "hr_complete" in keywords:
                logger.info(f"[WI] ⏰ HR categories complete - transitioning to COMPLETE stage")
                session.current_stage = WI_InterviewStage.COMPLETE
                return q
            session.add_exchange(q, expected_keywords=keywords, question_type="hr")
            ack = await self._generate_dynamic_ack("hr response", "hr")
            return f"{ack} {q}"

        return "That's interesting. Tell me more?"

    # =========================================================================
    # EVALUATION - With Q&A Feedback Format
    # =========================================================================

    async def generate_fast_evaluation(self, session) -> Tuple[str, Dict[str, float]]:
        """Generate comprehensive evaluation with Q&A feedback format per round"""
        await self.client_manager.initialize()

        comm_exchanges = []; tech_exchanges = []; hr_exchanges = []
        tech_accuracies = []; hr_accuracies = []

        for ex in session.exchanges:
            exchange_data = {
                "question": ex.ai_message,
                "answer": ex.user_response if ex.user_response else "[SILENT - No response]",
                "is_silent": not ex.user_response or ex.answer_quality == "silence",
                "answer_quality": ex.answer_quality, "accuracy": ex.technical_accuracy
            }
            if ex.stage == WI_InterviewStage.COMMUNICATION:
                comm_exchanges.append(exchange_data)
            elif ex.stage == WI_InterviewStage.TECHNICAL:
                tech_exchanges.append(exchange_data)
                if ex.technical_accuracy is not None:
                    tech_accuracies.append(ex.technical_accuracy)
            elif ex.stage == WI_InterviewStage.HR:
                hr_exchanges.append(exchange_data)
                if ex.technical_accuracy is not None:
                    hr_accuracies.append(ex.technical_accuracy)

        tech_accuracy_avg = sum(tech_accuracies) / len(tech_accuracies) if tech_accuracies else 0.5
        hr_accuracy_avg = sum(hr_accuracies) / len(hr_accuracies) if hr_accuracies else 0.5
        total_technical_qs = len(tech_exchanges)
        total_hr_qs = len(hr_exchanges)
        total_comm_qs = len(comm_exchanges)

        async def get_feedback_for_qa(question: str, answer: str, round_type: str, is_silent: bool) -> str:
            if is_silent:
                return "Candidate remained silent. Try to respond even with partial thoughts."
            prompt = f"""Give brief feedback (1-2 sentences) for this {round_type} interview answer.
Question: {question}
Answer: {answer}
Be constructive. If good, praise briefly. If weak, suggest improvement."""
            try:
                resp = await self.client_manager.openai_client.chat.completions.create(
                    model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=100)
                return resp.choices[0].message.content.strip()
            except:
                return "Response recorded."

        evaluation_parts = []

        if comm_exchanges:
            evaluation_parts.append("=" * 60)
            evaluation_parts.append("COMMUNICATION ROUND FEEDBACK")
            evaluation_parts.append("=" * 60)
            for i, ex in enumerate(comm_exchanges, 1):
                feedback = await get_feedback_for_qa(ex["question"], ex["answer"], "communication", ex["is_silent"])
                evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}")
                evaluation_parts.append(f"    User Answer: {ex['answer']}")
                evaluation_parts.append(f"    Feedback: {feedback}")
                evaluation_parts.append("-" * 40)

        if tech_exchanges:
            evaluation_parts.append("\n" + "=" * 60)
            evaluation_parts.append("TECHNICAL ROUND FEEDBACK")
            evaluation_parts.append("=" * 60)
            for i, ex in enumerate(tech_exchanges, 1):
                feedback = await get_feedback_for_qa(ex["question"], ex["answer"], "technical", ex["is_silent"])
                accuracy_str = f" (Accuracy: {ex['accuracy']:.0%})" if ex["accuracy"] is not None else ""
                evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}")
                evaluation_parts.append(f"    User Answer: {ex['answer']}")
                evaluation_parts.append(f"    Feedback: {feedback}{accuracy_str}")
                evaluation_parts.append("-" * 40)

        if hr_exchanges:
            evaluation_parts.append("\n" + "=" * 60)
            evaluation_parts.append("HR/BEHAVIORAL ROUND FEEDBACK")
            evaluation_parts.append("=" * 60)
            for i, ex in enumerate(hr_exchanges, 1):
                feedback = await get_feedback_for_qa(ex["question"], ex["answer"], "HR/behavioral", ex["is_silent"])
                evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}")
                evaluation_parts.append(f"    User Answer: {ex['answer']}")
                evaluation_parts.append(f"    Feedback: {feedback}")
                evaluation_parts.append("-" * 40)

        evaluation_parts.append("\n" + "=" * 60)
        evaluation_parts.append("OVERALL SUMMARY")
        evaluation_parts.append("=" * 60)

        silent_count = sum(1 for ex in comm_exchanges + tech_exchanges + hr_exchanges if ex["is_silent"])
        total_tech_generated = getattr(session, 'total_technical_questions_generated', total_technical_qs)

        summary_prompt = f"""Provide a brief overall interview summary (4-5 sentences) for {session.student_name}.
METRICS:
- Communication Questions: {total_comm_qs}
- Technical Questions Generated: {total_tech_generated}
- Technical Questions Answered: {total_technical_qs}
- Technical Accuracy: {tech_accuracy_avg:.0%}
- HR Questions: {total_hr_qs}
- Correct Answers: {session.correct_answers}
- Partial Answers: {session.partial_answers}
- Weak Answers: {session.wrong_answers}
- Silent/No Response: {silent_count}
Include: 1. Overall performance summary 2. Key strengths 3. Areas to improve 4. Final recommendation"""

        summary_resp = await self.client_manager.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=[{"role": "user", "content": summary_prompt}], temperature=0.3, max_tokens=400)
        overall_summary = summary_resp.choices[0].message.content.strip()
        evaluation_parts.append(f"\n{overall_summary}")

        evaluation_parts.append("\n" + "-" * 40)
        evaluation_parts.append("STATISTICS:")
        evaluation_parts.append(f"  • Total Technical Questions Generated: {total_tech_generated}")
        evaluation_parts.append(f"  • Technical Accuracy: {tech_accuracy_avg:.0%}")
        evaluation_parts.append(f"  • Questions Answered Well: {session.correct_answers}/{total_technical_qs + total_hr_qs}")
        evaluation_parts.append(f"  • Partial Answers: {session.partial_answers}")
        evaluation_parts.append(f"  • Needs Improvement: {session.wrong_answers}")
        evaluation_parts.append(f"  • Silent Responses: {silent_count}")

        evaluation = "\n".join(evaluation_parts)

        # Generate numerical scores
        score_prompt = f"""Based on this interview, provide scores (0-10) for each criteria.
METRICS:
- Technical Accuracy: {tech_accuracy_avg:.0%}
- Correct Answers: {session.correct_answers}/{total_technical_qs}
- Communication Questions: {total_comm_qs}
- HR Questions: {total_hr_qs}
- Silent Responses: {silent_count}
SCORING CRITERIA:
1. Communication (20%): Clarity, engagement, listening skills
2. Technical (30%): Accuracy, depth, problem-solving - USE THE {tech_accuracy_avg:.0%} ACCURACY
3. Leadership (15%): Initiative, decision-making, examples
4. Behaviour (20%): Professionalism, attitude, self-awareness
5. Confidence (15%): Composure, conviction, handling pressure
IMPORTANT: Technical score should reflect the {tech_accuracy_avg:.0%} accuracy rate. Deduct points for silent responses.
Reply in EXACT format:
communication: X
technical: X
leadership: X
behaviour: X
confidence: X"""

        sc_resp = await self.client_manager.openai_client.chat.completions.create(
            model=config.OPENAI_MODEL, messages=[{"role": "user", "content": score_prompt}], temperature=0.1, max_tokens=200)
        score_text = sc_resp.choices[0].message.content.lower()

        scores = {}
        for key in ["communication", "technical", "leadership", "behaviour", "confidence"]:
            m = re.search(rf"{key}[:\s]*(\d+\.?\d*)", score_text)
            if m:
                scores[f"{key}_score"] = min(float(m.group(1)), 10.0)
            else:
                scores[f"{key}_score"] = round(tech_accuracy_avg * 10, 1) if key == "technical" else 5.0

        scores["technical_accuracy"] = round(tech_accuracy_avg * 100, 1)
        scores["hr_accuracy"] = round(hr_accuracy_avg * 100, 1)
        scores["questions_correct"] = session.correct_answers
        scores["questions_partial"] = session.partial_answers
        scores["questions_wrong"] = session.wrong_answers
        scores["questions_silent"] = silent_count
        scores["total_questions"] = total_technical_qs + total_hr_qs + total_comm_qs

        w = getattr(config, 'EVALUATION_CRITERIA', {"communication_weight": 0.20, "technical_weight": 0.30, "leadership_weight": 0.15, "behaviour_weight": 0.20, "confidence_weight": 0.15})
        scores["weighted_overall"] = round(
            scores.get("communication_score", 5) * w.get("communication_weight", 0.2) +
            scores.get("technical_score", 5) * w.get("technical_weight", 0.3) +
            scores.get("leadership_score", 5) * w.get("leadership_weight", 0.15) +
            scores.get("behaviour_score", 5) * w.get("behaviour_weight", 0.2) +
            scores.get("confidence_score", 5) * w.get("confidence_weight", 0.15), 1
        )

        logger.info(f"[WI] Evaluation complete - Overall: {scores['weighted_overall']}/10, Tech Accuracy: {scores['technical_accuracy']}%, Silent: {silent_count}")

        # Build evaluation_details
        evaluation_details = {"rounds": {"communication": [], "technical": [], "hr": []}, "overall_summary": overall_summary, "recommendations": []}
        for ex_data, round_key in [(comm_exchanges, "communication"), (tech_exchanges, "technical"), (hr_exchanges, "hr")]:
            for item in ex_data:
                evaluation_details["rounds"][round_key].append({
                    "question": item["question"], "answer": item["answer"],
                    "feedback": item.get("feedback", ""), "accuracy": item.get("accuracy"),
                    "is_silent": item.get("is_silent", False),
                })

        # Parse feedback from evaluation text into evaluation_details
        current_round = None
        qa_idx = {"communication": 0, "technical": 0, "hr": 0}
        for line in evaluation.split("\n"):
            line_stripped = line.strip()
            if "COMMUNICATION ROUND" in line_stripped.upper():
                current_round = "communication"
            elif "TECHNICAL ROUND" in line_stripped.upper():
                current_round = "technical"
            elif "HR" in line_stripped.upper() and "ROUND" in line_stripped.upper():
                current_round = "hr"
            elif "OVERALL SUMMARY" in line_stripped.upper():
                current_round = None
            elif current_round and line_stripped.startswith("Feedback:"):
                fb = line_stripped.split("Feedback:", 1)[1].strip()
                fb = re.sub(r'\s*\(Accuracy:\s*[\d.]+%\)', '', fb).strip()
                idx = qa_idx[current_round]
                if idx < len(evaluation_details["rounds"][current_round]):
                    evaluation_details["rounds"][current_round][idx]["feedback"] = fb
                    qa_idx[current_round] = idx + 1

        recommendations = []
        if scores.get("technical_score", 5) < 6:
            recommendations.append("Focus on deepening technical knowledge in core areas discussed during the interview.")
        if scores.get("communication_score", 5) < 6:
            recommendations.append("Practice articulating thoughts more clearly and completely during conversations.")
        if scores.get("confidence_score", 5) < 6:
            recommendations.append("Build confidence by practicing mock interviews and presenting technical topics.")
        if scores.get("leadership_score", 5) < 6:
            recommendations.append("Prepare specific examples of leadership, initiative, and decision-making from past experiences.")
        if silent_count > total_technical_qs * 0.3:
            recommendations.append("Reduce silent responses — even partial answers demonstrate engagement and willingness to try.")
        if scores.get("technical_accuracy", 0) < 50:
            recommendations.append("Review core technical concepts and practice explaining them in your own words.")
        if not recommendations:
            recommendations.append("Continue building on your strong foundation with advanced topics and real-world practice.")
        evaluation_details["recommendations"] = recommendations
        scores["evaluation_details"] = evaluation_details

        return evaluation, scores


# =============================================================================
# =============================================================================
#  SECTION 3:  WEEKEND MOCK TEST  (kept original names)
# =============================================================================
# =============================================================================

class AIService:
    """Production AI service for question generation and evaluation (weekend_mocktest)"""
    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        self.client = Groq(api_key=config.GROQ_API_KEY, timeout=getattr(config, "GROQ_TIMEOUT", 60))
        self._test_connection()
        logger.info("[MT] AI Service initialized")

    def _test_connection(self):
        try:
            response = self.client.chat.completions.create(
                model=getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": "Hello"}],
                max_completion_tokens=10
            )
            if not response.choices:
                raise Exception("No response from AI service")
        except Exception as e:
            raise Exception(f"AI service connection failed: {e}")

    def _call_llm_with_retries(self, prompt: str, max_tokens: int, temperature: float = None) -> str:
        if temperature is None:
            temperature = getattr(config, "GROQ_TEMPERATURE", 0.7)
        max_retries = getattr(config, "MAX_RETRIES", 3)
        delay = getattr(config, "RETRY_DELAY", 2)
        last_error = None
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=getattr(config, "GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens
                )
                if not completion.choices:
                    raise Exception("No response from LLM")
                response = completion.choices[0].message.content.strip()
                if len(response) < 100:
                    raise Exception("Response too short")
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"[MT] LLM attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
        raise Exception(f"LLM failed after {max_retries} attempts: {last_error}")

    def _parse_single_question(self, section: str, user_type: str, qn: int) -> Dict[str, Any]:
        lines = [ln.strip() for ln in section.split('\n') if ln.strip()]
        data = {"question_number": qn, "title": f"Question {qn}", "difficulty": "Medium", "type": "General", "question": "", "options": None}
        current = None
        q_lines, options = [], []
        for ln in lines:
            if ln.startswith("## Title:"):
                data["title"] = ln.replace("## Title:", "").strip()
            elif ln.startswith("## Difficulty:"):
                data["difficulty"] = ln.replace("## Difficulty:", "").strip()
            elif ln.startswith("## Type:"):
                data["type"] = ln.replace("## Type:", "").strip()
            elif ln.startswith("## Question:"):
                current = "q"
            elif ln.startswith("## Options:") and user_type == "non_dev":
                current = "o"
            elif current == "q":
                if not ln.startswith("##"):
                    q_lines.append(ln)
            elif current == "o" and user_type == "non_dev":
                if re.match(r'^[A-D]\)', ln):
                    option_text = ln[3:].strip()
                    if option_text:
                        options.append(option_text)
        data["question"] = "\n".join(q_lines).strip()
        if user_type == "non_dev":
            data["options"] = options if len(options) == 4 else None
        if not data["question"] or len(data["question"]) < 50:
            raise Exception("Question too short")
        if user_type == "non_dev" and not data["options"]:
            raise Exception("MCQ missing options")
        return data

    def _parse_questions_response(self, response: str, user_type: str) -> List[Dict[str, Any]]:
        questions = []
        sections = re.split(r'=== QUESTION \d+ ===', response)[1:]
        for i, sec in enumerate(sections, 1):
            try:
                q = self._parse_single_question(sec, user_type, i)
                if q:
                    questions.append(q)
            except Exception as e:
                logger.warning(f"[MT] Failed to parse question {i}: {e}")
        return questions

    def _extract_scores_fallback(self, response: str, n: int) -> List[int]:
        pats = re.findall(r'(?:^|\s)([01](?:\s*,\s*[01])+)(?:\s|$)', response)
        for p in pats:
            arr = [int(s.strip()) for s in p.split(',')]
            if len(arr) == n:
                return arr
        logger.warning("[MT] Using fallback scoring")
        return [1 if i % 2 == 0 else 0 for i in range(n)]

    def _extract_feedbacks_fallback(self, response: str, n: int) -> List[str]:
        lines = response.split('\n')
        fbs = []
        for ln in lines:
            if 'question' in ln.lower() and any(w in ln.lower() for w in ['correct', 'incorrect', 'good', 'poor']):
                fbs.append(ln.strip())
                if len(fbs) == n:
                    break
        while len(fbs) < n:
            fbs.append(f"Question {len(fbs)+1}: Evaluated")
        return fbs[:n]

    def _parse_evaluation_response(self, response: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores, feedbacks = [], []
        m_scores = re.search(r'SCORES:\s*\[(.*?)\]', response, re.DOTALL)
        if m_scores:
            score_str = m_scores.group(1)
            scores = [int(s.strip()) for s in score_str.split(',') if s.strip().isdigit()]
        m_fb = re.search(r'FEEDBACK:\s*\[(.*?)\]', response, re.DOTALL)
        if m_fb:
            fb_str = m_fb.group(1)
            feedbacks = [f.strip().strip('"\'') for f in fb_str.split('|')]
        if not scores or len(scores) != len(qa_pairs):
            scores = self._extract_scores_fallback(response, len(qa_pairs))
        if not feedbacks or len(feedbacks) != len(qa_pairs):
            feedbacks = self._extract_feedbacks_fallback(response, len(qa_pairs))
        if len(scores) != len(qa_pairs):
            raise Exception(f"Score count mismatch: {len(scores)} vs {len(qa_pairs)}")
        if len(feedbacks) != len(qa_pairs):
            feedbacks = [f"Question {i+1}: {'Correct' if scores[i] else 'Incorrect'}" for i in range(len(qa_pairs))]
        return {"scores": scores, "feedbacks": feedbacks, "total_correct": sum(scores), "evaluation_report": response}

    def generate_questions_batch(self, user_type: str, context: str) -> List[Dict[str, Any]]:
        logger.info(f"[MT] Generating {getattr(config, 'QUESTIONS_PER_TEST', 10)} {user_type} questions")
        prompt = PromptTemplates.create_batch_questions_prompt(user_type, context, getattr(config, "QUESTIONS_PER_TEST", 10))
        response = self._call_llm_with_retries(prompt, getattr(config, "GROQ_MAX_TOKENS", 3000))
        questions = self._parse_questions_response(response, user_type)
        if not questions:
            raise Exception("No valid questions generated")
        return questions

    def evaluate_test_batch(self, user_type: str, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"[MT] Evaluating {len(qa_pairs)} {user_type} answers")
        prompt = PromptTemplates.create_evaluation_prompt(user_type, qa_pairs)
        response = self._call_llm_with_retries(prompt, getattr(config, "EVALUATION_MAX_TOKENS", 2000), getattr(config, "EVALUATION_TEMPERATURE", 0.3))
        return self._parse_evaluation_response(response, qa_pairs)


# Singleton as in weekend_mocktest
_ai_service_singleton: Optional[AIService] = None

def get_ai_service() -> AIService:
    global _ai_service_singleton
    if _ai_service_singleton is None:
        _ai_service_singleton = AIService()
    return _ai_service_singleton