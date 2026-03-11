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
import httpx
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

from .conversation_patterns import (
    ConversationTracker, get_response_for_quality, get_encouragement,
    detect_checkin_response
)

logger = logging.getLogger(__name__)

ROUND_DURATIONS = {
    "introduction": 60,
    "communication": 300,
    "technical": 1500,
    "hr": 600,
}

# If user gives no real response for 5 consecutive questions, auto-skip to next round
MAX_CONSECUTIVE_SILENCE = 5

TECHNICAL_QUESTION_TEMPLATES = [
    "Can you explain what {tech} is and how you've used it in your work?",
    "What are the key components or features of {tech} that you worked with?",
    "How does {tech} fit into the overall architecture of your projects?",
    "Walk me through the basic workflow when working with {tech}.",
    "What's the purpose of {tech} and why is it important in your domain?",
    "Describe a specific project where you implemented {tech}.",
    "What was your day-to-day work with {tech} like?",
    "How did you configure or set up {tech} in your environment?",
    "What tools, commands, or transactions did you use when working with {tech}?",
    "Can you give me an example of how you used {tech} to solve a real business problem?",
    "What was the most challenging issue you faced with {tech} and how did you resolve it?",
    "Describe a bug or error you encountered in {tech} and your debugging approach.",
    "How do you troubleshoot problems when {tech} isn't working correctly?",
    "Tell me about a time when {tech} failed unexpectedly. How did you handle it?",
    "What's the most complex problem you solved using {tech}?",
    "What best practices do you follow when working with {tech}?",
    "How do you ensure quality and avoid errors when implementing {tech}?",
    "What documentation or standards do you follow for {tech}?",
    "How do you test your work with {tech} before deploying to production?",
    "What common mistakes should be avoided when working with {tech}?",
    "How does {tech} integrate with other systems or components you've worked with?",
    "What performance considerations do you keep in mind when using {tech}?",
    "How do you handle security aspects when working with {tech}?",
    "What improvements or optimizations have you made to {tech} processes?",
    "How do you train or guide others on using {tech}?",
]

TECHNICAL_BEHAVIORAL_QUESTIONS = [
    "Tell me about the most difficult bug you encountered while working with {tech}. How did you debug it?",
    "Describe a situation where {tech} was not performing as expected. What steps did you take to identify the root cause?",
    "Walk me through a time when you had to troubleshoot a critical {tech} issue under pressure. What was your approach?",
    "Tell me about a {tech} problem that took you a long time to solve. What made it so challenging?",
    "Describe a scenario where you had to fix someone else's {tech} code or configuration. How did you approach it?",
    "Tell me about a technical decision you made regarding {tech} that you later had to reconsider. What did you learn?",
    "Describe a time when you had to choose between two different approaches in {tech}. How did you decide?",
    "Walk me through a situation where you disagreed with a colleague about how to implement something in {tech}. How was it resolved?",
    "Tell me about a time when you had to balance performance vs. maintainability in your {tech} work. What trade-offs did you make?",
    "Describe a {tech} implementation where you had to work within significant constraints. How did you handle it?",
    "Tell me about a time when you had to learn a new feature or version of {tech} quickly. How did you approach it?",
    "Describe a situation where you made a mistake with {tech} in production. How did you handle it and what did you learn?",
    "Walk me through how you stay updated with changes and best practices in {tech}.",
    "Tell me about a {tech} concept that was initially difficult for you to understand. How did you master it?",
    "Describe a time when you had to adapt your {tech} approach based on feedback or changing requirements. What changed?",
]

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

COMMUNICATION_TRANSITIONS = [
    "That's interesting! ", "Nice! ", "Great to know! ", "Thanks for sharing! ",
    "That sounds wonderful! ", "How lovely! ", "That's cool! ", "Awesome! ",
    "That's really nice! ", "Wonderful! ", "Oh, that's great! ", "I like that! ",
    "Sounds fun! ", "That's fantastic! ", "How interesting! ", "Good to know! ",
]
FOLLOWUP_ACKS = ["Oh interesting!", "That's nice!", "I see!", "That sounds great!", "Nice!", "Lovely!", "Oh really?", "That's cool!", "Wow!", "Fascinating!"]
TECHNICAL_GOOD_ACKS = ["Good explanation!", "That's correct!", "Nice approach!", "Well explained!", "Good point!", "Exactly right!", "Great understanding!", "Well done!", "Perfect!", "Excellent!"]
TECHNICAL_NEUTRAL_ACKS = ["I see.", "Okay.", "Alright.", "Got it.", "Understood.", "Fair enough."]
DONT_KNOW_RESPONSES = ["That's okay! Let me ask you something different.", "No problem at all! Here's another question.", "It's fine! Let's try a different one.", "No worries! Let me change the topic.", "That's alright! Moving to something else."]
WEAK_RESPONSE_ACKS = ["I see. Let me ask you something else.", "Okay, let's try a different question.", "Alright, let me move to another topic.", "Got it. Here's a different one.", "Understood. Let me ask something else."]
SKIP_RESPONSES = ["Sure! Let's move on.", "No problem, next one.", "Of course! Here's another.", "Got it, moving forward."]
REPEAT_RESPONSES = ["Of course! The question was:", "Sure, let me repeat:", "No problem! Here it is again:"]
HR_ACKS = ["Thank you for sharing.", "That's a good point.", "I appreciate that.", "Interesting.", "Good to know."]

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
    # ✅ FIX: Per-session lock prevents audio + silence from colliding
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

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
    """Daily-standup: separate pools for STT and LLM to prevent mutual blocking"""
    def __init__(self):
        self._groq_client = None
        self._openai_client = None
        # ✅ Two separate pools so STT never blocks LLM and vice versa
        self._stt_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="stt"
        )
        self._llm_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10, thread_name_prefix="llm"
        )

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
    def stt_executor(self):
        """Pool dedicated to speech-to-text (Groq) calls"""
        return self._stt_executor

    @property
    def llm_executor(self):
        """Pool dedicated to LLM (OpenAI) calls"""
        return self._llm_executor

    @property
    def executor(self):
        """Backward-compatible alias — defaults to LLM pool"""
        return self._llm_executor

    async def close_connections(self):
        if self._stt_executor:
            self._stt_executor.shutdown(wait=False)
        if self._llm_executor:
            self._llm_executor.shutdown(wait=False)
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
            # ✅ FIX: Use dedicated STT pool — never blocked by LLM calls
            return await loop.run_in_executor(
                self.client_manager.stt_executor, self._sync_transcribe, audio_data
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
# WEEKLY INTERVIEW (WI_*) - DATACLASSES
# =============================================================================

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
    last_pure_question: str = ""
    last_user_response: str = ""
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
    previously_asked_hr_questions: List[str]=field(default_factory=list)
    technical_question_count: int = 0
    behavioral_question_count: int = 0
    current_tech_index: int = 0
    current_hr_index: int = 0
    current_topic_index: int = 0
    tech_question_types_used: Dict[str, List[str]] = field(default_factory=dict)
    extracted_technologies: List[str] = field(default_factory=list)
    extracted_topics_for_questions: List[str] = field(default_factory=list)
    extracted_projects: List[str] = field(default_factory=list)
    extracted_challenges: List[str] = field(default_factory=list)
    extracted_team_info: List[str] = field(default_factory=list)
    technical_answers: List[Dict[str, Any]] = field(default_factory=list)
    correct_answers: int = 0
    partial_answers: int = 0
    wrong_answers: int = 0
    is_finalized: bool = False
    _last_real_question: str = ""  # Survives round transitions (unlike conversation_state.last_pure_question)
    consecutive_no_response: int = 0  # Tracks consecutive questions with no real answer — auto-skip after MAX
    # ADD this field to the WI_InterviewSession dataclass (after line ~350):
    conversation_tracker: ConversationTracker = field(default_factory=ConversationTracker)
    
    def __post_init__(self):
        self.interview_start_time = self.created_at
        logger.info(f"[WI] Session initialized. Interview start time: {self.interview_start_time}")

    def start_round(self, stage):
        current_time = time.time()
        logger.info(f"[WI] ===== STARTING ROUND: {stage.value} =====")
        self.round_start_times[stage.value] = current_time
        self.current_stage = stage
        self.conversation_state = WI_ConversationState()
        self.silence_prompt_count = 0  # Reset so new round gets fresh silence prompts
        self.consecutive_no_response = 0  # Reset silence streak for new round

    def get_round_elapsed_time(self):
        current_stage_value = self.current_stage.value
        current_time = time.time()
        if current_stage_value not in self.round_start_times:
            self.round_start_times[current_stage_value] = current_time
            return 0.0
        return current_time - self.round_start_times[current_stage_value]

    def get_round_elapsed_minutes(self): return self.get_round_elapsed_time() / 60
    
    def get_total_interview_time_minutes(self):
        if not hasattr(self, 'interview_start_time') or self.interview_start_time is None:
            self.interview_start_time = self.created_at
        return (time.time() - self.interview_start_time) / 60
    
    def get_questions_in_current_round(self): return self.questions_per_round.get(self.current_stage.value, 0)

    def add_exchange(self, ai_message, user_response="", quality=0.0, concept="", is_followup=False, answer_quality="neutral", expected_keywords=None, technical_accuracy=None, question_type="general"):
        ex = WI_ConversationExchange(timestamp=time.time(), stage=self.current_stage, ai_message=ai_message, user_response=user_response, transcript_quality=quality, concept=concept, is_followup=is_followup, answer_quality=answer_quality, expected_keywords=expected_keywords or [], technical_accuracy=technical_accuracy, question_type=question_type)
        self.exchanges.append(ex)
        self.questions_per_round[self.current_stage.value] = self.questions_per_round.get(self.current_stage.value, 0) + 1
        self.questions_asked.append(ai_message)
        # Don't overwrite _last_real_question with non-question responses (gibberish/silence prompts)
        non_question_phrases = [
            "i'm sorry, i didn't catch", "could you please repeat your answer",
            "take your time", "i'm here when you're ready", "no rush",
            "whenever you're ready", "no hurry", "don't worry",
            "i'm listening", "think it through", "no pressure",
            "are you ready", "can i continue", "still thinking",
            "repeat your answer", "didn't catch that",
            "feel free to take", "you can answer whenever", "completely fine",
            "let me try a different question", "let's move on to something else",
            "i'll ask you something different",
            "that concludes", "concludes our hr round", "you did great",
            "great interview", "generate your detailed feedback",
        ]
        is_non_question = any(phrase in ai_message.lower() for phrase in non_question_phrases)
        if '?' in ai_message and not is_non_question:
            parts = ai_message.split('?')
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i].strip()
                if len(part) > 10:
                    for sep in ['. ', '! ', '\n']:
                        if sep in part: part = part.split(sep)[-1].strip()
                    self.conversation_state.last_pure_question = part + '?'
                    self._last_real_question = part + '?'
                    break
        elif not is_non_question:
            self.conversation_state.last_pure_question = ai_message
            self._last_real_question = ai_message

    def update_last_response(self, user_response, quality, answer_quality="neutral", technical_accuracy=None):
        if self.exchanges:
            self.exchanges[-1].user_response = user_response
            self.exchanges[-1].answer_quality = answer_quality
            self.exchanges[-1].technical_accuracy = technical_accuracy
            if technical_accuracy is not None:
                if technical_accuracy >= 0.7: self.correct_answers += 1
                elif technical_accuracy >= 0.4: self.partial_answers += 1
                else: self.wrong_answers += 1
        self.last_answer_quality = answer_quality

    def get_stage_conversation_history(self, stage, limit=10):
        exs = [e for e in self.exchanges if e.stage == stage][-limit:]
        return "\n".join([f"Q: {e.ai_message}\nA: {e.user_response}" for e in exs if e.user_response])

    def get_questions_asked_in_round(self, stage):
        return [e.ai_message for e in self.exchanges if e.stage == stage]

    def get_last_user_response(self):
        for ex in reversed(self.exchanges):
            if ex.user_response: return ex.user_response
        return ""
    
    def get_conversation_by_round(self):
        result = {"communication": [], "technical": [], "hr": []}
        for ex in self.exchanges:
            exchange_data = {"question": ex.ai_message, "answer": ex.user_response or "[NO RESPONSE]", "timestamp": ex.timestamp, "answer_quality": ex.answer_quality, "is_followup": ex.is_followup, "technical_accuracy": ex.technical_accuracy}
            if ex.stage == WI_InterviewStage.COMMUNICATION: result["communication"].append(exchange_data)
            elif ex.stage == WI_InterviewStage.TECHNICAL: result["technical"].append(exchange_data)
            elif ex.stage == WI_InterviewStage.HR: result["hr"].append(exchange_data)
        return result

# =============================================================================
# WI CLIENT MANAGER & FRAGMENT MANAGER
# =============================================================================

class WI_SharedClientManager:
    def __init__(self):
        self.openai_client: Optional[AsyncOpenAI] = None
        self.groq_client: Optional[AsyncGroq] = None
        self.executor = ThreadPoolExecutor(max_workers=16)  # Support 16 concurrent audio processing tasks
        self._initialized = False
    async def initialize(self):
        if self._initialized: return
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.groq_client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=httpx.Timeout(30.0, connect=10.0),
            max_retries=0,  # We handle retries ourselves with fallback
        )
        self._initialized = True
    async def close_connections(self):
        if self.openai_client: await self.openai_client.close()
        if self.groq_client: await self.groq_client.close()
        self.executor.shutdown(wait=True)

wi_shared_clients = WI_SharedClientManager()

class WI_EnhancedInterviewFragmentManager:
    def __init__(self, client_manager, session):
        self.client_manager = client_manager
        self.session = session
    def initialize_fragments(self, summaries):
        if not summaries: return False
        self.session.content_context = "\n".join([s.get("summary", "") for s in summaries])
        self._extract_summary_info(self.session.content_context)
        self.session.start_round(WI_InterviewStage.INTRODUCTION)
        return True
    def _extract_summary_info(self, content):
        content_lower = content.lower()
        sap_keywords = ["sap", "abap", "fiori", "hana", "s/4hana", "s4hana", "mm", "sd", "fico", "pp", "wm", "ewm", "ariba", "successfactors", "bw", "btp", "t-code", "tcode", "transaction", "idoc", "bapi", "rfc", "smartforms", "sapscript", "odata", "client administration", "scc4", "sccl", "scc3", "basis"]
        developer_keywords = ["python", "javascript", "react", "node", "fastapi", "django", "flask", "mongodb", "mysql", "postgresql", "docker", "kubernetes", "aws", "azure", "java", "spring", "typescript", "angular", "vue", "express", "api", "rest", "graphql"]
        sap_matches = [k for k in sap_keywords if k in content_lower]
        dev_matches = [k for k in developer_keywords if k in content_lower]
        self.session.extracted_topics_for_questions = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('#') or line.endswith(':') or any(word in line.lower() for word in ['understanding', 'creating', 'configuring', 'implementing', 'troubleshooting', 'best practices', 'types of', 'step-by-step'])):
                topic = line.strip('#').strip('0123456789.').strip(':').strip()
                if len(topic) > 5 and len(topic) < 100: self.session.extracted_topics_for_questions.append(topic)
        concept_patterns = [r"(?:about|understand|learn)\s+(.+?)(?:\.|,|and|$)", r"(?:creating|configuring|implementing)\s+(.+?)(?:\.|,|and|$)", r"(?:using|with)\s+([A-Z][a-zA-Z0-9\s]+)(?:\.|,|and|$)", r"(?:T-code|transaction)\s+([A-Z0-9]+)"]
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) > 3 and len(match) < 50: self.session.extracted_topics_for_questions.append(match.strip())
        seen = set()
        unique_topics = []
        for topic in self.session.extracted_topics_for_questions:
            topic_lower = topic.lower()
            if topic_lower not in seen and len(topic) > 5: seen.add(topic_lower); unique_topics.append(topic)
        self.session.extracted_topics_for_questions = unique_topics[:20]
        if len(sap_matches) > len(dev_matches): self.session.extracted_technologies = list(set(sap_matches))[:15]
        elif len(dev_matches) > 0: self.session.extracted_technologies = list(set(dev_matches))[:15]
        else: self.session.extracted_technologies = []
        project_patterns = [r"worked on (.+?)(?:\.|,|and)", r"built (.+?)(?:\.|,|and)", r"developed (.+?)(?:\.|,|and)", r"implemented (.+?)(?:\.|,|and)", r"created (.+?)(?:\.|,|and)", r"configured (.+?)(?:\.|,|and)", r"managed (.+?)(?:\.|,|and)"]
        projects = []
        for pattern in project_patterns: projects.extend(re.findall(pattern, content_lower))
        self.session.extracted_projects = list(set(projects))[:10]
        challenge_patterns = [r"challenge.*?was (.+?)(?:\.|,)", r"difficult.*?(.+?)(?:\.|,)", r"problem.*?(.+?)(?:\.|,)", r"issue.*?was (.+?)(?:\.|,)", r"troubleshoot.*?(.+?)(?:\.|,)"]
        challenges = []
        for pattern in challenge_patterns: challenges.extend(re.findall(pattern, content_lower))
        self.session.extracted_challenges = list(set(challenges))[:5]
        if any(word in content_lower for word in ["team", "collaborate", "together", "group", "lead"]): self.session.extracted_team_info = ["worked in team"]
        logger.info(f"[WI] Extracted Technologies: {self.session.extracted_technologies[:5]}")
    def should_continue_round(self, stage):
        if stage == WI_InterviewStage.INTRODUCTION: return not self.session.introduction_completed
        duration = ROUND_DURATIONS.get(stage.value, 600)
        return self.session.get_round_elapsed_time() < duration
    def get_round_time_remaining(self):
        duration = ROUND_DURATIONS.get(self.session.current_stage.value, 600)
        return max(0, duration - self.session.get_round_elapsed_time())
    def add_question(self, question, concept, is_followup=False): pass


# =============================================================================
# NEW: HUMAN VOICE DETECTION + AUDIO PREPROCESSING + DEVICE HEALTH MONITOR
# =============================================================================

class HumanVoiceDetector:
    """Detects human voice and rejects non-human sounds (TV, fan, traffic, music)."""
    VOICE_FREQ_LOW = 60
    VOICE_FREQ_HIGH = 4000
    VOICE_ENERGY_THRESHOLD = 0.008  # Lowered — frontend VAD already filters noise, so audio reaching here IS voice
    VOICE_RATIO_THRESHOLD = 0.20
    ZCR_LOW = 0.01
    ZCR_HIGH = 0.45
    MIN_CONFIDENCE = 0.15  # Lowered — frontend VAD + silence detection already confirmed this is human speech
    def __init__(self, sample_rate=16000): self.sample_rate = sample_rate
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
                        logger.debug("[VAD] Decoded as WAV: %d samples, sr=%d", len(samples), self.sample_rate)
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
                    logger.debug("[VAD] Decoded with ffmpeg: %d samples, sr=%d", len(samples), target_sr)
                    return samples
                else:
                    logger.warning("[VAD] ffmpeg decode failed (rc=%d): %s", result.returncode, result.stderr[:200].decode(errors='replace'))
            except subprocess.TimeoutExpired:
                logger.warning("[VAD] ffmpeg decode timed out")
            except FileNotFoundError:
                logger.warning("[VAD] ffmpeg not found on system, cannot decode compressed audio")
            except Exception as e:
                logger.warning("[VAD] ffmpeg decode error: %s", e)
            logger.warning("[VAD] Could not decode audio data (%d bytes)", len(audio_data))
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
        if (end - start) < len(samples) * 0.5:
            logger.debug("[AUDIO] Trim would remove >50%% of audio, skipping trim")
            return samples
        return samples[start:end]
    def _to_wav_bytes(self, samples):
        pcm = (samples * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
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
            logger.error(f"[AUDIO] Preprocessing failed: {e}")
            return audio_data

class AudioDeviceHealthMonitor:
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
            return {"healthy": False, "action": "warn_user", "issue": issue, "message": "Audio device disconnected. Please check your headphones/microphone. You can continue with your built-in microphone."}
        return {"healthy": True, "action": "continue", "issue": issue}
    def reset(self): self.last_good_time = time.time(); self.consecutive_bad = 0; self.disconnect_detected = False

# =============================================================================
# ENHANCED WI_OptimizedAudioProcessor
# =============================================================================
class GroqRateLimiter:
    """Token-bucket rate limiter to prevent Groq API overload."""
    def __init__(self, max_calls_per_minute=30):
        self.max_calls = max_calls_per_minute
        self.window = 60.0
        self.calls = []
        self._lock = asyncio.Lock()
    async def acquire(self):
        async with self._lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.window]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.window - (now - self.calls[0])
                logger.warning(f"[RATE] Groq rate limit hit, throttling for {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                self.calls = [t for t in self.calls if time.time() - t < self.window]
            self.calls.append(time.time())

_groq_rate_limiter = GroqRateLimiter(max_calls_per_minute=30)


class WI_OptimizedAudioProcessor:
    def __init__(self, client_manager):
        self.client_manager = client_manager
        self.voice_detector = HumanVoiceDetector()
        self.audio_preprocessor = AudioPreprocessor()
        self.device_monitor = AudioDeviceHealthMonitor()
        self.HALLUCINATION_PHRASES = [
            # === Original hallucinations ===
            "the speaker is answering questions about their", "interview response",
            "the speaker is answering", "answering questions about their work",
            "work experience, technical skills", "technical skills, and projects",
            "thank you for watching", "thanks for watching", "please subscribe",
            "like and subscribe", "see you in the next", "bye bye", "goodbye",
            "thank you for listening", "the end", "music", "applause", "laughter",
            "silence", "inaudible", "unintelligible", "foreign",
            "speaking foreign language", "don't forget to subscribe", "hit the bell",
            "leave a comment", "check out my", "link in description", "sponsored by",
            # === Whisper silence hallucinations (generates fake speech from noise) ===
            "i'm doing great", "thanks for asking", "please continue",
            "i'm gonna say", "i'm gonna be", "i'm going to",
            "well, my friends", "bama aum", "aum", "om",
            "yar yar", "yar, yar", "blah blah", "la la la",
            "hmm hmm hmm", "mmm mmm", "uh huh uh huh",
            "gonna be thinking about it", "thinking about it",
            "i think so", "i guess so", "yeah yeah yeah",
            "okay okay", "alright alright", "right right right",
            "you know what i mean", "you know what i'm saying",
            "so so so", "um um um", "uh uh uh",
            "this is a test", "testing testing", "hello hello",
            "can you hear me", "is this on", "one two three",
            "the the the", "a a a", "and and and",
            "i don't know what to say", "i have nothing to say",
            "subtitles by", "translated by", "captioned by",
            "copyright", "all rights reserved", "narrator",
            "chapter", "verse", "ameen", "amen", "namaste",
            "shukriya", "dhanyavaad", "bahut", "accha",
            # === Whisper Indian accent artifacts ===
            "bhai", "yaar", "acha", "theek hai", "kya",
            "haan ji", "nahin", "ji haan",
        ]

    def _decode_to_wav(self, audio_data: bytes) -> bytes:
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            logger.debug("[DECODE] Audio is already WAV format")
            return audio_data
        try:
            result = subprocess.run(
                ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 'pipe:1'],
                input=audio_data, capture_output=True, timeout=10
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                logger.info("[DECODE] Converted %d bytes -> %d bytes WAV (ffmpeg)", len(audio_data), len(result.stdout))
                return result.stdout
            else:
                logger.warning("[DECODE] ffmpeg conversion failed (rc=%d): %s", result.returncode, result.stderr[:300].decode(errors='replace'))
                return None
        except subprocess.TimeoutExpired:
            logger.warning("[DECODE] ffmpeg timed out converting audio"); return None
        except FileNotFoundError:
            logger.error("[DECODE] ffmpeg not found"); return None
        except Exception as e:
            logger.error("[DECODE] Audio decode error: %s", e); return None

    async def transcribe_audio_fast(self, audio_data: bytes) -> Tuple[str, float]:
        await self.client_manager.initialize()
        # Minimum ~1 second of audio (16kHz × 2 bytes × 1 sec = 32000 bytes raw)
        # Compressed webm is smaller, so 16000 bytes ≈ ~1 sec
        if len(audio_data) < 16000:
            logger.info(f"[WI] Audio too short ({len(audio_data)} bytes < 16000), skipping")
            return "", 0.0
        
        loop = asyncio.get_event_loop()
        
        # Run CPU-bound ffmpeg decode in executor (doesn't block event loop for other users)
        decoded_wav = await loop.run_in_executor(
            self.client_manager.executor, self._decode_to_wav, audio_data
        )
        if decoded_wav is None:
            logger.warning("[WI] Could not decode audio, skipping"); return "", 0.0
        
        # Check WAV duration — reject if < 1.5 seconds (likely speaker echo, not real speech)
        wav_samples = len(decoded_wav) / 2  # 16-bit = 2 bytes per sample
        wav_duration_sec = wav_samples / 16000  # 16kHz sample rate
        if wav_duration_sec < 1.5:
            logger.info(f"[WI] WAV too short ({wav_duration_sec:.1f}s < 1.5s), likely speaker echo — skipping")
            return "", 0.0
        
        # Run CPU-bound device health check in executor
        device_health = await loop.run_in_executor(
            self.client_manager.executor, self.device_monitor.check_audio_health, decoded_wav
        )
        if not device_health["healthy"]:
            if device_health["action"] == "warn_user":
                logger.warning(f"[WI] Device disconnect: {device_health.get('message', '')}"); return "__DEVICE_DISCONNECTED__", 0.0
            elif device_health["action"] == "wait_reconnect":
                logger.info(f"[WI] Waiting for device reconnect: {device_health.get('message', '')}"); return "__DEVICE_RECONNECTING__", 0.0
        
        # Run CPU-bound VAD (numpy FFT) in executor
        is_voice, vad_confidence, vad_details = await loop.run_in_executor(
            self.client_manager.executor, self.voice_detector.is_human_voice, decoded_wav
        )
        if not is_voice:
            logger.info(f"[WI] Non-human sound rejected (conf={vad_confidence:.2f}). Skipping transcription."); return "", 0.0
        logger.info(f"[WI] Human voice confirmed (confidence={vad_confidence:.2f})")
        
        # Run CPU-bound audio preprocessing in executor
        processed_audio = await loop.run_in_executor(
            self.client_manager.executor, self.audio_preprocessor.preprocess, decoded_wav
        )
        logger.info(f"[WI] Audio preprocessed: {len(audio_data)} -> {len(processed_audio)} bytes")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(processed_audio); temp_path = tf.name
        try:
            with open(temp_path, "rb") as f: audio_bytes = f.read()
            raw_text = await self._transcribe_with_fallback(temp_path, audio_bytes)
            if not raw_text: return "", 0.0
            
            # Check if Whisper produced a hallucination from noise/speaker echo
            if self._is_whisper_hallucination(raw_text):
                logger.info(f"[WI] Whisper hallucination rejected: '{raw_text[:80]}...'")
                return "", 0.0
            
            cleaned_text = self._remove_hallucinations(raw_text)
            confidence = self._calculate_confidence(cleaned_text)
            confidence = (confidence + vad_confidence) / 2
            if confidence < 0.3: return "", confidence
            final_text = self._final_cleanup(cleaned_text)
            if len(final_text.split()) < 2: return "", 0.2
            self.device_monitor.consecutive_bad = 0; self.device_monitor.disconnect_detected = False
            return final_text, confidence
        except Exception as e:
            logger.error(f"[WI] Transcription error: {e}"); return "", 0.0
        finally:
            try: os.unlink(temp_path)
            except: pass

    def _is_whisper_hallucination(self, raw_text):
        """Detect Whisper hallucinations from speaker echo / background noise.
        
        Whisper generates confident-sounding but nonsensical text when fed:
        - AI's own speech leaking through speakers
        - Background noise (fan, AC, traffic)  
        - Very short audio clips with ambient sound
        
        Returns True if the text is likely a hallucination.
        """
        if not raw_text: return True
        text = raw_text.lower().strip()
        words = text.split()
        word_count = len(words)
        
        # 1. Check for exact hallucination phrases (Whisper generates these from noise)
        exact_hallucinations = [
            "thank you.", "thanks.", "bye.", "bye bye.", "goodbye.",
            "thank you for watching.", "thanks for watching.",
            "please subscribe.", "see you next time.",
            "you", "thank you", "thanks", "bye", "hmm", "huh",
            "okay", "ok", "oh", "ah", "um", "uh", "so", "yeah",
        ]
        if text.rstrip('.!?,') in exact_hallucinations:
            logger.info(f"[HALLUCINATION] Exact match: '{text}'")
            return True
        
        # 2. Repetitive word pattern (yar yar yar, hmm hmm hmm, etc.)
        if word_count >= 3:
            unique_words = set(w.strip('.,!?') for w in words)
            if len(unique_words) <= 2:
                logger.info(f"[HALLUCINATION] Repetitive: '{text}' ({len(unique_words)} unique words)")
                return True
        
        # 3. Very short with no real content (< 4 real words after removing fillers)
        fillers = {'uh', 'um', 'oh', 'ah', 'eh', 'hmm', 'huh', 'so', 'like', 'okay', 
                   'ok', 'yeah', 'well', 'right', 'and', 'but', 'the', 'a', 'i', 'im',
                   "i'm", 'gonna', 'going', 'to', 'be', 'its', "it's"}
        real_words = [w.strip('.,!?') for w in words if w.strip('.,!?') not in fillers and len(w.strip('.,!?')) > 1]
        if len(real_words) < 2 and word_count <= 8:
            logger.info(f"[HALLUCINATION] Too few real words: {len(real_words)} in '{text}'")
            return True
        
        # 4. Grammatically broken patterns (Whisper noise artifacts)
        broken_patterns = [
            r'\b(\w+)\s+\1\s+\1',  # Triple word: "yar yar yar"
            r'very\s+too\b',        # "very too" is never valid English
            r'\bfriends\s+are\s+very\s+too\b',  # Specific hallucination seen in logs
            r'\bgonna\s+(?:say|be)\s.*gonna\s+(?:say|be)',  # Repeated "gonna say/be"
            r'(?:hm+\s*){3,}',      # "hmm hmm hmm"
            r'(?:ya+r?\s*[,.]?\s*){3,}',  # "yar, yar, yar"
        ]
        for pattern in broken_patterns:
            if re.search(pattern, text):
                logger.info(f"[HALLUCINATION] Broken pattern in: '{text}'")
                return True
        
        # 5. Whisper "echo" detection — if transcript sounds like it's repeating the AI question
        # These are common when mic picks up AI's TTS playback
        ai_echo_phrases = [
            "please continue", "let me ask", "here's a question",
            "let's move on", "that's interesting", "good to know",
            "tell me about", "can you describe", "what do you think",
            "great to hear", "let's get to know", "nice chatting",
            "how are you doing", "ready to get started",
            "welcome to your", "weekly interview", "three rounds",
            "communication round", "technical round", "hr round",
            "behavioral questions",
        ]
        echo_matches = sum(1 for phrase in ai_echo_phrases if phrase in text)
        if echo_matches >= 2:
            logger.info(f"[HALLUCINATION] AI echo detected ({echo_matches} matches): '{text[:60]}'")
            return True
        
        # 6. Nonsense words commonly generated by Whisper from noise
        nonsense_words = [
            'bama', 'aum', 'namaste', 'shukriya', 'dhanyavaad',
            'hauptrablers', 'kafir', 'kristian', 'corazn', 'servicio',
            'kampf', 'anarchist', 'cornered', 'puppet', 'taser',
            'pewdiepie', 'morpheus', 'voldemort',
        ]
        nonsense_count = sum(1 for w in words if w.strip('.,!?') in nonsense_words)
        if nonsense_count >= 1 and word_count <= 5:
            logger.info(f"[HALLUCINATION] Nonsense words in short text: '{text}'")
            return True
        
        return False

    async def _transcribe_with_fallback(self, temp_path, audio_bytes):
        """Try Groq with retry, fall back to OpenAI Whisper on failure."""
        whisper_prompt = "Interview candidate speaking about SAP, technical projects, work experience, and professional skills."
        
        # ── Attempt 1 & 2: Groq with exponential backoff ──
        for attempt in range(2):
            try:
                await _groq_rate_limiter.acquire()
                tr = await asyncio.wait_for(
                    self.client_manager.groq_client.audio.transcriptions.create(
                        file=(temp_path, audio_bytes), model="whisper-large-v3-turbo",
                        language="en", prompt=whisper_prompt,
                    ),
                    timeout=25.0,
                )
                return tr.text.strip() if hasattr(tr, 'text') else ""
            except (asyncio.TimeoutError, Exception) as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[GROQ] Attempt {attempt+1} failed: {e}, retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
        
        # ── Attempt 3: OpenAI Whisper API fallback ──
        try:
            logger.info("[FALLBACK] Groq failed twice, falling back to OpenAI Whisper")
            await self.client_manager.initialize()
            tr = await asyncio.wait_for(
                self.client_manager.openai_client.audio.transcriptions.create(
                    file=("audio.wav", audio_bytes), model="whisper-1",
                    language="en", prompt=whisper_prompt,
                ),
                timeout=30.0,
            )
            return tr.text.strip() if hasattr(tr, 'text') else ""
        except Exception as e:
            logger.error(f"[FALLBACK] OpenAI Whisper also failed: {e}")
            return ""

    def _remove_hallucinations(self, text):
        if not text: return ""
        result = text.lower()
        for phrase in self.HALLUCINATION_PHRASES: result = result.replace(phrase, "")
        cleaned = ""
        for char in result:
            if char.isascii() or char in ".,?!'\"- ": cleaned += char
        cleaned = re.sub(r'[.]{2,}', '.', cleaned); cleaned = re.sub(r'[,]{2,}', ',', cleaned); cleaned = re.sub(r'\s+', ' ', cleaned)
        words = cleaned.split()
        if len(words) > 3:
            deduped = []; repeat_count = 0; last_word = ""
            for word in words:
                if word.lower() == last_word.lower():
                    repeat_count += 1
                    if repeat_count <= 1: deduped.append(word)
                else: repeat_count = 0; deduped.append(word)
                last_word = word
            cleaned = " ".join(deduped)
        return cleaned.strip()

    def _calculate_confidence(self, text):
        if not text: return 0.0
        words = text.split(); word_count = len(words)
        if word_count < 2: return 0.1
        real_speech_indicators = {'i', 'we', 'my', 'our', 'the', 'this', 'that', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'did', 'work', 'worked', 'use', 'used', 'project', 'system', 'data', 'client', 'team', 'experience', 'years', 'developed', 'created', 'managed', 'handled', 'implemented', 'configured', 'learned', 'know', 'think', 'believe', 'like', 'want', 'need', 'yes', 'no', 'because', 'so', 'and', 'but', 'or', 'for', 'with'}
        text_lower = text.lower()
        indicator_count = sum(1 for word in real_speech_indicators if word in text_lower)
        indicator_score = min(indicator_count / 5, 1.0); length_score = min(word_count / 10, 1.0)
        gibberish_penalty = 0.0
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.5: gibberish_penalty += 0.3
        if re.search(r'[a-z]{10,}', text_lower): gibberish_penalty += 0.2
        confidence = (indicator_score * 0.5 + length_score * 0.5) - gibberish_penalty
        return max(0.0, min(1.0, confidence))

    def _final_cleanup(self, text):
        if not text: return ""
        text = text.strip()
        if text: text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        if text and text[-1] not in '.?!': text += '.'
        return text

# =============================================================================
# WI CONVERSATION MANAGER - Main Logic
# =============================================================================

class WI_OptimizedConversationManager:
    def __init__(self, client_manager): self.client_manager = client_manager
    def _detect_user_intent(self, user_response):
        r = user_response.lower().strip()
        skip_phrases = [
            "skip this question", "skip the question", "skip question",
            "next question", "next question please", "move on",
            "next one", "next one please", "pass this", "let's skip",
            "i want to skip", "can we skip", "please skip",
            "can you skip", "skip please", "go to next",
        ]
        if r in ["skip", "next", "pass", "next please", "skip please"]:
            return "skip"
        if any(phrase in r for phrase in skip_phrases):
            return "skip"
        repeat_phrases = [
            "repeat the question", "repeat that question", "repeat question",
            "can you repeat", "could you repeat", "please repeat",
            "repeat please", "repeat it please", "say that again",
            "say it again", "say again please", "what was the question",
            "what's the question", "i didn't hear", "i didn't catch",
            "can you say that again", "one more time", "come again",
            "tell me the question again", "ask me again", "repeat it",
        ]
        if r in ["repeat", "repeat please", "say again", "come again", "pardon"]:
            return "repeat"
        if any(phrase in r for phrase in repeat_phrases):
            negation_patterns = ["don't repeat", "dont repeat", "do not repeat",
                               "no need to repeat", "not repeat", "without repeat",
                               "don't want to repeat", "dont want to repeat",
                               "no repeat", "stop repeat"]
            if any(neg in r for neg in negation_patterns):
                return "normal"
            return "repeat"
        cant_answer_phrases = [
            "i don't know", "i dont know", "i'm not sure", "im not sure",
            "no idea", "can't answer", "cant answer", "don't remember",
            "dont remember", "not sure about that", "i have no idea",
            "i don't have any idea", "no clue",
        ]
        if any(phrase in r for phrase in cant_answer_phrases):
            return "dont_know"
        return "normal"

    def _is_gibberish(self, text):
        if not text: return True
        text_lower = text.lower().strip()
        words = text_lower.split()
        word_count = len(words)
        ascii_chars = sum(1 for c in text if c.isascii())
        if len(text) > 0 and (ascii_chars / len(text)) < 0.8: return True
        if word_count > 5:
            unique_ratio = len(set(words)) / word_count
            if unique_ratio < 0.3: return True
        nonsense_patterns = [r'(.)\1{4,}', r'\b(\w+)\s+\1\s+\1\s+\1']
        for pattern in nonsense_patterns:
            if re.search(pattern, text_lower): return True
        hallucinations = [
            "thank you for watching", "please subscribe", "like and subscribe",
            "see you next time", "bye bye bye", "youtube", "mcdonald",
            "link in description", "check out my", "sponsored by",
            "the speaker is answering", "interview response",
        ]
        if any(h in text_lower for h in hallucinations): return True
        fillers = ['uh', 'um', 'oh', 'ah', 'eh', 'so', 'yeah', 'like', 'okay', 'right', 'well']
        filler_count = sum(1 for w in words if w.strip('.,!?') in fillers)
        if word_count > 5 and filler_count / word_count > 0.35:
            logger.info(f"[GIBBERISH] Too many fillers: {filler_count}/{word_count} = {filler_count/word_count:.0%}")
            return True
        whisper_random_nouns = [
            'milk', 'bomb', 'taiwan', 'soviet', 'penguin', 'puppet', 'taser',
            'iphone', 'platinum', 'kiss', 'cornered', 'lung', 'dance',
            'cooking', 'cabinet', 'alcohol', 'armor', 'dynasty', 'camera',
            'buffet', 'elsa', 'puppy', 'napkins', 'iron', 'pits', 'legs',
            'weather pattern', 'body', 'nooks', 'kampf', 'anarchist',
            'corazn', 'servicio', 'hauptrablers', 'kafir', 'kristian',
        ]
        random_noun_count = sum(1 for noun in whisper_random_nouns if noun in text_lower)
        if random_noun_count >= 2:
            logger.info(f"[GIBBERISH] Whisper random nouns detected: {random_noun_count}")
            return True
        if word_count > 10:
            comma_count = text_lower.count(',')
            if comma_count > word_count * 0.25:
                tech_words = ['sap', 'client', 'transaction', 'system', 'data', 'server',
                             'config', 'table', 'module', 'basis', 'abap', 'fiori', 'user',
                             'scc4', 'sccl', 'scc3', 'rfc', 'sm50', 'su01', 'se09']
                has_any_tech = any(tw in text_lower for tw in tech_words)
                if not has_any_tech:
                    logger.info(f"[GIBBERISH] Too many commas ({comma_count}) with no tech content")
                    return True
        if word_count > 30:
            sentences = re.split(r'[.!?]', text_lower)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            if len(sentences) < 2:
                filler_ratio = filler_count / word_count if word_count > 0 else 0
                if filler_ratio > 0.2:
                    logger.info(f"[GIBBERISH] Long text, no sentences, high fillers")
                    return True
        if word_count > 8:
            gibberish_score = 0.0
            filler_ratio = filler_count / word_count
            if filler_ratio > 0.25: gibberish_score += 0.35
            elif filler_ratio > 0.15: gibberish_score += 0.2
            elif filler_ratio > 0.08: gibberish_score += 0.1
            if random_noun_count >= 2: gibberish_score += 0.4
            elif random_noun_count == 1: gibberish_score += 0.25
            tech_words = ['sap', 'client', 'transaction', 'system', 'data', 'server',
                         'config', 'table', 'module', 'basis', 'abap', 'fiori', 'user',
                         'scc4', 'sccl', 'scc3', 'rfc', 'sm50', 'su01', 'se09',
                         'copy', 'transport', 'login', 'authorization', 'profile',
                         'instance', 'dispatcher', 'kernel', 'parameter', 'landscape']
            has_any_tech = any(tw in text_lower for tw in tech_words)
            if not has_any_tech: gibberish_score += 0.3
            comma_count = text_lower.count(',')
            if comma_count > word_count * 0.2: gibberish_score += 0.15
            if gibberish_score >= 0.50:
                logger.info(f"[GIBBERISH] Combined score {gibberish_score:.2f} (fillers={filler_ratio:.0%}, random_nouns={random_noun_count}, tech={has_any_tech})")
                return True
        return False

    def _is_off_topic(self, user_response, stage, session=None):
        """Detect if user's answer is completely unrelated to the current round.
        
        Returns (True, detected_topic) if off-topic, (False, None) if on-topic.
        
        Rules:
        - Communication round: almost everything is on-topic (casual chat)
        - Technical round: must relate to tech/work/projects — NOT movies, food, shopping
        - HR round: must relate to work experiences, behavior, career — NOT random topics
        """
        if stage == WI_InterviewStage.COMMUNICATION:
            return False, None  # Casual round, anything goes
        
        if stage == WI_InterviewStage.INTRODUCTION:
            return False, None
        
        r = user_response.lower().strip()
        words = r.split()
        word_count = len(words)
        
        # Very short answers — handled by "weak" quality, not off-topic
        if word_count < 5:
            return False, None
        
        # ── Off-topic indicators: topics that NEVER belong in tech/HR answers ──
        off_topic_categories = {
            "movies/entertainment": ['movie', 'film', 'netflix', 'series', 'episode', 'actor', 'actress', 'bollywood', 'hollywood', 'avengers', 'marvel', 'dc comics', 'spider-man', 'batman'],
            "food/cooking": ['recipe', 'cooking', 'biryani', 'pizza', 'burger', 'restaurant', 'kitchen', 'ingredients', 'breakfast', 'lunch', 'dinner', 'snack', 'dessert', 'ice cream'],
            "shopping": ['shopping', 'mall', 'bought clothes', 'discount', 'sale', 'amazon order', 'flipkart', 'online shopping', 'market', 'grocery'],
            "sports/games": ['cricket match', 'ipl', 'football match', 'world cup', 'scored goals', 'batting', 'bowling', 'pubg', 'free fire', 'gaming'],
            "social media": ['instagram', 'snapchat', 'tiktok', 'reels', 'followers', 'viral video', 'trending', 'influencer'],
            "personal life": ['girlfriend', 'boyfriend', 'dating', 'wedding', 'party last night', 'went to beach', 'vacation photos', 'temple visit', 'pilgrimage'],
            "random topics": ['weather today', 'traffic jam', 'politics', 'election', 'petrol price', 'gold rate', 'stock market crash', 'lottery', 'horoscope', 'zodiac'],
        }
        
        detected_category = None
        off_topic_matches = 0
        
        for category, keywords in off_topic_categories.items():
            for kw in keywords:
                if kw in r:
                    off_topic_matches += 1
                    detected_category = category
        
        # Need at least 1 off-topic keyword match
        if off_topic_matches == 0:
            return False, None
        
        # ── On-topic indicators: if user mentions ANY of these, it's likely on-topic ──
        # (Even if they also mention food — e.g. "I configured the SAP system after lunch")
        if stage == WI_InterviewStage.TECHNICAL:
            tech_indicators = [
                'sap', 'abap', 'fiori', 'hana', 'basis', 'client', 'transaction', 'tcode',
                't-code', 'config', 'system', 'server', 'data', 'table', 'module', 'rfc',
                'bapi', 'idoc', 'odata', 'transport', 'landscape', 'kernel', 'profile',
                'python', 'javascript', 'react', 'node', 'api', 'database', 'mongodb',
                'docker', 'aws', 'code', 'function', 'class', 'error', 'debug', 'deploy',
                'project', 'implement', 'configure', 'develop', 'build', 'test', 'query',
                'algorithm', 'architecture', 'framework', 'library', 'repository', 'git',
                'sql', 'html', 'css', 'backend', 'frontend', 'microservice', 'pipeline',
                'work', 'team', 'task', 'requirement', 'sprint', 'agile', 'production',
            ]
            # Also check session-extracted technologies
            if session and session.extracted_technologies:
                tech_indicators.extend([t.lower() for t in session.extracted_technologies])
            
            has_tech = any(t in r for t in tech_indicators)
            if has_tech:
                return False, None  # Has tech content, not off-topic
            
        elif stage == WI_InterviewStage.HR:
            hr_indicators = [
                'team', 'lead', 'manage', 'project', 'deadline', 'challenge', 'conflict',
                'colleague', 'boss', 'manager', 'feedback', 'improve', 'learn', 'grow',
                'career', 'goal', 'strength', 'weakness', 'decision', 'responsibility',
                'initiative', 'collaborate', 'communicate', 'prioritize', 'pressure',
                'failure', 'success', 'achievement', 'experience', 'situation', 'approach',
                'problem', 'solution', 'work', 'office', 'company', 'organization',
                'professional', 'skill', 'role', 'position', 'interview', 'internship',
            ]
            has_hr = any(t in r for t in hr_indicators)
            if has_hr:
                return False, None  # Has HR-relevant content
        
        # Off-topic keyword found AND no on-topic content → it's off-topic
        logger.info(f"[OFF-TOPIC] Detected category: {detected_category}, matches: {off_topic_matches}")
        return True, detected_category

    def _assess_answer_quality(self, user_response, stage=None, session=None):
        if not user_response: return "silence"
        if self._is_gibberish(user_response): return "gibberish"
        intent = self._detect_user_intent(user_response)
        if intent != "normal": return "skip" if intent == "skip" else ("repeat" if intent == "repeat" else "cant_answer")
        # Check for off-topic content (only in technical/HR rounds)
        if stage and stage in [WI_InterviewStage.TECHNICAL, WI_InterviewStage.HR]:
            is_offtopic, category = self._is_off_topic(user_response, stage, session)
            if is_offtopic: return "off_topic"
        words = len(user_response.split())
        if words <= 3: return "weak"
        strong = ["because", "therefore", "for example", "specifically", "implemented", "experience", "i think", "used", "worked", "built", "designed", "configured", "created", "developed", "managed", "handled"]
        if words >= 20 and any(k in user_response.lower() for k in strong): return "strong"
        return "neutral" if words >= 10 else "weak"

    async def _evaluate_technical_accuracy(self, session, question, answer, expected_keywords):
        if not answer or len(answer.split()) < 3: return 0.0
        await self.client_manager.initialize()
        prompt = f"""Evaluate this technical interview answer.\n\nQuestion: {question}\nAnswer: {answer}\nContext (user's work): {session.content_context[:500] if session.content_context else 'General'}\n\nRate accuracy from 0.0 to 1.0:\n- 1.0 = Correct, detailed, shows understanding\n- 0.7 = Mostly correct, some details\n- 0.5 = Partially correct, missing key points\n- 0.3 = Vague or mostly incorrect\n- 0.0 = Wrong or no real answer\n\nReply with ONLY a number between 0.0 and 1.0"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=10)
            score_text = resp.choices[0].message.content.strip()
            score = float(re.search(r"(\d+\.?\d*)", score_text).group(1))
            return min(max(score, 0.0), 1.0)
        except:
            answer_lower = answer.lower()
            if expected_keywords:
                matches = sum(1 for k in expected_keywords if k.lower() in answer_lower)
                return min(matches / len(expected_keywords), 1.0)
            return 0.5 if len(answer.split()) > 10 else 0.3

    def _extract_topics_from_response(self, response, session=None):
        response_lower = response.lower()
        if session and session.extracted_technologies: return [t for t in session.extracted_technologies if t in response_lower]
        all_tech = ["python", "javascript", "react", "node", "api", "database", "mongodb", "mysql", "docker", "aws", "frontend", "backend", "testing", "debugging", "git", "sap", "abap", "fiori", "hana", "mm", "sd", "fico"]
        return [t for t in all_tech if t in response_lower]

    def _get_unique_transition(self, session):
        used = session.conversation_state.used_transitions
        available = [t for t in COMMUNICATION_TRANSITIONS if t not in used] or COMMUNICATION_TRANSITIONS
        t = random.choice(available)
        session.conversation_state.used_transitions.append(t)
        if len(session.conversation_state.used_transitions) > 10: session.conversation_state.used_transitions = session.conversation_state.used_transitions[-10:]
        return t

    def _should_followup(self, session, quality):
        if quality in ["weak", "cant_answer", "silence", "skip", "repeat"]: return False
        if session.conversation_state.followups_on_topic >= 2: return False
        return random.random() < (0.6 if quality == "strong" else 0.4)

    def _extract_question_from_response(self, ai_message):
        if not ai_message: return "Could you please repeat your answer?"
        prefixes_to_remove = ["Of course! The question was:", "Sure, let me repeat:", "No problem! Here it is again:", "Let me repeat that:", "Here's the question again:"]
        cleaned = ai_message.strip()
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix): cleaned = cleaned[len(prefix):].strip()
        if '?' in cleaned:
            parts = cleaned.split('?')
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i].strip()
                if len(part) > 10:
                    for sep in ['. ', '! ', '\n']:
                        if sep in part: part = part.split(sep)[-1].strip()
                    return part + '?'
            last_q_idx = cleaned.rfind('?')
            return cleaned[:last_q_idx + 1].strip()
        return cleaned

    def _adjust_difficulty(self, session, quality):
        if session.current_stage != WI_InterviewStage.TECHNICAL: return
        if quality == "strong": session.current_difficulty = "hard" if session.current_difficulty == "medium" else "medium"
        elif quality in ["weak", "cant_answer"]: session.current_difficulty = "easy"

    async def _generate_communication_question(self, session, is_first=False):
        await self.client_manager.initialize()
        asked = session.get_questions_asked_in_round(WI_InterviewStage.COMMUNICATION)
        topics = ["weekend plans", "favorite food", "travel dreams", "morning routine", "favorite movie or show", "music preferences", "childhood memories", "dream vacation", "favorite season", "cooking or eating out", "pets or animals", "sports or fitness", "books or reading", "family traditions", "city or countryside", "coffee or tea", "early bird or night owl", "relaxation methods", "learning something new", "favorite holiday", "hometown memories", "friends and social life", "dream job as a child", "favorite game", "weather preferences"]
        used_topics = session.communication_topics_covered
        available = [t for t in topics if t not in used_topics]
        if not available: available = topics
        chosen_topic = random.choice(available)
        session.communication_topics_covered.append(chosen_topic)
        prompt = f"""Generate ONE friendly casual question about: {chosen_topic}\nKeep it natural like a human conversation.\nAlready asked (DO NOT repeat): {asked[-5:]}\nMAX 12 words. Just the question."""
        resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9, max_tokens=30)
        q = resp.choices[0].message.content.strip()
        q_lower = q.lower()
        for asked_q in asked:
            if self._is_similar_question(q_lower, asked_q.lower()):
                q = random.choice([f"What do you think about {chosen_topic}?", f"Tell me about your {chosen_topic}?", f"How do you feel about {chosen_topic}?"]); break
        return q if '?' in q else q + "?"

    def _is_similar_question(self, q1, q2):
        q1_clean = q1.lower().strip().rstrip('?').strip(); q2_clean = q2.lower().strip().rstrip('?').strip()
        if q1_clean == q2_clean: return True
        words1 = set(q1_clean.split()); words2 = set(q2_clean.split())
        common_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'your', 'you', 'can', 'do', 'did', 'does', 'tell', 'me', 'about', 'describe', 'explain'}
        words1 = words1 - common_words; words2 = words2 - common_words
        if len(words1) == 0 or len(words2) == 0: return False
        overlap = len(words1 & words2); min_len = min(len(words1), len(words2))
        return overlap / min_len > 0.4

    def _get_off_topic_response(self, session=None, stage=None):
        """Return a context-aware off-topic redirect.
        
        - 1st off-topic: gentle redirect
        - 2nd off-topic: firmer redirect mentioning the round
        - 3rd+ off-topic: firm redirect + warning
        """
        if not hasattr(self, '_consecutive_off_topic'): self._consecutive_off_topic = 0
        self._consecutive_off_topic += 1
        
        # Get current round name for context
        round_name = "this topic"
        if stage == WI_InterviewStage.TECHNICAL:
            round_name = "your technical work"
            if session and session.extracted_technologies:
                current_techs = [t for t in session.extracted_technologies if t not in (session.silent_topics or [])]
                if current_techs:
                    round_name = f"your work with {current_techs[0]}"
        elif stage == WI_InterviewStage.HR:
            round_name = "your work experiences and professional situations"
        
        if self._consecutive_off_topic >= 3:
            # Firm warning
            responses = [
                f"I notice your answers aren't related to {round_name}. Please try to focus on the question. Let me ask another one.",
                f"We need to stay focused on {round_name}. Let me try a different question.",
                f"That's not related to what I asked. Let's get back to {round_name}.",
            ]
        elif self._consecutive_off_topic >= 2:
            # Medium redirect
            responses = [
                f"That doesn't seem related to {round_name}. No worries, let me ask something else.",
                f"I think that's a bit off-topic. Let me ask about {round_name} instead.",
                f"Let's focus on {round_name}. Here's a different question.",
            ]
        else:
            # Gentle redirect (1st time)
            responses = [
                "I think you might not be aware of that, let me ask you something different.",
                "No worries, let me move on to a different question.",
                "That's okay, I'll ask you something else instead.",
                "I see, let me try a different topic.",
                "Alright, let's switch to another question.",
                "No problem at all, let me ask you something you might be more familiar with.",
                "That's fine, I'll move on to a different one.",
                "Okay, don't worry about that, here's another question for you.",
                "Let me ask you something different instead.",
                "I understand, let's try a different question.",
            ]
        
        if not hasattr(self, '_last_off_topic_idx'): self._last_off_topic_idx = -1
        idx = random.randint(0, len(responses) - 1)
        while idx == self._last_off_topic_idx and len(responses) > 1:
            idx = random.randint(0, len(responses) - 1)
        self._last_off_topic_idx = idx
        return responses[idx]
    
    def _reset_off_topic_counter(self):
        """Reset consecutive off-topic counter when user gives an on-topic answer."""
        self._consecutive_off_topic = 0

    async def _generate_dynamic_ack(self, context, tone="friendly"):
        await self.client_manager.initialize()
        prompts = {
            "weak": "Generate ONE short understanding response when someone gives unclear answer. Like 'I see, let me try another question' or 'Okay, let's move on'. MAX 8 words.",
            "good": "Generate ONE short positive acknowledgment like 'That's nice!' or 'Good to know!' MAX 5 words.",
            "technical_good": "Generate ONE short acknowledgment for a good technical answer. Like 'Good point.' or 'Right.' or 'Okay, good.' MAX 4 words. Do NOT say 'impressive' or 'exactly right'.",
            "technical_weak": "Generate ONE short understanding response for unclear technical answer. Like 'I see.' or 'Okay.' MAX 5 words.",
            "cant_answer": "Generate ONE short supportive response when someone can't answer, like 'No problem, let's try something else'. MAX 10 words.",
            "transition": "Generate ONE short transition phrase like 'Okay!' or 'Alright.' MAX 3 words. Do NOT say 'impressive' or 'great insight'.",
            "hr": "Generate ONE short professional acknowledgment like 'Thank you for sharing' or 'Good point'. MAX 5 words.",
        }
        prompt = prompts.get(tone, prompts["good"])
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.9, max_tokens=20)
            ack = resp.choices[0].message.content.strip().replace('"', '').replace("'", "")
            if not ack.endswith(('!', '.', '?')): ack += '!'
            return ack
        except:
            fallbacks = {"weak": "I see. Let me ask something else.", "good": "Nice!", "technical_good": "Good explanation!", "technical_weak": "Okay, let's try another one.", "cant_answer": "No problem! Let's move on.", "transition": "Interesting!", "hr": "Thank you."}
            return fallbacks.get(tone, "Okay!")

    async def _generate_communication_followup(self, session, user_response):
        await self.client_manager.initialize()
        prompt = f"""User said: "{user_response[:100]}"\nGenerate a short follow-up question. MAX 12 words."""
        resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=30)
        q = resp.choices[0].message.content.strip()
        return q if '?' in q else q + "?"

    async def _generate_technical_question(self, session, user_response="", include_behavioral=False):
        await self.client_manager.initialize()
        if not hasattr(session, 'total_technical_questions_generated'): session.total_technical_questions_generated = 0
        session.total_technical_questions_generated += 1
        if not hasattr(session, 'used_technical_templates'): session.used_technical_templates = []
        if not hasattr(session, 'used_behavioral_templates'): session.used_behavioral_templates = []
        all_asked_questions = list(session.questions_asked)
        response_quality = "none"; should_followup = False; prefix = ""
        if user_response:
            response_lower = user_response.lower().strip(); word_count = len(response_lower.split())
            bad_indicators = ["thank you", "skip", "next", "i don't know", "no idea", "can't answer", "pass", "move on", "bye", "i can't", "don't understand", "not sure", "no clue", "don't remember", "hello", "hi", "okay", "ok", "yes", "no"]
            words = response_lower.split(); unique_words = set(words)
            is_repetitive = len(words) > 3 and len(unique_words) < len(words) * 0.4
            tech_keywords = ['sap', 'client', 'transaction', 't-code', 'config', 'system', 'data', 'user', 'table', 'module', 'basis', 'abap', 'fiori', 'report', 'program', 'function', 'process', 'implement', 'configure', 'setup', 'install', 'error', 'issue', 'problem', 'solution', 'project', 'team', 'work', 'experience', 'used', 'created', 'developed', 'managed', 'handled', 'deployed']
            has_tech_content = any(kw in response_lower for kw in tech_keywords)
            irrelevant = ['mcdonald', 'youtube', 'google', 'phone', 'rupee', 'otp', 'video', 'movie', 'song', 'food', 'hospital', 'cookie']
            has_irrelevant = any(irr in response_lower for irr in irrelevant)
            is_bad_answer = (word_count < 8 or is_repetitive or has_irrelevant or any(indicator == response_lower.strip() for indicator in bad_indicators) or (word_count < 15 and not has_tech_content))
            if is_bad_answer:
                response_quality = "bad"; prefix = self._get_off_topic_response(session=session, stage=WI_InterviewStage.TECHNICAL) + " "
                if session.exchanges:
                    last_q = session.exchanges[-1].ai_message.lower()
                    for tech in (session.extracted_technologies or []):
                        if tech.lower() in last_q and tech not in session.silent_topics: session.silent_topics.append(tech); break
            elif word_count >= 20 and has_tech_content:
                response_quality = "good"; should_followup = True; prefix = self._get_encouragement() + " "
        if should_followup and user_response:
            follow_up = await self._generate_followup_from_answer(session, user_response, all_asked_questions)
            if follow_up: return f"{prefix}{follow_up}", ["followup"]
        technologies = [t for t in (session.extracted_technologies or []) if t not in session.silent_topics]
        if not technologies: technologies = ["your work experience", "your daily tasks", "your technical skills"]
        total_qs = session.technical_question_count + session.behavioral_question_count
        should_be_behavioral = (include_behavioral and total_qs > 0 and total_qs % 4 == 3)
        if should_be_behavioral:
            session.behavioral_question_count += 1
            return await self._generate_technical_behavioral_question_dynamic(session, technologies, all_asked_questions, prefix)
        session.technical_question_count += 1
        tech_idx = session.current_tech_index % len(technologies); chosen_tech = technologies[tech_idx]; session.current_tech_index += 1
        question = await self._generate_dynamic_question_from_summary(session, chosen_tech, all_asked_questions)
        full_question = f"{prefix}{question}" if prefix else question
        if chosen_tech not in session.technical_topics_covered: session.technical_topics_covered.append(chosen_tech)
        return full_question, [chosen_tech]

    async def _generate_technical_behavioral_question_dynamic(self, session, technologies, all_asked, prefix=""):
        await self.client_manager.initialize()
        tech_idx = session.current_tech_index % len(technologies); chosen_tech = technologies[tech_idx]
        summary_context = session.content_context[:1000] if session.content_context else ""
        prompt = f"""Generate ONE technical behavioral interview question for a candidate who works with {chosen_tech}.

CANDIDATE'S BACKGROUND:
{summary_context}

Ask about a REAL TECHNICAL SCENARIO specifically related to {chosen_tech} as mentioned in the background above.
ONLY ask about topics that appear in the candidate's background — do NOT invent unrelated topics.

Vary the phrasing. Use ONE of these styles randomly:
- "Tell me about a time when..."
- "Walk me through how you handled..."
- "What was the most difficult part of working with..."
- "How did you approach [specific task] with..."
- "What would you do if [specific situation] happened with..."

DO NOT ask generic HR questions like "tell me about leadership".
DO NOT start every question with "Can you describe a challenge..."

ALREADY ASKED (DO NOT REPEAT):
{chr(10).join(all_asked[-15:])}

Generate ONE specific question (MAX 25 words):"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=60)
            question = resp.choices[0].message.content.strip().strip('"').strip("'")
            if not question.endswith('?'): question += '?'
        except Exception as e:
            logger.error(f"Error generating technical behavioral question: {e}")
            question = f"Tell me about a challenging technical problem you solved with {chosen_tech}?"
        full_question = f"{prefix}{question}" if prefix else question
        session.used_behavioral_questions.append(question)
        logger.info(f"[WI] Technical Behavioral (Dynamic): {question[:60]}...")
        return full_question, [chosen_tech, "technical_behavioral"]

    async def _generate_dynamic_question_from_summary(self, session, tech, all_asked):
        await self.client_manager.initialize()
        summary = session.content_context or "General technical work"

        # ── FIX 1: Question Type Rotation (8 types, shuffled per session) ──
        # Each session gets a random order of 8 fundamentally different question types
        # This ensures variety at scale — no two interviews feel the same
        if not hasattr(session, '_question_type_order'):
            session._question_type_order = [
                "theory", "practical", "scenario", "troubleshooting",
                "comparison", "architecture", "best_practice", "real_world",
            ]
            random.shuffle(session._question_type_order)
            session._question_type_index = 0

        q_type = session._question_type_order[session._question_type_index % len(session._question_type_order)]
        session._question_type_index += 1

        # Track recent question starters to avoid repetitive phrasing
        if not hasattr(session, '_recent_q_starters'): session._recent_q_starters = []
        avoid_starters = ", ".join(session._recent_q_starters[-3:]) if session._recent_q_starters else "none"

        type_instructions = {
            "theory": f"Ask a CONCEPT KNOWLEDGE question about {tech}. Test what they know.\nExamples: 'What is {tech}?', 'What are the key features of {tech}?', 'Explain the purpose of {tech}.'\nDo NOT ask about challenges. Just test their understanding.",
            "practical": f"Ask a HOW-TO / STEPS question about {tech}. Ask about the PROCESS.\nExamples: 'How do you configure {tech}?', 'What steps do you follow to set up {tech}?', 'Walk me through using {tech}.'",
            "scenario": f"Ask a REAL EXPERIENCE question about {tech}. Ask about a SPECIFIC situation they faced.\nExamples: 'Tell me about a time you used {tech} to solve a problem.', 'Describe a project where {tech} was critical.'",
            "troubleshooting": f"Ask a DEBUGGING / ERROR HANDLING question about {tech}.\nExamples: 'What would you do if {tech} threw an error?', 'How do you troubleshoot issues with {tech}?', 'What common problems occur with {tech}?'",
            "comparison": f"Ask a COMPARE / DIFFERENTIATE question about {tech}.\nExamples: 'What is the difference between [X] and [Y] in {tech}?', 'When would you choose [approach A] over [approach B]?'",
            "architecture": f"Ask a SYSTEM DESIGN / ARCHITECTURE question about {tech}.\nExamples: 'How does {tech} fit into the overall system?', 'What components interact with {tech}?', 'How would you design a solution using {tech}?'",
            "best_practice": f"Ask a BEST PRACTICES / STANDARDS question about {tech}.\nExamples: 'What best practices do you follow with {tech}?', 'How do you ensure quality when working with {tech}?', 'What mistakes should be avoided?'",
            "real_world": f"Ask for a SPECIFIC REAL EXAMPLE from their work with {tech}.\nExamples: 'Give me a concrete example of how you used {tech}.', 'What was the output or result of your {tech} work?', 'Show me your understanding with a real example.'",
        }

        prompt = f"""Generate ONE technical interview question for a candidate.

CANDIDATE'S WORK SUMMARY:
{summary[:1500]}

TOPIC: {tech}

QUESTION TYPE: {q_type.upper()}
{type_instructions[q_type]}

STRICT RULES:
1. ONLY ask about topics that appear in the candidate's work summary above
2. Do NOT ask about topics NOT in the summary (no random SAP modules, no PP, no MM unless mentioned)
3. The question MUST be specifically about {tech} as mentioned in the summary
4. Do NOT start with "Can you describe a challenge..." — vary the phrasing

ALREADY ASKED (DO NOT REPEAT OR ASK SIMILAR):
{chr(10).join(all_asked[-15:])}

PHRASING RULE: Do NOT start the question with the same words as recent questions.
Recent question starters to AVOID: {avoid_starters}

MAX 20 words. Just the question:"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.8, max_tokens=50)
            question = resp.choices[0].message.content.strip().strip('"').strip("'")
            if not question.endswith('?'): question += '?'
            # Track question starter for anti-repetition
            first_words = ' '.join(question.split()[:3])
            session._recent_q_starters.append(first_words)
            if len(session._recent_q_starters) > 6: session._recent_q_starters = session._recent_q_starters[-6:]
            logger.info(f"[WI] Technical Q ({q_type}): {question[:60]}...")
            return question
        except Exception as e:
            logger.error(f"Error generating dynamic question: {e}")
            return f"Tell me more about your experience with {tech}?"

    def _get_encouragement(self):
        responses = [
            "Good explanation.", "Well explained.", "Good answer.", "Right, good.",
            "That's correct.", "Good point.", "Nice, you know this well.",
            "Okay, good.", "That makes sense.", "Good understanding.",
            "Right.", "Yes, that's correct.", "Good, I can see you understand this.",
        ]
        if not hasattr(self, '_last_enc_idx'): self._last_enc_idx = -1
        idx = random.randint(0, len(responses) - 1)
        while idx == self._last_enc_idx and len(responses) > 1:
            idx = random.randint(0, len(responses) - 1)
        self._last_enc_idx = idx
        return responses[idx]

    async def _generate_followup_from_answer(self, session, user_response, all_asked):
        await self.client_manager.initialize()
        summary = session.content_context[:500] if session.content_context else ""
        prompt = f"""The candidate answered: "{user_response[:300]}"

Their work context: {summary}

Generate ONE short follow-up question to dig deeper into what they mentioned.
ONLY ask about topics that appear in their work context above.
Ask about: Specific details, How they did it, What tools they used, Results achieved.

ALREADY ASKED (DO NOT REPEAT):
{chr(10).join(all_asked[-10:])}

MAX 15 words. Just the question:"""
        try:
            resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=40)
            question = resp.choices[0].message.content.strip()
            if not question.endswith('?'): question += '?'
            is_duplicate = any(self._is_similar_question(question.lower(), aq.lower()) for aq in all_asked)
            if not is_duplicate: return question
        except: pass
        return None

    def _normalize_question(self, question):
        if not question: return ""
        q = question.lower().strip().rstrip('?').strip()
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'your', 'you', 'can', 'do', 'did', 'does', 'tell', 'me', 'about', 'describe', 'explain', 'please', 'could', 'would', 'should', 'to', 'in', 'on', 'for', 'with'}
        words = [w for w in q.split() if w not in stop_words and len(w) > 2]
        return ' '.join(sorted(words))

    async def _generate_hr_question(self, session, db_manager=None):
        if not hasattr(session, 'asked_question_hashes'):
            session.asked_question_hashes = set()
            for q in session.questions_asked: session.asked_question_hashes.add(self._normalize_question(q))
        if not hasattr(session, 'hr_category_counts'): session.hr_category_counts = {'introduction': 0, 'behavioral': 0, 'leadership': 0, 'logical_thinking': 0}
        if not hasattr(session, 'hr_questions_by_category'): session.hr_questions_by_category = {}
        CATEGORY_LIMITS = {'introduction': 2, 'behavioral': 3, 'leadership': 3, 'logical_thinking': 2}
        if not session.previously_asked_hr_questions and db_manager:
            try:
                session.previously_asked_hr_questions = await db_manager.get_hr_questions_asked(session.student_id, limit=200)
                logger.info(f"[HR] Loaded {len(session.previously_asked_hr_questions)} previously asked HR questions")
                for q in session.previously_asked_hr_questions: session.asked_question_hashes.add(self._normalize_question(q))
            except Exception as e:
                logger.warning(f"[HR] Could not load previous HR questions: {e}"); session.previously_asked_hr_questions = []
        if not session.hr_questions_by_category:
            if db_manager:
                try: await self._load_hr_questions_by_category(session, db_manager)
                except Exception as e: logger.warning(f"[HR] Could not load from MongoDB: {e}")
            if not session.hr_questions_by_category:
                logger.warning("[HR] Using fallback questions")
                session.hr_questions_by_category = {'introduction': GENERIC_TECHNICAL_QUESTIONS[:5], 'behavioral': HR_QUESTIONS_POOL[:5], 'leadership': HR_QUESTIONS_POOL[5:10], 'logical_thinking': HR_QUESTIONS_POOL[10:15]}
        total_hr_asked = sum(session.hr_category_counts.values())
        logger.info(f"[HR] Total HR questions asked so far: {total_hr_asked}")
        logger.info(f"[HR] Category counts: {session.hr_category_counts}")
        target_category = None
        category_order = ['introduction', 'behavioral', 'leadership', 'logical_thinking']
        for category in category_order:
            if session.hr_category_counts[category] < CATEGORY_LIMITS[category]: target_category = category; break
        if target_category is None:
            logger.info("[HR] All category limits reached - HR round complete")
            return "Thank you! That concludes our HR round. You did great!", ["hr_complete"]
        logger.info(f"[HR] Asking from category: {target_category} (current: {session.hr_category_counts[target_category]}/{CATEGORY_LIMITS[target_category]})")
        category_questions = session.hr_questions_by_category.get(target_category, [])
        if not category_questions:
            logger.warning(f"[HR] No questions available for category: {target_category}")
            for fallback_cat in category_order:
                if fallback_cat != target_category and session.hr_questions_by_category.get(fallback_cat):
                    category_questions = session.hr_questions_by_category[fallback_cat]; target_category = fallback_cat; break
        all_asked = set(session.used_hr_questions) | set(session.previously_asked_hr_questions)
        selected_question = None
        shuffled = category_questions.copy(); random.shuffle(shuffled)
        for question in shuffled:
            q_normalized = self._normalize_question(question)
            if q_normalized not in session.asked_question_hashes:
                is_similar = False
                for asked_q in all_asked:
                    if self._is_similar_question(question.lower(), asked_q.lower()): is_similar = True; break
                if not is_similar: selected_question = question; break
        if not selected_question and category_questions:
            selected_question = random.choice(category_questions); logger.warning(f"[HR] All questions in {target_category} used, selecting random")
        if not selected_question:
            fallback_questions = {'introduction': "What motivated you to choose your career path?", 'behavioral': "Tell me about a challenging situation you faced at work.", 'leadership': "Describe a time when you took initiative on a project.", 'logical_thinking': "How do you approach solving complex problems?"}
            selected_question = fallback_questions.get(target_category, "What are your career goals?")
        session.asked_question_hashes.add(self._normalize_question(selected_question))
        session.used_hr_questions.append(selected_question)
        session.hr_category_counts[target_category] += 1
        if db_manager:
            try: await db_manager.store_hr_question_asked(student_id=session.student_id, question=selected_question, session_id=session.session_id)
            except Exception as e: logger.warning(f"[HR] Could not store question: {e}")
        logger.info(f"[HR] Selected [{target_category.upper()}] ({session.hr_category_counts[target_category]}/{CATEGORY_LIMITS[target_category]}): {selected_question[:60]}...")
        return selected_question, ["hr", target_category]

    async def _load_hr_questions_by_category(self, session, db_manager):
        try:
            from pymongo import MongoClient
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client["ml_notes"]; collection = db["HR&Managerial_Interview_Questions"]
            logger.info("[HR] Loading questions from MongoDB by category...")
            doc = collection.find_one({"candidate_type": "fresher"})
            if not doc: logger.warning("[HR] No 'fresher' document found, trying any document"); doc = collection.find_one({})
            if not doc: logger.error("[HR] Collection is empty!"); client.close(); return
            session.hr_questions_by_category = {'introduction': [], 'behavioral': [], 'leadership': [], 'logical_thinking': []}
            for category in ['introduction', 'behavioral', 'leadership', 'logical_thinking']:
                if category in doc and isinstance(doc[category], dict):
                    category_data = doc[category]
                    if "questions" in category_data and isinstance(category_data["questions"], list):
                        questions = []
                        for q_obj in category_data["questions"]:
                            if isinstance(q_obj, dict) and "text" in q_obj:
                                q_text = str(q_obj["text"]).strip()
                                if len(q_text) > 10: questions.append(q_text)
                        session.hr_questions_by_category[category] = questions
                        logger.info(f"[HR] Loaded {len(questions)} questions from '{category}'")
                else: logger.warning(f"[HR] Category '{category}' not found in document")
            client.close()
            total = sum(len(qs) for qs in session.hr_questions_by_category.values())
            logger.info(f"[HR] Total questions loaded: {total}")
        except Exception as e:
            logger.error(f"[HR] Error loading questions by category: {e}")
            import traceback; traceback.print_exc(); raise
        
    async def _generate_smart_followup(self, session, user_response, current_stage):
        await self.client_manager.initialize()
        prompt = f"""User said: "{user_response[:80]}"\nGenerate a short follow-up question. MAX 12 words."""
        resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=30)
        q = resp.choices[0].message.content.strip()
        return q if '?' in q else q + "?"

    async def generate_first_question(self, session): return await self.generate_introduction(session)

    async def generate_introduction(self, session):
        return f"""Hello {session.student_name}! Welcome to your weekly interview session. I'm excited to chat with you today!\n\nWe'll have three rounds:\n• First, a Communication round (about 5 minutes) where we'll have a casual conversation and get to know each other.\n• Then, a Technical round (about 25 minutes) where we'll discuss your recent work and technical knowledge.\n• Finally, an HR round (about 10 minutes) with some behavioral questions.\n\nSo, how are you doing today? Ready to get started?"""

    async def generate_silence_response(self, session):
        # Increment here — this is the single source of truth for silence counting
        session.silence_prompt_count += 1
        count = session.silence_prompt_count
        
        # Progressive responses — get more helpful as silence continues
        if count == 1:
            responses = [
                "No rush, just think about it and let me know.",
                "Take your time, I'm listening.",
                "It's okay, think it through and answer when ready.",
                "No pressure at all, just share your thoughts whenever you're ready.",
                "Take a moment to think, I'll wait.",
            ]
        elif count == 2:
            responses = [
                "Are you ready? I can repeat the question if that helps.",
                "Still thinking? That's totally fine. Want me to repeat it?",
                "Can I help? I can rephrase the question if you'd like.",
                "Should I move on to a different question, or would you like more time?",
                "No worries! Want me to repeat, or shall we try a different one?",
            ]
        else:
            responses = [
                "Let me try a different question, no problem at all.",
                "That's okay, let's move on to something else.",
                "No worries, I'll ask you something different.",
            ]
        
        if not hasattr(session, '_last_silence_idx'): session._last_silence_idx = -1
        idx = random.randint(0, len(responses) - 1)
        while idx == session._last_silence_idx and len(responses) > 1:
            idx = random.randint(0, len(responses) - 1)
        session._last_silence_idx = idx
        return responses[idx]

    async def generate_fast_response(self, session, user_response, db_manager=None):
        await self.client_manager.initialize()
        
        # ── ECHO DETECTION ──
        if user_response and session.exchanges:
            last_ai_msg = session.exchanges[-1].ai_message.lower()
            user_lower = user_response.lower().strip()
            user_words = set(user_lower.split())
            ai_words = set(last_ai_msg.split())
            if len(user_words) >= 3 and len(ai_words) >= 3:
                overlap = len(user_words & ai_words)
                overlap_ratio = overlap / max(len(user_words), 1)
                if overlap_ratio >= 0.85:
                    logger.info(f"[WI] ECHO DETECTED — treating as silence")
                    user_response = ""
            if len(user_lower) >= 15 and user_lower in last_ai_msg:
                logger.info(f"[WI] ECHO DETECTED (substring) — treating as silence")
                user_response = ""
        
        quality = self._assess_answer_quality(user_response, stage=session.current_stage, session=session)
        logger.info(f"[WI] Quality: {quality}, Stage: {session.current_stage.value}")
        
        # Reset silence counter on non-silence response
        if quality != "silence":
            session.silence_prompt_count = 0
        
        # Track consecutive no-response
        if quality in ("silence", "gibberish"):
            session.consecutive_no_response += 1
            logger.info(f"[WI] Consecutive no-response: {session.consecutive_no_response}/{MAX_CONSECUTIVE_SILENCE}")
        elif quality not in ("repeat",):
            if session.current_stage not in (WI_InterviewStage.TECHNICAL, WI_InterviewStage.HR):
                session.consecutive_no_response = 0
        
        session.conversation_state.last_user_response = user_response
        mentioned_tech = self._extract_topics_from_response(user_response, session)
        session.conversation_state.user_mentioned_tech.extend(mentioned_tech)

        # ══════════════════════════════════════════════════════════
        # REPEAT HANDLING (Pure question, no prefix)
        # ══════════════════════════════════════════════════════════
        if quality == "repeat":
            if session.exchanges:
                if session._last_real_question:
                    original_question = session._last_real_question
                elif session.conversation_state.last_pure_question:
                    original_question = session.conversation_state.last_pure_question
                else:
                    last_ai_msg = session.exchanges[-1].ai_message
                    original_question = self._extract_question_from_response(last_ai_msg)
                session.last_was_repeat = True
                logger.info(f"[WI] REPEAT — returning pure question: {original_question[:80]}...")
                return original_question
            return "Let me start with a question!"

        session.last_was_repeat = False
        
        # ══════════════════════════════════════════════════════════
        # CHECK FOR ROUND TRANSITIONS
        # ══════════════════════════════════════════════════════════
        if session.current_stage == WI_InterviewStage.INTRODUCTION:
            session.introduction_completed = True
            session.start_round(WI_InterviewStage.COMMUNICATION)
            q = await self._generate_communication_question(session, True)
            return f"Great to hear! Let's get to know you. {q}"
        
        elapsed = session.get_round_elapsed_minutes()
        logger.info(f"[WI] Stage: {session.current_stage.value}, Elapsed: {elapsed:.2f} min")
        
        # Auto-skip to next round if max silence reached
        if session.consecutive_no_response >= MAX_CONSECUTIVE_SILENCE:
            logger.info(f"[WI] MAX SILENCE REACHED — auto-skipping round")
            if session.current_stage == WI_InterviewStage.COMMUNICATION:
                session.start_round(WI_InterviewStage.TECHNICAL)
                q, keywords = await self._generate_technical_question(session)
                session.add_exchange(q, expected_keywords=keywords, question_type="technical")
                return f"Let's move on to the technical round. {q}"
            elif session.current_stage == WI_InterviewStage.TECHNICAL:
                session.start_round(WI_InterviewStage.HR)
                q, keywords = await self._generate_hr_question(session, db_manager)
                if "hr_complete" in keywords:
                    session.current_stage = WI_InterviewStage.COMPLETE
                    return "Thank you! Great interview. Let me generate your detailed feedback..."
                session.add_exchange(q, expected_keywords=keywords, question_type="hr")
                return f"Let's move on to HR questions. {q}"
            elif session.current_stage == WI_InterviewStage.HR:
                session.current_stage = WI_InterviewStage.COMPLETE
                return "Thank you! Great interview. Let me generate your detailed feedback..."
        
        # Check time limits
        if session.current_stage == WI_InterviewStage.COMMUNICATION and elapsed >= 5:
            session.start_round(WI_InterviewStage.TECHNICAL)
            q, keywords = await self._generate_technical_question(session)
            session.add_exchange(q, expected_keywords=keywords, question_type="technical")
            return f"Nice chatting! Now let's discuss your technical work. {q}"
        elif session.current_stage == WI_InterviewStage.TECHNICAL and elapsed >= 25:
            session.start_round(WI_InterviewStage.HR)
            q, keywords = await self._generate_hr_question(session, db_manager)
            if "hr_complete" in keywords:
                session.current_stage = WI_InterviewStage.COMPLETE
                return "Thank you! Great interview. Let me generate your detailed feedback..."
            session.add_exchange(q, expected_keywords=keywords, question_type="hr")
            return f"Great technical discussion! Now some behavioral questions. {q}"
        elif session.current_stage == WI_InterviewStage.HR and elapsed >= 10:
            session.current_stage = WI_InterviewStage.COMPLETE
            return "Thank you! Great interview. Let me generate your detailed feedback..."
        
        # ══════════════════════════════════════════════════════════
        # DETECT IF THIS IS A CHECK-IN RESPONSE
        # ══════════════════════════════════════════════════════════
        checkin_intent = detect_checkin_response(user_response)
        if checkin_intent == "move_on":
            # User said "that's all" / "next" after a check-in
            logger.info(f"[WI] User wants to move on after check-in")
            # FORCE generate new question by skipping to generate_next action
            # Don't just set quality="normal" - that still triggers check-in logic
            # We need to jump directly to question generation
            pass  # Will be handled by forcing action_type below
        
        # ══════════════════════════════════════════════════════════
        # GET CONVERSATIONAL RESPONSE BASED ON QUALITY
        # ══════════════════════════════════════════════════════════
        response_text, action_type = get_response_for_quality(
            quality=quality,
            stage=session.current_stage.value,
            tracker=session.conversation_tracker,
            silence_count=session.silence_prompt_count,
        )
        
        # ══════════════════════════════════════════════════════════
        # OVERRIDE ACTION TYPE IF USER WANTS TO MOVE ON
        # ══════════════════════════════════════════════════════════
        if checkin_intent == "move_on":
            # User explicitly said "next" / "that's all" / "move on"
            # Force generate new question regardless of what get_response_for_quality returned
            action_type = "generate_next"
            response_text = ""
            logger.info(f"[WI] Overriding action_type to generate_next (user said move on)")
        
        logger.info(f"[WI] action_type={action_type}, response_text='{response_text[:60]}'")
        
        # ══════════════════════════════════════════════════════════
        # HANDLE EACH ACTION TYPE
        # ══════════════════════════════════════════════════════════
        
        # ── SILENCE PROMPT (just return, don't generate question) ──
        if action_type == "silence_prompt":
            return response_text
        
        # ── CHECK-IN (just return, wait for user's response) ──
        if action_type in ("checkin_strong", "checkin_weak", "checkin_followup"):
            return response_text
        
        # ── GENERATE NEXT QUESTION (all other cases) ──
        if session.current_stage == WI_InterviewStage.COMMUNICATION:
            q = await self._generate_communication_question(session)
            session.add_exchange(q, question_type="communication")
            return f"{response_text} {q}" if response_text else q
        
        elif session.current_stage == WI_InterviewStage.TECHNICAL:
            # Evaluate accuracy if previous was technical
            accuracy = 0.0
            accuracy_evaluated = False
            if session.exchanges and session.exchanges[-1].question_type == "technical":
                last_ex = session.exchanges[-1]
                accuracy = await self._evaluate_technical_accuracy(
                    session, last_ex.ai_message, user_response, last_ex.expected_keywords
                )
                session.update_last_response(user_response, 0.8, quality, accuracy)
                logger.info(f"[WI] Technical accuracy: {accuracy:.2f}")
                accuracy_evaluated = True
            
            # Reset silence counter only if accuracy > 0
            if quality not in ("silence", "gibberish", "skip", "cant_answer", "repeat"):
                if accuracy_evaluated and accuracy > 0.0:
                    session.consecutive_no_response = 0
                elif accuracy_evaluated and accuracy == 0.0:
                    session.consecutive_no_response += 1
                elif not accuracy_evaluated:
                    session.consecutive_no_response = 0
            
            self._adjust_difficulty(session, quality)
            self._reset_off_topic_counter()
            
            # Add encouragement for strong answers
            prefix = ""
            if accuracy >= 0.7:
                prefix = get_encouragement(session.conversation_tracker) + " "
            
            q, keywords = await self._generate_technical_question(session, user_response, True)
            session.add_exchange(q, expected_keywords=keywords, question_type="technical")
            return f"{prefix}{response_text} {q}" if response_text else f"{prefix}{q}"
        
        elif session.current_stage == WI_InterviewStage.HR:
            # Check if all categories done
            if hasattr(session, 'hr_category_counts'):
                HR_CATEGORY_LIMITS = {'introduction': 2, 'behavioral': 3, 'leadership': 3, 'logical_thinking': 2}
                all_done = all(
                    session.hr_category_counts.get(cat, 0) >= limit
                    for cat, limit in HR_CATEGORY_LIMITS.items()
                )
                if all_done:
                    session.current_stage = WI_InterviewStage.COMPLETE
                    return "Thank you! Great interview. Let me generate your detailed feedback..."
            
            # Evaluate accuracy if previous was HR
            accuracy = 0.0
            accuracy_evaluated = False
            if session.exchanges and session.exchanges[-1].question_type == "hr":
                last_ex = session.exchanges[-1]
                accuracy = await self._evaluate_technical_accuracy(
                    session, last_ex.ai_message, user_response, last_ex.expected_keywords
                )
                session.update_last_response(user_response, 0.8, quality, accuracy)
                accuracy_evaluated = True
            
            if quality not in ("silence", "gibberish", "skip", "cant_answer", "repeat"):
                if accuracy_evaluated and accuracy > 0.0:
                    session.consecutive_no_response = 0
                elif accuracy_evaluated and accuracy == 0.0:
                    session.consecutive_no_response += 1
                elif not accuracy_evaluated:
                    session.consecutive_no_response = 0
            
            self._reset_off_topic_counter()
            
            prefix = ""
            if accuracy >= 0.7:
                prefix = get_encouragement(session.conversation_tracker) + " "
            
            q, keywords = await self._generate_hr_question(session, db_manager)
            if "hr_complete" in keywords:
                session.current_stage = WI_InterviewStage.COMPLETE
                return "Thank you! Great interview. Let me generate your detailed feedback..."
            session.add_exchange(q, expected_keywords=keywords, question_type="hr")
            return f"{prefix}{response_text} {q}" if response_text else f"{prefix}{q}"
        
        return "That's interesting. Tell me more?"

    async def generate_fast_evaluation(self, session) -> Tuple[str, Dict[str, float]]:
        await self.client_manager.initialize()
        comm_exchanges = []; tech_exchanges = []; hr_exchanges = []; tech_accuracies = []; hr_accuracies = []
        for ex in session.exchanges:
            if ex.answer_quality in ["silence", "gibberish"] and not ex.user_response:
                continue
            exchange_data = {"question": ex.ai_message, "answer": ex.user_response if ex.user_response else "[SILENT - No response]", "is_silent": not ex.user_response or ex.answer_quality == "silence", "answer_quality": ex.answer_quality, "accuracy": ex.technical_accuracy}
            if ex.stage == WI_InterviewStage.COMMUNICATION: comm_exchanges.append(exchange_data)
            elif ex.stage == WI_InterviewStage.TECHNICAL:
                exchange_data["is_behavioral_in_tech"] = (ex.question_type == "technical_behavioral")
                tech_exchanges.append(exchange_data); (tech_accuracies.append(ex.technical_accuracy) if ex.technical_accuracy is not None else None)
            elif ex.stage == WI_InterviewStage.HR: hr_exchanges.append(exchange_data); (hr_accuracies.append(ex.technical_accuracy) if ex.technical_accuracy is not None else None)
        tech_accuracy_avg = sum(tech_accuracies) / len(tech_accuracies) if tech_accuracies else 0.5
        hr_accuracy_avg = sum(hr_accuracies) / len(hr_accuracies) if hr_accuracies else 0.5
        total_technical_qs = len(tech_exchanges); total_hr_qs = len(hr_exchanges); total_comm_qs = len(comm_exchanges)
        async def get_batch_feedback(exchanges, round_type):
            """Get feedback for ALL exchanges in one API call instead of one per question."""
            if not exchanges:
                return []
            qa_text = ""
            for i, ex in enumerate(exchanges, 1):
                if ex["is_silent"]:
                    qa_text += f"\nQ{i}: {ex['question']}\nA{i}: [SILENT - No response]\n"
                else:
                    qa_text += f"\nQ{i}: {ex['question']}\nA{i}: {ex['answer'][:200]}\n"
            prompt = f"""Give brief feedback (1 sentence each) for these {round_type} interview answers.
Reply in format:
Q1: feedback here
Q2: feedback here
...

{qa_text}

For silent responses, say "No response given. Try to attempt even partial answers."
Be constructive. If good, praise briefly. If weak, suggest improvement."""
            try:
                resp = await asyncio.wait_for(
                    self.client_manager.openai_client.chat.completions.create(
                        model=config.OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=len(exchanges) * 60
                    ),
                    timeout=45.0,
                )
                result_text = resp.choices[0].message.content.strip()
                feedbacks = []
                lines = result_text.split('\n')
                current_fb = ""
                for line in lines:
                    line = line.strip()
                    if re.match(r'^Q\d+:', line):
                        if current_fb:
                            feedbacks.append(current_fb)
                        current_fb = re.sub(r'^Q\d+:\s*', '', line)
                    elif line and current_fb:
                        current_fb += " " + line
                if current_fb:
                    feedbacks.append(current_fb)
                # Pad if fewer feedbacks parsed than exchanges
                while len(feedbacks) < len(exchanges):
                    feedbacks.append("Response recorded.")
                return feedbacks[:len(exchanges)]
            except Exception as e:
                logger.error(f"[WI] Batch feedback error: {e}")
                return ["Response recorded." for _ in exchanges]

        evaluation_parts = []
        
        # Get feedback in batches (1 API call per round instead of 1 per question)
        if comm_exchanges:
            comm_feedbacks = await get_batch_feedback(comm_exchanges, "communication")
            evaluation_parts.append("=" * 60); evaluation_parts.append("COMMUNICATION ROUND FEEDBACK"); evaluation_parts.append("=" * 60)
            for i, (ex, feedback) in enumerate(zip(comm_exchanges, comm_feedbacks), 1):
                evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}"); evaluation_parts.append(f"    User Answer: {ex['answer']}"); evaluation_parts.append(f"    Feedback: {feedback}"); evaluation_parts.append("-" * 40)
        if tech_exchanges:
            # Split pure technical from behavioral-in-technical
            pure_tech_exchanges = [ex for ex in tech_exchanges if not ex.get("is_behavioral_in_tech", False)]
            behavioral_tech_exchanges = [ex for ex in tech_exchanges if ex.get("is_behavioral_in_tech", False)]
            
            if pure_tech_exchanges:
                pure_tech_feedbacks = await get_batch_feedback(pure_tech_exchanges, "technical")
                evaluation_parts.append("\n" + "=" * 60); evaluation_parts.append(f"TECHNICAL ROUND FEEDBACK ({len(pure_tech_exchanges)} questions)"); evaluation_parts.append("=" * 60)
                for i, (ex, feedback) in enumerate(zip(pure_tech_exchanges, pure_tech_feedbacks), 1):
                    accuracy_str = f" (Accuracy: {ex['accuracy']:.0%})" if ex["accuracy"] is not None else ""
                    evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}"); evaluation_parts.append(f"    User Answer: {ex['answer']}"); evaluation_parts.append(f"    Feedback: {feedback}{accuracy_str}"); evaluation_parts.append("-" * 40)
            
            if behavioral_tech_exchanges:
                beh_feedbacks = await get_batch_feedback(behavioral_tech_exchanges, "technical behavioral")
                evaluation_parts.append("\n" + "=" * 60); evaluation_parts.append(f"TECHNICAL BEHAVIORAL QUESTIONS ({len(behavioral_tech_exchanges)} questions)"); evaluation_parts.append("=" * 60)
                for i, (ex, feedback) in enumerate(zip(behavioral_tech_exchanges, beh_feedbacks), 1):
                    accuracy_str = f" (Accuracy: {ex['accuracy']:.0%})" if ex["accuracy"] is not None else ""
                    evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}"); evaluation_parts.append(f"    User Answer: {ex['answer']}"); evaluation_parts.append(f"    Feedback: {feedback}{accuracy_str}"); evaluation_parts.append("-" * 40)
        if hr_exchanges:
            hr_feedbacks = await get_batch_feedback(hr_exchanges, "HR/behavioral")
            evaluation_parts.append("\n" + "=" * 60); evaluation_parts.append("HR/BEHAVIORAL ROUND FEEDBACK"); evaluation_parts.append("=" * 60)
            for i, (ex, feedback) in enumerate(zip(hr_exchanges, hr_feedbacks), 1):
                evaluation_parts.append(f"\nQ{i}. AI Question: {ex['question']}"); evaluation_parts.append(f"    User Answer: {ex['answer']}"); evaluation_parts.append(f"    Feedback: {feedback}"); evaluation_parts.append("-" * 40)
        evaluation_parts.append("\n" + "=" * 60); evaluation_parts.append("OVERALL SUMMARY"); evaluation_parts.append("=" * 60)
        silent_count = sum(1 for ex in comm_exchanges + tech_exchanges + hr_exchanges if ex["is_silent"])
        pure_tech_count = sum(1 for ex in tech_exchanges if not ex.get("is_behavioral_in_tech", False))
        behavioral_in_tech_count = len(tech_exchanges) - pure_tech_count
        summary_prompt = f"""Provide a brief overall interview summary (4-5 sentences) for {session.student_name}.\n\nMETRICS:\n- Communication Questions: {total_comm_qs}\n- Technical Questions: {pure_tech_count}\n- Technical Behavioral Questions: {behavioral_in_tech_count}\n- Technical Accuracy: {tech_accuracy_avg:.0%}\n- HR Questions: {total_hr_qs}\n- Correct Answers: {session.correct_answers}\n- Partial Answers: {session.partial_answers}\n- Weak Answers: {session.wrong_answers}\n- Silent/No Response: {silent_count}\n\nInclude: Overall performance, Key strengths (2-3), Areas to improve (2-3), Final recommendation"""
        summary_resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": summary_prompt}], temperature=0.3, max_tokens=400)
        overall_summary = summary_resp.choices[0].message.content.strip()
        evaluation_parts.append(f"\n{overall_summary}")
        evaluation_parts.append("\n" + "-" * 40); evaluation_parts.append("STATISTICS:")
        evaluation_parts.append(f"  Total Questions: {total_comm_qs + total_technical_qs + total_hr_qs}")
        evaluation_parts.append(f"  Technical Questions: {pure_tech_count} (+ {behavioral_in_tech_count} behavioral)")
        evaluation_parts.append(f"  Technical Accuracy: {tech_accuracy_avg:.0%}")
        evaluation_parts.append(f"  Questions Answered Well: {session.correct_answers}")
        evaluation_parts.append(f"  Partial Answers: {session.partial_answers}")
        evaluation_parts.append(f"  Needs Improvement: {session.wrong_answers}")
        evaluation_parts.append(f"  Silent Responses: {silent_count}")
        evaluation = "\n".join(evaluation_parts)
        score_prompt = f"""Score this interview candidate on a scale of 0-10 for each criteria.

ACTUAL PERFORMANCE METRICS (use these to determine scores):
- Technical Accuracy: {tech_accuracy_avg:.0%}
- Correct Answers: {session.correct_answers}
- Partial Answers: {session.partial_answers}
- Wrong/Weak Answers: {session.wrong_answers}
- Silent/No Response: {silent_count}
- Total Questions: {total_comm_qs + total_technical_qs + total_hr_qs}
- Communication Questions: {total_comm_qs}
- Technical Questions: {pure_tech_count}
- Technical Behavioral Questions: {behavioral_in_tech_count}
- HR Questions: {total_hr_qs}

STRICT SCORING RULES:
- If Technical Accuracy is below 20%, technical score MUST be 2 or below
- If Technical Accuracy is below 50%, technical score MUST be 4 or below
- If Correct Answers is 0, technical score MUST be 1 or 2
- If Wrong Answers > 10, overall scores should be LOW (1-4 range)
- If most responses were incoherent or gibberish, confidence and communication MUST be 3 or below
- Do NOT give generous scores. Be honest and accurate based on the metrics above.

Reply in EXACT format (just the scores, nothing else):
communication: X
technical: X
leadership: X
behaviour: X
confidence: X"""
        sc_resp = await self.client_manager.openai_client.chat.completions.create(model=config.OPENAI_MODEL, messages=[{"role": "user", "content": score_prompt}], temperature=0.1, max_tokens=200)
        score_text = sc_resp.choices[0].message.content.lower()
        scores = {}
        for key in ["communication", "technical", "leadership", "behaviour", "confidence"]:
            m = re.search(rf"{key}[:\s]*(\d+\.?\d*)", score_text)
            if m: scores[f"{key}_score"] = min(float(m.group(1)), 10.0)
            else:
                if key == "technical": scores[f"{key}_score"] = round(tech_accuracy_avg * 10, 1)
                else: scores[f"{key}_score"] = 5.0
        tech_cap = tech_accuracy_avg * 10
        if tech_cap < 2.0: tech_cap = max(tech_cap, 1.0)
        if scores.get("technical_score", 0) > tech_cap + 1.5:
            logger.info(f"[WI] Capping technical score from {scores['technical_score']} to {round(tech_cap + 1.0, 1)} (accuracy={tech_accuracy_avg:.0%})")
            scores["technical_score"] = round(tech_cap + 1.0, 1)
        if session.correct_answers == 0 and session.wrong_answers > 5:
            for key in ["communication", "technical", "leadership", "behaviour", "confidence"]:
                score_key = f"{key}_score"
                if scores.get(score_key, 0) > 4.0:
                    logger.info(f"[WI] Capping {key} from {scores[score_key]} to 4.0 (0 correct, {session.wrong_answers} wrong)")
                    scores[score_key] = min(scores[score_key], 4.0)
        gibberish_ratio = silent_count / max(total_comm_qs + total_technical_qs + total_hr_qs, 1)
        wrong_ratio = session.wrong_answers / max(total_comm_qs + total_technical_qs + total_hr_qs, 1)
        if wrong_ratio > 0.6 or gibberish_ratio > 0.4:
            for key in ["communication", "confidence"]:
                score_key = f"{key}_score"
                if scores.get(score_key, 0) > 3.0:
                    scores[score_key] = min(scores[score_key], 3.0)
        scores["technical_accuracy"] = round(tech_accuracy_avg * 100, 1)
        scores["hr_accuracy"] = round(hr_accuracy_avg * 100, 1)
        scores["questions_correct"] = session.correct_answers
        scores["questions_partial"] = session.partial_answers
        scores["questions_wrong"] = session.wrong_answers
        scores["questions_silent"] = silent_count
        scores["total_questions"] = total_technical_qs + total_hr_qs + total_comm_qs
        scores["communication_questions"] = total_comm_qs
        scores["technical_questions"] = pure_tech_count
        scores["behavioral_in_technical_questions"] = behavioral_in_tech_count
        scores["technical_questions_total"] = total_technical_qs
        scores["hr_questions"] = total_hr_qs
        w = {"communication_weight": 0.20, "technical_weight": 0.30, "leadership_weight": 0.15, "behaviour_weight": 0.20, "confidence_weight": 0.15}
        scores["weighted_overall"] = round(scores.get("communication_score", 5) * w.get("communication_weight", 0.2) + scores.get("technical_score", 5) * w.get("technical_weight", 0.3) + scores.get("leadership_score", 5) * w.get("leadership_weight", 0.15) + scores.get("behaviour_score", 5) * w.get("behaviour_weight", 0.2) + scores.get("confidence_score", 5) * w.get("confidence_weight", 0.15), 1)
        logger.info(f"[WI] Evaluation complete - Overall: {scores['weighted_overall']}/10, Tech Accuracy: {scores['technical_accuracy']}%")
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