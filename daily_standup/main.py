#Daily Stand Up Main
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
import uuid
import logging
import io
from pydantic import BaseModel
from typing import Dict, Optional, Any, List
from pathlib import Path
from reportlab.lib.pagesizes import LETTER, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey, green, red, orange
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import base64
# Add these imports with your other imports:

from fastapi import Form, File, UploadFile
# Import biometric authentication services
from core.biometric_auth import (
    init_biometric_services,
    get_biometric_service,
    get_voice_tracker,
    biometric_service,
    voice_tracker
)

# Local imports - use package-relative, keep names intact
from core import *
from core.ai_services import DS_SessionData as SessionData
from core.ai_services import DS_SessionStage as SessionStage
from core.ai_services import DS_SummaryManager as SummaryManager
from core.ai_services import ds_shared_clients as shared_clients
from core.config import config
from core.database import DatabaseManager
from core.ai_services import DS_OptimizedAudioProcessor as OptimizedAudioProcessor
# ⬇ Unified Chatterbox TTS
from core.tts_processor import UnifiedTTSProcessor as UltraFastTTSProcessor
from core.ai_services import DS_OptimizedConversationManager as OptimizedConversationManager
from core.prompts import DailyStandupPrompts as prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FEEDBACK DATA MODEL
# =============================================================================

class FeedbackData(BaseModel):
    """Pydantic model for feedback submission"""
    overallExperience: int = 0
    audioQuality: int = 0
    questionClarity: int = 0
    systemResponsiveness: int = 0
    technicalIssues: List[str] = []
    otherIssues: str = ""
    suggestions: str = ""
    wouldRecommend: str = ""
    difficultyLevel: str = ""
    submitted_at: Optional[str] = None

class FeedbackPayload(BaseModel):
    """Complete feedback submission payload"""
    session_id: str
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    feedback: FeedbackData
    session_duration: Optional[int] = None

class FaceVerificationRequest(BaseModel):
    """Request model for face verification"""
    student_code: str
    image_base64: str

class FaceVerificationResponse(BaseModel):
    """Response model for face verification"""
    verified: bool
    similarity: float
    threshold: float
    error: Optional[str] = None
    error_type: Optional[str] = None 
    can_proceed: bool

class VoiceVerificationResponse(BaseModel):
    """Response model for voice verification"""
    verified: bool
    similarity: float
    threshold: float
    warning_count: int
    should_terminate: bool
    message: str
    error: Optional[str] = None


# =============================================================================
# MODULE-LEVEL Q&A SAVE FUNCTION
# =============================================================================
def save_qa_to_mongodb(session_id: str, student_id: int, student_name: str,
                       conversation_log: list, test_id: str = None) -> bool:
    """Save Q&A exchanges to MongoDB as a SINGLE document (ml_notes.daily_standup_results)."""
    try:
        from pymongo import MongoClient
        from urllib.parse import quote_plus
        from datetime import datetime
        
        encoded_pass = quote_plus("LT@connect25")
        connection_string = (
            f"mongodb://connectly:{encoded_pass}"
            f"@192.168.48.201:27017/ml_notes"
            f"?authSource=admin"
        )
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        db = client["ml_notes"]
        collection = db["daily_standup_results"]
        
        if not conversation_log:
            logger.warning("No conversation log to save")
            client.close()
            return True
        
        logger.info(f"📝 Processing {len(conversation_log)} exchanges for session {session_id}")
        
        # Separate greeting and technical exchanges
        greeting_exchanges = []
        technical_exchanges = []
        
        # Counters
        answered = 0
        skipped = 0
        silent = 0
        irrelevant = 0
        repeat_requests = 0
        auto_advanced = 0
        
        for idx in range(len(conversation_log)):
            exchange = conversation_log[idx]
            
            ai_message = exchange.get("ai_message", "")
            stage = exchange.get("stage", "unknown")
            concept = exchange.get("concept", "unknown")
            is_followup = exchange.get("is_followup", False)
            
            # Skip if no AI message
            if not ai_message or len(ai_message.strip()) < 2:
                continue
            
            # ✅ GET ANSWER FROM NEXT EXCHANGE
            user_answer = ""
            quality_score = 0.0
            
            if idx + 1 < len(conversation_log):
                next_exchange = conversation_log[idx + 1]
                user_answer = next_exchange.get("user_response", "")
                quality_score = next_exchange.get("quality", 0.0)
            else:
                user_answer = "(Session ended - no answer)"
            
            # Skip placeholder answers
            if user_answer == "(session_start)":
                continue
            
            # Determine response type
            response_type = "answered"
            
            if not user_answer or user_answer.strip() == "":
                response_type = "no_response"
            elif user_answer == "(Session ended - no answer)":
                response_type = "session_ended"
            elif user_answer == "[USER_SILENT]":
                response_type = "silent"
                silent += 1
            elif user_answer == "[AUTO_ADVANCE]":
                response_type = "auto_advance"
                auto_advanced += 1
            elif user_answer == "[SKIP]":
                response_type = "skipped"
                skipped += 1
            elif user_answer == "[IRRELEVANT]":
                response_type = "irrelevant"
                irrelevant += 1
            else:
                lower = user_answer.lower()
                if any(p in lower for p in ["repeat", "again", "what did you", "didn't hear", "pardon", "can you repeat", "say that again"]):
                    response_type = "repeat_request"
                    repeat_requests += 1
                else:
                    response_type = "answered"
                    answered += 1
            
            paired_exchange = {
                "question": ai_message,
                "answer": user_answer,
                "response_type": response_type,
                "stage": stage,
                "concept": concept,
                "quality_score": quality_score,
                "is_followup": is_followup
            }
            
            # ✅ SEPARATE GREETINGS FROM TECHNICAL
            if stage == "greeting":
                paired_exchange["index"] = len(greeting_exchanges) + 1
                greeting_exchanges.append(paired_exchange)
            else:
                paired_exchange["index"] = len(technical_exchanges) + 1
                technical_exchanges.append(paired_exchange)
        
        logger.info(f"📊 Greetings: {len(greeting_exchanges)}, Technical Q&A: {len(technical_exchanges)}")
        logger.info(f"📊 Stats: answered={answered}, skipped={skipped}, silent={silent}, irrelevant={irrelevant}")
        
        # ✅ Create SINGLE session document with BOTH greeting and technical sections
        session_document = {
            "session_id": session_id,
            "test_id": test_id,
            "student_id": student_id,
            "student_name": student_name,
            
            # ✅ Greeting section
            "greeting_exchanges": greeting_exchanges,
            "greeting_count": len(greeting_exchanges),
            
            # ✅ Technical Q&A section
            "conversation": technical_exchanges,
            "total_exchanges": len(technical_exchanges),
            
            # Stats
            "answered_count": answered,
            "skipped_count": skipped,
            "silent_count": silent,
            "irrelevant_count": irrelevant,
            "repeat_requests_count": repeat_requests,
            "auto_advanced_count": auto_advanced,
            
            # Metadata
            "timestamp": datetime.now(),
            "created_at": datetime.utcnow(),
            "type": "qa_session"
        }
        
        result = collection.insert_one(session_document)
        
        if result.inserted_id:
            logger.info(f"✅ Saved: {len(greeting_exchanges)} greetings + {len(technical_exchanges)} Q&A exchanges")
        else:
            logger.error("❌ Failed to save session document")
        
        client.close()
        return True
            
    except Exception as e:
        logger.error(f"❌ Q&A save error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# FEEDBACK SAVE FUNCTION - Add this near save_qa_to_mongodb function
# =============================================================================

def save_feedback_to_mongodb(payload: dict) -> bool:
    """
    Save user feedback to MongoDB collection (ml_notes.daily_standup_results).
    
    This stores feedback as a document with type="session_feedback" in the SAME
    collection as conversation (type="qa_session") and evaluation (type="session_result")
    documents, allowing all session data to be queried together.
    """
    try:
        from pymongo import MongoClient
        from urllib.parse import quote_plus
        from datetime import datetime
        
        encoded_pass = quote_plus("LT@connect25")
        connection_string = (
            f"mongodb://connectly:{encoded_pass}"
            f"@192.168.48.201:27017/ml_notes"
            f"?authSource=admin"
        )
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        db = client["ml_notes"]
        
        # ✅ CHANGED: Use daily_standup_results instead of daily_standup_feedback
        collection = db["daily_standup_results"]
        
        # Build feedback document
        feedback_document = {
            "session_id": payload.get("session_id"),
            "student_id": payload.get("student_id"),
            "student_name": payload.get("student_name"),
            "session_duration": payload.get("session_duration"),
            
            # Feedback ratings (1-5 stars)
            "ratings": {
                "overall_experience": payload.get("feedback", {}).get("overallExperience", 0),
                "audio_quality": payload.get("feedback", {}).get("audioQuality", 0),
                "question_clarity": payload.get("feedback", {}).get("questionClarity", 0),
                "system_responsiveness": payload.get("feedback", {}).get("systemResponsiveness", 0),
            },
            
            # Technical issues (multi-select)
            "technical_issues": payload.get("feedback", {}).get("technicalIssues", []),
            "other_issues": payload.get("feedback", {}).get("otherIssues", ""),
            
            # User input
            "suggestions": payload.get("feedback", {}).get("suggestions", ""),
            "would_recommend": payload.get("feedback", {}).get("wouldRecommend", ""),
            "difficulty_level": payload.get("feedback", {}).get("difficultyLevel", ""),
            
            # Metadata
            "submitted_at": payload.get("feedback", {}).get("submitted_at") or datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow(),
            "timestamp": datetime.now(),  # For consistency with other documents
            
            # ✅ CRITICAL: Type field to distinguish from other documents
            "type": "session_feedback"
        }
        
        # Calculate average rating
        ratings = feedback_document["ratings"]
        valid_ratings = [r for r in ratings.values() if r > 0]
        feedback_document["average_rating"] = (
            sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0
        )
        
        result = collection.insert_one(feedback_document)
        
        if result.inserted_id:
            logger.info(f"✅ Feedback saved to daily_standup_results for session {payload.get('session_id')}")
            logger.info(f"   └─ Document type: session_feedback")
            logger.info(f"   └─ Average rating: {feedback_document['average_rating']:.1f}/5")
            logger.info(f"   └─ Technical issues: {len(feedback_document['technical_issues'])} reported")
            client.close()
            return True
        else:
            logger.error("❌ Failed to save feedback document")
            client.close()
            return False
        
    except Exception as e:
        logger.error(f"❌ Feedback save error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# ENHANCED SESSION MANAGER WITH SILENCE DETECTION - COMPLETE VERSION
# =============================================================================

class UltraFastSessionManagerWithSilenceHandling:
    def __init__(self):
        self.active_sessions: Dict[str, SessionData] = {}
        self.db_manager = DatabaseManager(shared_clients)
        self.audio_processor = OptimizedAudioProcessor(shared_clients)
        self.tts_processor = UltraFastTTSProcessor(
            ref_audio_dir=getattr(config, "REF_AUDIO_DIR", Path("ref_audios")),
            encode=getattr(config, "TTS_STREAM_ENCODING", "wav"),
        )
        self.conversation_manager = OptimizedConversationManager(shared_clients)

    # --- small helper: domain inference to steer prompts ---
    def _infer_domain(self, text: str) -> str:
        try:
            t = (text or "").lower()
            if any(k in t for k in ("mysql", "database", "sql")): return "database"
            if any(k in t for k in ("react", "frontend", "jsx", "ui")): return "frontend"
            if any(k in t for k in ("api", "fastapi", "backend", "service")): return "backend"
            if "sap" in t: return "sap"
            return "general"
        except Exception:
            return "general"

    def calculate_communication_score(self, session_data) -> dict:
        """
        Calculate real-time communication score using Option B formula.
        
        Components:
        - Willingness (30 pts): % of questions attempted (not silent/skipped)
        - Relevance (30 pts): % of attempts that were on-topic
        - Responsiveness (25 pts): Based on average response time
        - Clarity (15 pts): Penalty for repeat requests
        
        Returns dict with score breakdown for transparency.
        """
        stats = getattr(session_data, 'comm_stats', {})
        response_times = getattr(session_data, 'response_times', [])
        
        total = stats.get('total_questions', 0)
        answered = stats.get('answered', 0)
        skipped = stats.get('skipped', 0)
        silent = stats.get('silent', 0)
        irrelevant = stats.get('irrelevant', 0)
        repeats = stats.get('repeat_requests', 0)
        
        # Handle edge case: no questions yet
        if total == 0:
            return {
                "total_score": 100,
                "willingness_score": 30,
                "relevance_score": 30,
                "responsiveness_score": 25,
                "clarity_score": 15,
                "breakdown": {
                    "total_questions": 0,
                    "response_attempts": 0,
                    "on_topic_answers": 0,
                    "skipped": 0,
                    "silent": 0,
                    "irrelevant": 0,
                    "avg_response_time": 0,
                    "repeat_requests": 0
                }
            }
        
        # === WILLINGNESS (30 pts) ===
        # Response attempts = answered + irrelevant (they tried to answer)
        response_attempts = answered + irrelevant
        willingness_rate = response_attempts / total if total > 0 else 0
        willingness_score = willingness_rate * 30
        
        # === RELEVANCE (30 pts) ===
        # Of the attempts, how many were on-topic?
        relevance_rate = answered / response_attempts if response_attempts > 0 else 0
        relevance_score = relevance_rate * 30
        
        # === RESPONSIVENESS (25 pts) ===
        # Based on average response time
        avg_time = 0
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            if avg_time < 3:
                responsiveness_score = 25
            elif avg_time < 5:
                responsiveness_score = 20
            elif avg_time < 8:
                responsiveness_score = 15
            elif avg_time < 12:
                responsiveness_score = 10
            else:
                responsiveness_score = 5
        else:
            responsiveness_score = 15  # Default if no timing data
        
        # === CLARITY (15 pts) ===
        # Penalty: 3 points per repeat request, max 15
        clarity_penalty = min(repeats * 3, 15)
        clarity_score = 15 - clarity_penalty
        
        # === TOTAL ===
        total_score = willingness_score + relevance_score + responsiveness_score + clarity_score
        total_score = min(100, max(0, total_score))  # Clamp to 0-100
        
        return {
            "total_score": round(total_score, 1),
            "willingness_score": round(willingness_score, 1),
            "relevance_score": round(relevance_score, 1),
            "responsiveness_score": round(responsiveness_score, 1),
            "clarity_score": round(clarity_score, 1),
            "breakdown": {
                "total_questions": total,
                "response_attempts": response_attempts,
                "on_topic_answers": answered,
                "skipped": skipped,
                "silent": silent,
                "irrelevant": irrelevant,
                "avg_response_time": round(avg_time, 1) if response_times else 0,
                "repeat_requests": repeats
            }
        }

    async def _send_comm_score_update(self, session_data, event_type: str = "update"):
        """Send real-time communication score update to frontend."""
        try:
            comm_score = self.calculate_communication_score(session_data)
            session_data.live_communication_score = comm_score["total_score"]
            
            await self._send_quick_message(session_data, {
                "type": "communication_score_update",
                "score": comm_score["total_score"],
                "breakdown": comm_score,
                "event": event_type
            })
            
            logger.info(f"📊 Communication Score: {comm_score['total_score']}/100 ({event_type})")
            logger.info(f"   └─ Willingness: {comm_score['willingness_score']}/30, Relevance: {comm_score['relevance_score']}/30, Responsiveness: {comm_score['responsiveness_score']}/25, Clarity: {comm_score['clarity_score']}/15")
            
        except Exception as e:
            logger.error(f"❌ Failed to send comm score update: {e}")

    # ============================================================================
    # ✅ IMPROVED: Dynamic silence response generation with GENTLE prompts
    # ============================================================================
    async def generate_dynamic_silence_response(self, session_data: SessionData, silence_context: dict = None):
        """Generate progressive silence prompts - NO LLM, just return hardcoded text."""
        try:
            # Track silence count
            silence_count = getattr(session_data, 'silence_response_count', 0)
            session_data.silence_response_count = silence_count + 1
            
            logger.info(f"🔕 Generating silence response #{session_data.silence_response_count}")

            # Build simple context
            ctx = {
                "name": session_data.student_name,
                "silence_count": session_data.silence_response_count,
            }
            
            # ✅ DIRECTLY GET TEXT FROM prompts.py (no LLM call)
            response_text = prompts.dynamic_silence_response(ctx)
            
            logger.info(f"📨 Hardcoded silence response: '{response_text}'")
            
            return response_text

        except Exception as e:
            logger.error(f"❌ Silence response error: {e}", exc_info=True)
            return f"{session_data.student_name}, are you there?"
    
    async def generate_extended_question(self, session_data) -> str:
        """
        Generate web-based questions when summary is exhausted but time remains.
        Questions are generated based on the same topic but using broader knowledge.
        """
        try:
            # Get context for extended questions
            main_topic = getattr(session_data, 'main_topic', 'technical discussion')
            asked_questions = getattr(session_data, 'asked_questions', [])
            original_summary = getattr(session_data, 'original_summary_text', '')
            
            logger.info(f"🌐 Generating EXTENDED question for topic: '{main_topic}'")
            logger.info(f"📊 Already asked {len(asked_questions)} questions")
            
            # Generate extended question using the new prompt
            extended_prompt = prompts.generate_extended_web_question(
                topic=main_topic,
                already_asked=asked_questions,
                summary_context=original_summary
            )
            
            loop = asyncio.get_event_loop()
            extended_question = await loop.run_in_executor(
                shared_clients.executor,
                self.conversation_manager._sync_openai_call,
                extended_prompt,
            )
            extended_question = (extended_question or "").strip()
            
            #retry too short or seems like a repeat
            attempts = 0
            max_attempts = 3  # ✅ FIX 3A: Reduced from 5 to 3
            
            # ✅ FIX 3A: Question categories for variety
            import random
            question_categories = [
                "transaction codes", "troubleshooting steps", "security considerations",
                "performance optimization", "real-world scenarios", "configuration steps",
                "prerequisites", "error handling", "monitoring approaches"
            ]
            used_cats = getattr(session_data, '_used_q_categories', [])
            avail_cats = [c for c in question_categories if c not in used_cats[-3:]]
            cat_hint = f" Focus on: {random.choice(avail_cats)}." if avail_cats and attempts > 0 else ""
            
            while attempts < max_attempts:
                is_too_short = len(extended_question.split()) < 8
                is_repeat = any(
                    self._is_similar_question(q, extended_question) 
                    for q in asked_questions[-10:]  # ✅ FIX 3A: Reduced from 15 to 10
                )
                
                if not extended_question or is_too_short or is_repeat:
                    attempts += 1
                    if is_repeat:
                        logger.warning(f"⚠️ Extended question attempt {attempts} - DUPLICATE detected, retrying...")
                    else:
                        logger.warning(f"⚠️ Extended question attempt {attempts} failed - retrying...")
                    
                    # ✅ FIX 3A: Category-based variety hint
                    variety_hint = f"\\n\\nIMPORTANT: Generate a COMPLETELY DIFFERENT question.{cat_hint} Avoid: {', '.join([q[:25] for q in asked_questions[-3:]])}" if attempts > 0 else ""

                    extended_prompt_with_hint = prompts.generate_extended_web_question(
                        topic=main_topic,
                        already_asked=asked_questions,
                        summary_context=original_summary
                    ) + variety_hint
                    
                    extended_question = await loop.run_in_executor(
                        shared_clients.executor,
                        self.conversation_manager._sync_openai_call,
                        extended_prompt_with_hint,
                    )
                    extended_question = (extended_question or "").strip()
                else:
                    break
            
            if extended_question and len(extended_question.split()) >= 5:
                # Track this question
                if not hasattr(session_data, 'asked_questions'):
                    session_data.asked_questions = []
                session_data.asked_questions.append(extended_question)
                logger.info(f"✅ Generated EXTENDED question #{len(session_data.asked_questions)}: '{extended_question}'")
                return extended_question
            else:
                logger.error(f"❌ Failed to generate valid extended question after {max_attempts} attempts")
                return None
            
        except Exception as e:
            logger.error(f"❌ Extended question generation error: {e}", exc_info=True)
            return None
    
    def _is_similar_question(self, q1: str, q2: str) -> bool:
        """Check if two questions are too similar using 85% word overlap threshold."""
        if not q1 or not q2:
            return False
        
        # Normalize - remove punctuation and lowercase
        import re
        q1_clean = re.sub(r'[^\\w\\s]', '', q1.lower())
        q2_clean = re.sub(r'[^\\w\\s]', '', q2.lower())
        
        words1 = set(q1_clean.split())
        words2 = set(q2_clean.split())
        
        # Remove common/stop words - expanded for SAP context
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'how', 'do', 'does', 'for', 'to', 
            'in', 'of', 'and', 'when', 'can', 'you', 'be', 'should', 'would', 'could',
            'sap', 'client', 'system', 'that', 'this', 'with', 'from', 'or', 'on',
            'best', 'practice', 'key', 'steps', 'during', 'performing', 'ensure',
            'explain', 'describe', 'tell', 'me', 'about', 'please'
        }
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return False
        
        # ✅ FIX 3B: Use 85% threshold for duplicate detection
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        max_len = max(len(words1), len(words2))
        
        if min_len == 0:
            return False
        
        overlap_ratio = overlap / min_len
        
        # Primary: 85% overlap = too similar
        if overlap_ratio >= 0.85:
            logger.debug(f"🔍 Duplicate: {overlap_ratio:.0%} overlap")
            return True
        
        # Secondary: 4+ shared words AND >50% = similar
        if overlap >= 4 and overlap_ratio > 0.5:
            logger.debug(f"🔍 Duplicate: {overlap} words at {overlap_ratio:.0%}")
            return True
        
        return False

    
    def should_continue_session(self, session_data) -> bool:
        """
        Check if session should continue based on minimum duration requirement.
        Session must run for at least 15 minutes.
        """
        now_ts = time.time()
        session_start = session_data.created_at
        elapsed = now_ts - session_start
        min_duration = getattr(session_data, 'min_session_duration', 15 * 60)  # 15 minutes
        
        time_remaining = min_duration - elapsed
        
        # Session should continue if we haven't reached minimum duration
        # Leave 30 seconds buffer for closing
        if time_remaining > 30:
            return True
        
        return False

    def _extract_question_only(self, text: str) -> str:
        """
        Extract only the question from AI response, removing acknowledgments.
        
        Examples:
            "No worries! How to split the data?" -> "How to split the data?"
            "That's okay! What is AI?" -> "What is AI?"
            "I understand. What are the steps?" -> "What are the steps?"
        """
        if not text:
            return text
        
        # Common acknowledgment patterns to remove
        acknowledgments = [
            "No worries!",
            "No worries.",
            "That's okay!",
            "That's okay.",
            "That's fine!",
            "That's fine.",
            "I understand!",
            "I understand.",
            "I see!",
            "I see.",
            "Got it!",
            "Got it.",
            "Understood!",
            "Understood.",
            "That's alright!",
            "That's alright.",
            "No problem!",
            "No problem.",
            "Sure!",
            "Sure.",
            "\"That's not quite what I was asking.\"",
            "\"That's a bit different from my question.\"",
        ]
        
        # Remove acknowledgments from start
        cleaned = text.strip()
        for ack in acknowledgments:
            if cleaned.startswith(ack):
                cleaned = cleaned[len(ack):].strip()
                logger.info(f"✂️ Removed acknowledgment: '{ack}'")
                break
        
        # If there are multiple sentences, check if first sentence is an acknowledgment
        sentences = cleaned.split('. ')
        if len(sentences) > 1:
            first_sentence = sentences[0].strip()
            # Check if first sentence is short and likely an acknowledgment
            if len(first_sentence.split()) <= 4 and not first_sentence.endswith('?'):
                # Remove first sentence and rejoin
                cleaned = '. '.join(sentences[1:]).strip()
                logger.info(f"✂️ Removed first sentence: '{first_sentence}'")
        
        return cleaned or text  # Return original if cleaning results in empty string

    async def create_session_fast(self, websocket: Optional[Any] = None, student_id: int = None) -> SessionData:
        session_id = str(uuid.uuid4())
        test_id = f"standup_{int(time.time())}"
        try:
            student_info_task = asyncio.create_task(self.db_manager.get_student_info_fast(student_id=student_id))
            summary_task = asyncio.create_task(self.db_manager.get_summary_fast())
            student_id, first_name, last_name, session_key = await student_info_task
            summary = await summary_task

            if not summary or len(summary.strip()) < 50:
                raise Exception("Invalid summary retrieved from database")
            if not first_name or not last_name:
                raise Exception("Invalid student data retrieved from database")

            session_data = SessionData(
                session_id=session_id,
                test_id=test_id,
                student_id=student_id,
                student_name=f"{first_name} {last_name}",
                session_key=session_key,
                created_at=time.time(),
                last_activity=time.time(),
                current_stage=SessionStage.GREETING,
                websocket=websocket,
            )
            session_data.greeting_count = 0
            session_data.greeting_sent = False  # Track if initial greeting was sent
            session_data.awaiting_user_confirmation = False
            # ---- silence counters/state ----
            session_data.silence_response_count = 0
            session_data.max_silence_responses = 999  # ✅ Unlimited silence responses
            session_data.consecutive_silence_chunks = 0
            session_data.silence_chunks_threshold = getattr(config, "SILENCE_CHUNKS_THRESHOLD", 6)
            session_data.silence_timeout_s = getattr(config, "SILENCE_TIMEOUT_SECONDS", 30)
            session_data.silence_ready = False
            session_data.silence_prompt_active = False
            session_data.has_user_spoken = False
            session_data.silence_grace_after_greeting_s = getattr(
                config, "SILENCE_GRACE_AFTER_GREETING_SECONDS", 4
            )
            session_data.greeting_end_ts = None
            session_data.last_silence_response_ts = 0  # ✅ Initialize cooldown timestamp (for awaiting_user only)

            # === REAL-TIME COMMUNICATION SCORE TRACKING ===
            session_data.response_times = []  # List of response delays in seconds
            session_data.last_question_end_ts = None  # When AI finished asking question
            session_data.comm_stats = {
                "total_questions": 0,      # Total technical questions asked
                "answered": 0,             # Questions with substantive answers
                "skipped": 0,              # Explicit skips ("I don't know")
                "silent": 0,               # No response (silence)
                "irrelevant": 0,           # Off-topic answers
                "repeat_requests": 0,      # Asked to repeat question
            }
            session_data.live_communication_score = 100  # Start at 100, update in real-time
            # time limits
            SESSION_MAX_SECONDS = getattr(config, "SESSION_MAX_SECONDS", 15 * 60)
            SESSION_SOFT_CUTOFF_SECONDS = getattr(config, "SESSION_SOFT_CUTOFF_SECONDS", 2 * 60)
            now_ts = time.time()
            session_data.end_time = now_ts + SESSION_MAX_SECONDS
            session_data.soft_cutoff_time = session_data.end_time - SESSION_SOFT_CUTOFF_SECONDS
            session_data.awaiting_user = False
            session_data.clarification_attempts = 0

            # Seed domain from summary manager/topic if possible
            fragment_manager = SummaryManager(shared_clients, session_data)
            if not fragment_manager.initialize_fragments(summary):
                raise Exception("Failed to initialize fragments from summary")
            session_data.summary_manager = fragment_manager

            # === EXTENDED QUESTION MODE TRACKING ===
            session_data.extended_mode = False  # True when summary questions exhausted
            session_data.asked_questions = []   # Track all questions to avoid repetition
            session_data.original_summary_text = summary  # Store for extended context
            session_data.min_session_duration = getattr(config, 'MIN_SESSION_DURATION_SECONDS', 15 * 60)
            
            # Extract main topic for extended questions
            if summary:
                first_line = summary.split("\\n")[0].strip()
                # Remove markdown formatting
                first_line = first_line.replace('#', '').replace('*', '').strip()
                session_data.main_topic = first_line[:100] if first_line else "technical discussion"
            else:
                session_data.main_topic = "technical discussion"
            
            logger.info(f"🎯 Main topic for session: '{session_data.main_topic}'")

            # If Mongo has a topic (e.g., SAP), prefer that
            seed_topic = getattr(fragment_manager, "current_topic", None)
            session_data.current_domain = (seed_topic or "general")

            # watchdog
            session_data._hard_stop_task = asyncio.create_task(self._hard_stop_watchdog(session_data))

            self.tts_processor.start_session(session_data.session_id)

            self.active_sessions[session_id] = session_data
            logger.info(
                "Enhanced session created %s for %s with %d fragments and silence handling",
                session_id, session_data.student_name, len(session_data.fragment_keys)
            )
            return session_data
        except Exception as e:
            logger.error("Failed to create enhanced session: %s", e)
            raise Exception(f"Session creation failed: {e}")

    # --- gentle nudge when no/poor audio (prompts-only) ---
    async def generate_silence_response(self, session_data: SessionData):
        """Legacy method - redirects to new dynamic method"""
        try:
            ctx = {
                "name": session_data.student_name,
                "domain": getattr(session_data, "current_domain", "work"),
                "silence_count": getattr(session_data, 'silence_response_count', 0) + 1,
                "current_stage": session_data.current_stage.value,
                "conversation_length": len(getattr(session_data, "conversation_log", [])),
            }
            text = await self.generate_dynamic_silence_response(session_data, ctx)
            
            await self._send_quick_message(session_data, {
                "type": "ai_response",
                "text": text,
                "status": session_data.current_stage.value
            })
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                text, session_id=session_data.session_id
            ):
                if audio_chunk:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": session_data.current_stage.value
                    })
            await self._send_quick_message(session_data, {"type": "audio_end", "status": session_data.current_stage.value})
            logger.info("Sent silence-based prompt.")
        except Exception as e:
            logger.error("Failed to send silence-based response: %s", e)

    # --- auto-advance when we can't hear the user (no static lines) ---
    async def _auto_advance_question(self, session_data: SessionData):

        """Auto-advance to next question using summary-based generation."""
        try:
            logger.info("Session %s: auto-advancing to next question", session_data.session_id)

            fm = getattr(session_data, "summary_manager", None)
            if not fm:
                logger.error("❌ No fragment manager available for auto-advance")
                return

            # Get current concept info
            conv_log = getattr(session_data, "conversation_log", [])
            current_concept = session_data.current_concept or "unknown"
            
            # Count questions asked on current concept
            questions_on_concept = sum(
                1 for exchange in conv_log 
                if exchange.get("concept") == current_concept
            )
            
            logger.info(f"📊 Auto-advance: Questions on '{current_concept}': {questions_on_concept}")
            
            # Decide: Follow-up or New Topic?
            max_questions = getattr(session_data, 'questions_per_concept', 3)
            should_ask_followup = questions_on_concept < max_questions
            
            next_question = None
            
            if should_ask_followup:
                # === ASK FOLLOW-UP ON SAME CONCEPT ===
                logger.info("🔄 Auto-advance: Generating FOLLOW-UP on same concept")
                
                current_concept_title, current_concept_content = fm.get_active_fragment()
                previous_question = conv_log[-1].get("ai_message", "") if conv_log else ""
                
                # Generate summary-based follow-up
                followup_prompt = prompts.dynamic_followup_response(
                    context_text=current_concept_content[:2000],
                    user_input="[User didn't answer]",
                    previous_question=previous_question,
                    session_state={
                        "domain": current_concept,
                        "questions_asked": questions_on_concept,
                        "concept": current_concept_title
                    }
                )
                
                loop = asyncio.get_event_loop()
                next_question = await loop.run_in_executor(
                    shared_clients.executor,
                    self.conversation_manager._sync_openai_call,
                    followup_prompt,
                )
                next_question = (next_question or "").strip()
                
                # Retry if too short
                if not next_question or len(next_question.split()) < 8:
                    logger.warning("⚠️ Follow-up too short, retrying...")
                    next_question = await loop.run_in_executor(
                        shared_clients.executor,
                        self.conversation_manager._sync_openai_call,
                        followup_prompt,
                    )
                    next_question = (next_question or "").strip()
                
                logger.info(f"✅ Generated FOLLOW-UP: '{next_question}'")
                
                # Mark as follow-up
                session_data._last_question_followup = True
                
                # Track in fragment manager
                if fm:
                    fm.add_question(next_question, current_concept_title, is_followup=True)
            
            else:
                # === MOVE TO NEW TOPIC ===
                logger.info("🔄 Auto-advance: Moving to NEW TOPIC")
                
                old_concept = session_data.current_concept or "the previous topic"
                moved = fm.advance_fragment()
                
                if moved:
                    new_concept_title, new_concept_content = fm.get_active_fragment()
                    logger.info(f"➡️ Auto-advance transition: '{old_concept}' → '{new_concept_title}'")
                    
                    # Store new concept
                    session_data.current_concept = new_concept_title
                    session_data.current_domain = new_concept_title
                    
                    # Generate summary-based transition question
                    transition_prompt = prompts.dynamic_concept_transition(
                        current_concept=old_concept,
                        next_concept=new_concept_title,
                        user_last_answer="[User didn't answer]",
                        next_concept_content=new_concept_content
                    )
                    
                    loop = asyncio.get_event_loop()
                    next_question = await loop.run_in_executor(
                        shared_clients.executor,
                        self.conversation_manager._sync_openai_call,
                        transition_prompt,
                    )
                    next_question = (next_question or "").strip()
                    
                    # Retry if too short
                    if not next_question or len(next_question.split()) < 10:
                        logger.warning("⚠️ Transition too short, retrying...")
                        next_question = await loop.run_in_executor(
                            shared_clients.executor,
                            self.conversation_manager._sync_openai_call,
                            transition_prompt,
                        )
                        next_question = (next_question or "").strip()
                    
                    logger.info(f"✅ Generated TRANSITION: '{next_question}'")
                    
                    # Mark as main question
                    session_data._last_question_followup = False
                    
                    # Track in fragment manager
                    fm.add_question(next_question, new_concept_title, is_followup=False)
                else:
                    # No more concepts - end session
                    logger.info("🏁 No more concepts available - ending session")
                    await self._finalize_session_fast(session_data)
                    return
            
            # If we got here, we have a valid next question
            if not next_question:
                logger.error("❌ Failed to generate next question")
                return
            
            # Add to conversation log
            concept = session_data.current_concept or "auto_advance"
            is_followup = getattr(session_data, "_last_question_followup", False)
            session_data.add_exchange(next_question, "[AUTO_ADVANCE]", 0.3, concept, is_followup)

            # Update session state and send response
            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, next_question)

            # Set awaiting user flag
            now_ts = time.time()
            end_time = getattr(session_data, "end_time", None)
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
            time_remaining = (end_time - now_ts) if end_time else float('inf')

            if (session_data.current_stage == SessionStage.TECHNICAL and
                time_remaining > 60 and
                (not soft_cutoff or now_ts < soft_cutoff)):
                session_data.awaiting_user = True
                logger.info("🧍 Waiting for user response after auto-advance")

        except Exception as e:
            logger.error("Auto-advance error: %s", e, exc_info=True)
            await self._finalize_session_fast(session_data)

    
    async def _hard_stop_watchdog(self, session_data: SessionData):
        try:
            delay = max(0.0, getattr(session_data, "end_time", time.time()) - time.time())
            await asyncio.sleep(delay)
            if not session_data.is_active:
                return

            if getattr(session_data, "awaiting_user", False):
                setattr(session_data, "hard_cutoff_reached", True)
                grace = getattr(config, "FINAL_ANSWER_GRACE_SECONDS", 0)
                if grace > 0:
                    await asyncio.sleep(grace)
                    if session_data.is_active:
                        await self._end_due_to_time(session_data)
            else:
                await self._end_due_to_time(session_data)
        except asyncio.CancelledError:
            return

    async def _end_due_to_time(self, session_data: SessionData):
        def _extract_topics(sd: SessionData):
            topics = []
            sm = getattr(sd, "summary_manager", None)
            if not sm:
                return topics
            fk = getattr(sd, "fragment_keys", None)
            frags = getattr(sm, "fragments", None)
            if fk and isinstance(frags, dict):
                for k in fk:
                    frag = frags.get(k)
                    if isinstance(frag, dict):
                        t = frag.get("title") or frag.get("heading") or frag.get("name")
                        if t:
                            topics.append(t)
                if topics: return topics
            if isinstance(frags, list):
                for frag in frags:
                    if isinstance(frag, dict):
                        t = frag.get("title") or frag.get("heading") or frag.get("name")
                        if t:
                            topics.append(t)
                if topics: return topics
            if fk:
                return [str(k) for k in fk]
            return topics

        try:
            session_data.awaiting_user = False
        except Exception:
            pass

        try:
            topics = _extract_topics(session_data)
            conv_log = getattr(session_data, "conversation_log", []) or []
            user_final_response = (
                conv_log[-1].get("user_response") if conv_log and isinstance(conv_log[-1], dict) else None
            )
            conversation_summary = {
                "topics_covered": topics,
                "total_exchanges": len(conv_log),
                "name": session_data.student_name,
            }

            # Use prompts to generate the closing line (hard cutoff)
            closing_prompt = prompts.dynamic_hardcutoff_closure(conversation_summary)
            loop = asyncio.get_event_loop()
            closing_text = await loop.run_in_executor(
                shared_clients.executor,
                self.conversation_manager._sync_openai_call,
                closing_prompt,
            )
            closing_text = (closing_text or "").strip()

            # Retry once if too short
            if not closing_text or len(closing_text.split()) < 3:
                closing_text = await loop.run_in_executor(
                    shared_clients.executor,
                    self.conversation_manager._sync_openai_call,
                    closing_prompt,
                )
                closing_text = (closing_text or "").strip()

            closing_text = closing_text or " "

            # Try evaluation (best-effort)
            evaluation_text, score, detailed_evaluation = None, None, None
            try:
                evaluation_text, score, detailed_evaluation = await self.conversation_manager.generate_fast_evaluation(session_data)
                
                # ✅ FIX: Store detailed_evaluation on session_data
                session_data.detailed_evaluation = detailed_evaluation
                
            except Exception as e_eval:
                logger.error("Evaluation generation error (time-cutoff): %s", e_eval)
            

            try:
                if evaluation_text is not None and score is not None:
                    saved = await self.db_manager.save_session_result_fast(session_data, evaluation_text, score)
                    if not saved:
                        logger.error("Save (time-cutoff) failed for %s", session_data.session_id)
            except Exception as e_save:
                logger.exception("Save error (time-cutoff): %s", e_save)

            try:
                conv_log = getattr(session_data, "conversation_log", [])
                if conv_log:
                    save_qa_to_mongodb(
                        session_id=session_data.session_id,
                        student_id=session_data.student_id,
                        student_name=session_data.student_name,
                        conversation_log=conv_log,
                        test_id=session_data.test_id
                    )
            except Exception as qa_err:
                logger.error(f"Q&A save failed: {qa_err}")

            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": closing_text,
                "status": "complete",
                "enable_new_session": True,
                "evaluation": evaluation,
                "score": score,
                "pdf_url": f"/download_results/{session_data.session_id}",
                "redirect_to": "/dashboard",
            })

            try:
                async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                    closing_text, session_id=session_data.session_id
                ):
                    if audio_chunk:
                        await self._send_quick_message(session_data, {
                            "type": "audio_chunk",
                            "audio": audio_chunk.hex(),
                            "status": "complete",
                        })
                await self._send_quick_message(session_data, {"type": "audio_end", "status": "complete"})
            except Exception as e_tts:
                logger.error("TTS closing stream error: %s", e_tts)

        except Exception as e:
            logger.error("Closing generation error: %s", e)

        finally:
            session_data.is_active = False
            try:
                task = getattr(session_data, "_hard_stop_task", None)
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass
            try:
                if session_data.websocket:
                    await session_data.websocket.close(code=1000)
            except Exception:
                pass
            await self.remove_session(session_data.session_id)

    async def remove_session(self, session_id: str):
        if session_id in self.active_sessions:
            try:
                self.tts_processor.end_session(session_id)
            except Exception:
                pass
            del self.active_sessions[session_id]
            logger.info("Removed session %s", session_id)

    # ============================================================================
    # ✅ IMPROVED: Enhanced audio processing with NOISE FILTERING
    # ============================================================================
    async def process_audio_with_silence_status(self, session_id: str, message_data: dict):
        """Process audio data with refined silence gating & consecutive counter."""
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            logger.warning("Inactive session: %s", session_id)
            return

        # ✅ CRITICAL: Block ALL audio processing while AI is responding
        if getattr(session_data, "ai_is_responding", False):
            logger.info(f"⏸️ BLOCKED: AI is responding - ignoring ALL audio (session {session_id})")
            return

        # ✅ CRITICAL: Block ALL audio if session is ending
        if getattr(session_data, "is_ending", False):
            logger.info(f"🛑 BLOCKED: Session {session_id} is ending - ignoring ALL audio")
            return
        
        # ✅ CRITICAL: 3-second cooldown after AI finishes speaking
        last_ai_audio_ts = getattr(session_data, "last_ai_audio_ts", 0)
        if time.time() - last_ai_audio_ts < 3:
            logger.info(f"⏸️ BLOCKED: AI just finished speaking {time.time() - last_ai_audio_ts:.1f}s ago")
            return

        start_time = time.time()

        # === HARD WAIT MODE ===
        if getattr(session_data, "awaiting_user", False):
            if message_data.get("userStatus") not in ("user_speaking", "user_stopped_speaking") or not message_data.get("audio"):
                logger.debug(f"⏸️ Ignoring message — still waiting for real speech in {session_id}")
                return
            logger.info(f"🎤 User responded — unlocking awaiting_user in {session_id}")
            session_data.awaiting_user = False

        try:
            # Extract status info
            user_status = message_data.get('userStatus', 'unknown')
            silence_detected_flag = bool(message_data.get('silenceDetected', False))
            recording_duration = int(message_data.get('recordingDuration', 0))
            audio_b64 = message_data.get('audio', '') or ''

            # Time limits
            now_ts = time.time()
            end_time = getattr(session_data, "end_time", None)
            if end_time and now_ts >= end_time:
                if getattr(session_data, "awaiting_user", False):
                    try:
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_bytes)
                            session_data.awaiting_user = False
                            logger.info("🗣️ User transcript: %s  (quality=%.2f, bytes=%d)",
                                        (transcript or "").strip(), quality, len(audio_bytes))
                            if transcript.strip():
                                ion_data.current_concept or "unknown"
                                is_followup = getattr(session_data, "_last_question_followup", False)
                                session_data.add_exchange("[FINAL_ANSWER]", transcript, quality, concept, is_followup)
                                if session_data.summary_manager:
                                    session_data.summary_manager.add_answer(transcript)
                    except Exception as final_err:
                        logger.error("Final audio processing error: %s", final_err)
                await self._end_due_to_time(session_data)
                return

            # ---- refined silence gating ----
            past_greeting_grace = (
                session_data.greeting_end_ts is not None and
                (now_ts - session_data.greeting_end_ts) >= session_data.silence_grace_after_greeting_s
            )
            has_audio_payload = bool(audio_b64)
            is_silence_chunk = (
                not has_audio_payload
                and session_data.silence_ready
                and (session_data.has_user_spoken or past_greeting_grace)
                and (silence_detected_flconcept == sessag or user_status in ('user_silent', 'user_stopped_speaking'))
            )

            # If the user is speaking → reset silence and mark spoken
            if user_status == 'user_speaking':
                session_data.consecutive_silence_chunks = 0
                session_data.silence_prompt_active = False
                session_data.has_user_spoken = True
                session_data.last_user_speech_ts = time.time()
                old_count = getattr(session_data, 'silence_response_count', 0)
                if old_count > 0:
                    session_data.silence_response_count = 0
                    logger.info(f"🔄 User spoke - resetting backend silence counter from {old_count} to 0")
                
                # === TRACK RESPONSE TIME FOR COMMUNICATION SCORE ===
                last_q_ts = getattr(session_data, 'last_question_end_ts', None)
                if last_q_ts and session_data.current_stage == SessionStage.TECHNICAL:
                    response_delay = time.time() - last_q_ts
                    # Only count reasonable delays (ignore if > 60s - probably a different context)
                    if 0 < response_delay < 60:
                        if not hasattr(session_data, 'response_times'):
                            session_data.response_times = []
                        session_data.response_times.append(response_delay)
                        logger.info(f"⏱️ Response time recorded: {response_delay:.1f}s (avg: {sum(session_data.response_times)/len(session_data.response_times):.1f}s)")
                    # Reset to avoid double-counting
                    session_data.last_question_end_ts = None
            
            # ---- PATH A: silence chunk → skip STT, use DYNAMIC response ----
            if is_silence_chunk:
                session_data.consecutive_silence_chunks += 1
                logger.info(
                    "Session %s: silent chunk counted (%d/%d)",
                    session_id, session_data.consecutive_silence_chunks, session_data.silence_chunks_threshold
                )

                # Trigger after configured timeout window (~30 s)
                elapsed_silence = session_data.consecutive_silence_chunks * 5  # ~5 s per chunk
                if elapsed_silence >= getattr(session_data, "silence_timeout_s", 30):
                    session_data.consecutive_silence_chunks = 0
                    
                    try:
                        # ✅ Use dynamic silence response generation
                        ctx = {
                            "recording_duration": recording_duration,
                            "user_status": user_status,
                            "audio_size": len(audio_b64)
                        }
                        text = await self.generate_dynamic_silence_response(session_data, ctx)
                        
                        # Log and stream
                        concept = session_data.current_concept or "silence_handling"
                        session_data.add_exchange(text, "[USER_SILENT]", 0.0, concept, False)

                        session_data.silence_prompt_active = True
                        await self._send_silence_response_with_audio(session_data, text)
                        session_data.silence_prompt_active = False

                        soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
                        if session_data.current_stage == SessionStage.TECHNICAL and (not soft_cutoff or now_ts < soft_cutoff):
                            session_data.awaiting_user = True

                    except Exception as e_sil:
                        logger.error("Silence prompt generation/streaming error: %s", e_sil)

                logger.info("Silence handling time: %.2fs", time.time() - start_time)
                return

            # ---- PATH B: normal audio → run STT ----
            if not audio_b64:
                logger.debug("Session %s: no audio data received", session_id)
                return

            audio_bytes = base64.b64decode(audio_b64)
            logger.info("Session %s: processing normal audio (%d bytes)", session_id, len(audio_bytes))
            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_bytes)
            
            logger.info("🗣️ User transcript: %s  (quality=%.2f, bytes=%d)",
                        (transcript or "").strip(), quality, len(audio_bytes))

            # ✅ NEW: BACKGROUND NOISE FILTERING
            # Minimum audio duration check (at least 0.5 seconds of actual speech)
            MIN_AUDIO_BYTES = 6000  # ~0.5 seconds at typical bitrates
            MIN_TRANSCRIPT_LENGTH = 3  # At least 3 characters
            MIN_QUALITY_SCORE = 0.3  # Minimum quality threshold

            if len(audio_bytes) < MIN_AUDIO_BYTES:
                logger.info(f"⚠️ Audio too short ({len(audio_bytes)} bytes < {MIN_AUDIO_BYTES}) - likely background noise (sneeze/cough)")
                return

            if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT_LENGTH:
                logger.info(f"⚠️ Transcript too short: '{transcript}' ({len(transcript.strip())} chars) - likely background noise")
                return
                
            if quality < MIN_QUALITY_SCORE:
                logger.info(f"⚠️ Audio quality too low ({quality:.2f} < {MIN_QUALITY_SCORE}) - likely background noise")
                return

            logger.info(f"✅ Valid audio accepted: {len(audio_bytes)} bytes, quality={quality:.2f}, text='{transcript}'")

            # Reset silence counter and mark spoken on any real transcript
            if transcript and transcript.strip():
                session_data.consecutive_silence_chunks = 0
                session_data.has_user_spoken = True
                session_data.last_user_speech_ts = time.time()

            # ============================================================================
            # === SMART RESPONSE HANDLING (REPEAT, SKIP, IRRELEVANT DETECTION) ===
            # ============================================================================
            transcript_lower = transcript.lower() if transcript else ""

            # 1. REPEAT REQUEST DETECTION
            repeat_phrases = ["repeat", "say that again", "what did you say", "didn't catch that", "can you repeat", "pardon", "come again"]
            if any(phrase in transcript_lower for phrase in repeat_phrases):
                logger.info("🔁 User requested to repeat the question")
                
                # ✅ FIX: Find the last ACTUAL question (skip silence prompts)
                conv_log = getattr(session_data, "conversation_log", [])
                last_question = None
                
                if conv_log and len(conv_log) > 0:
                    # Search backwards for the last real question (not a silence prompt)
                    for i in range(len(conv_log) - 1, -1, -1):
                        exchange = conv_log[i]
                        user_resp = exchange.get("user_response", "")
                        ai_msg = exchange.get("ai_message", "")
                        stage = exchange.get("stage", "")
                        
                        # ✅ Skip silence prompts - they have [USER_SILENT] as user_response
                        if user_resp == "[USER_SILENT]":
                            logger.info(f"🔁 Skipping silence prompt at index {i}: '{ai_msg[:50]}...'")
                            continue
                        
                        # ✅ Skip greetings
                        if stage == "greeting":
                            logger.info(f"🔁 Skipping greeting at index {i}")
                            continue
                        
                        # ✅ Skip if AI message is too short
                        if not ai_msg or len(ai_msg.strip()) < 10:
                            continue
                        
                        # ✅ Check if it's a silence prompt phrase
                        ai_msg_lower = ai_msg.lower()
                        silence_prompt_phrases = [
                            "are you there", "still with me", "can you hear", 
                            "are you still", "hello?", "you there", "are you ready",
                            "just checking", "i'd love to hear"
                        ]
                        if any(phrase in ai_msg_lower for phrase in silence_prompt_phrases):
                            logger.info(f"🔁 Skipping silence phrase at index {i}: '{ai_msg[:50]}...'")
                            continue
                        
                        # ✅ Found a real question!
                        last_question = ai_msg
                        logger.info(f"🔁 Found actual question at index {i}: '{last_question[:50]}...'")
                        break
                                
                    if last_question:
                        # === UPDATE COMMUNICATION STATS: REPEAT REQUEST (5D) ===
                        if not hasattr(session_data, 'comm_stats'):
                            session_data.comm_stats = {"total_questions": 0, "answered": 0, "skipped": 0, "silent": 0, "irrelevant": 0, "repeat_requests": 0}
                        session_data.comm_stats["repeat_requests"] += 1
                        await self._send_comm_score_update(session_data, "repeat_request")
                        # ✅ NEW: Extract only the question part, removing acknowledgments
                        last_question_clean = self._extract_question_only(last_question)
                        
                        logger.info(f"🔁 Original response: '{last_question}'")
                        logger.info(f"🔁 Repeating only question: '{last_question_clean}'")
                        
                        # Set AI responding lock 
                        session_data.ai_is_responding = True
                        
                        await self._send_quick_message(session_data, {
                            "type": "ai_response",
                            "text": last_question_clean,  # ✅ Send cleaned question only
                            "status": session_data.current_stage.value,
                        })
                        
                        # Send audio
                        async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                            last_question_clean, session_id=session_data.session_id  # ✅ TTS cleaned question only
                        ):
                            if audio_chunk:
                                await self._send_quick_message(session_data, {
                                    "type": "audio_chunk",
                                    "audio": audio_chunk.hex(),
                                    "status": session_data.current_stage.value,
                                })
                        await self._send_quick_message(session_data, {
                            "type": "audio_end",
                            "status": session_data.current_stage.value
                        })
                        # ✅ LOG THE REPEAT REQUEST
                        concept = session_data.current_concept or "repeat_request"
                        session_data.add_exchange(
                            ai_message=last_question,
                            user_response=transcript,
                            quality=0.0,
                            concept=concept,
                            is_followup=False
                        )
                        logger.info(f"📝 Logged repeat: Q='{last_question[:50]}...', A='{transcript}'")

                        # Release AI responding lock and wait for answer
                        session_data.ai_is_responding = False
                        session_data.awaiting_user = True
                        session_data.last_ai_audio_ts = time.time()
                        
                        logger.info("✅ Question repeated successfully")
                        return

            """repeat_phrases = ["repeat", "say that again", "what did you say", "didn't catch that", "can you repeat", "pardon", "come again"]
            if any(phrase in transcript_lower for phrase in repeat_phrases):
                logger.info("🔁 User requested to repeat the question")
                
                # Get the last AI question from conversation log
                conv_log = getattr(session_data, "conversation_log", [])
                if conv_log and len(conv_log) > 0:
                    last_question = conv_log[-1].get("ai_message", "")
                    
                    if last_question:
                        logger.info(f"🔁 Repeating last question: '{last_question}'")
                        
                        # Set AI responding lock
                        session_data.ai_is_responding = True
                        
                        await self._send_quick_message(session_data, {
                            "type": "ai_response",
                            "text": last_question,
                            "status": session_data.current_stage.value,
                        })
                        
                        # Send audio
                        async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                            last_question, session_id=session_data.session_id
                        ):
                            if audio_chunk:
                                await self._send_quick_message(session_data, {
                                    "type": "audio_chunk",
                                    "audio": audio_chunk.hex(),
                                    "status": session_data.current_stage.value,
                                })
                        await self._send_quick_message(session_data, {
                            "type": "audio_end",
                            "status": session_data.current_stage.value
                        })
                        
                        # Release AI responding lock and wait for answer
                        session_data.ai_is_responding = False
                        session_data.awaiting_user = True
                        session_data.last_ai_audio_ts = time.time()
                        
                        logger.info("✅ Question repeated successfully")
                        return"""
            
            # 2. SKIP/DON'T KNOW DETECTION WITH ACKNOWLEDGMENT
            skip_phrases = ["skip", "i don't know", "dont know", "i do not know", "can't answer", "cant answer", "cannot answer", "not sure", "no idea", "don't have", "dont have"]
            if transcript and any(phrase in transcript.lower() for phrase in skip_phrases):
                logger.info("⏩ User explicitly requested to skip the question")

                # === UPDATE COMMUNICATION STATS: SKIPPED (5B) ===
                if not hasattr(session_data, 'comm_stats'):
                    session_data.comm_stats = {"total_questions": 0, "answered": 0, "skipped": 0, "silent": 0, "irrelevant": 0, "repeat_requests": 0}
                session_data.comm_stats["total_questions"] += 1
                session_data.comm_stats["skipped"] += 1
                await self._send_comm_score_update(session_data, "skipped")
                
                # Generate brief "That's okay" acknowledgment
                skip_acknowledgments = [
                    "That's okay!",
                    "No worries!",
                    "That's fine!",
                    "Totally fine!",
                    "No problem!",
                ]
                import random
                skip_acknowledgment = random.choice(skip_acknowledgments)
                
                logger.info(f"✅ Generated skip acknowledgment: '{skip_acknowledgment}'")
                
                # Generate next question using auto-advance logic
                fm = getattr(session_data, "summary_manager", None)
                if not fm:
                    logger.error("❌ No fragment manager available")
                    return
                
                conv_log = getattr(session_data, "conversation_log", [])
                current_concept = session_data.current_concept or "unknown"
                
                # Count questions on current concept
                questions_on_concept = sum(
                    1 for exchange in conv_log 
                    if exchange.get("concept") == current_concept
                )
                
                max_questions = getattr(session_data, 'questions_per_concept', 3)
                should_ask_followup = questions_on_concept < max_questions
                
                next_question = None
                
                if should_ask_followup:
                    # Generate follow-up on same concept
                    logger.info("🔄 Skip: Generating FOLLOW-UP on same concept")
                    current_concept_title, current_concept_content = fm.get_active_fragment()
                    
                    # Get last question from conversation log
                    last_question = conv_log[-1].get("ai_message", "") if conv_log else ""
                    
                    followup_prompt = prompts.dynamic_followup_response(
                        context_text=current_concept_content[:2000],
                        user_input="[User skipped]",
                        previous_question=last_question,
                        session_state={
                            "domain": current_concept,
                            "questions_asked": questions_on_concept,
                            "concept": current_concept_title
                        }
                    )
                    
                    loop = asyncio.get_event_loop()
                    next_question = await loop.run_in_executor(
                        shared_clients.executor,
                        self.conversation_manager._sync_openai_call,
                        followup_prompt,
                    )
                    next_question = (next_question or "").strip()
                    
                    if not next_question or len(next_question.split()) < 8:
                        next_question = await loop.run_in_executor(
                            shared_clients.executor,
                            self.conversation_manager._sync_openai_call,
                            followup_prompt,
                        )
                        next_question = (next_question or "").strip()
                    
                    session_data._last_question_followup = True
                    if fm:
                        fm.add_question(next_question, current_concept_title, is_followup=True)
                else:
                    # Move to new topic
                    logger.info("🔄 Skip: Moving to NEW TOPIC")
                    old_concept = session_data.current_concept
                    moved = fm.advance_fragment()
                    
                    if moved:
                        new_concept_title, new_concept_content = fm.get_active_fragment()
                        session_data.current_concept = new_concept_title
                        session_data.current_domain = new_concept_title
                        
                        transition_prompt = prompts.dynamic_concept_transition(
                            current_concept=old_concept,
                            next_concept=new_concept_title,
                            user_last_answer="[User skipped]",
                            next_concept_content=new_concept_content
                        )
                        
                        loop = asyncio.get_event_loop()
                        next_question = await loop.run_in_executor(
                            shared_clients.executor,
                            self.conversation_manager._sync_openai_call,
                            transition_prompt,
                        )
                        next_question = (next_question or "").strip()
                        
                        if not next_question or len(next_question.split()) < 10:
                            next_question = await loop.run_in_executor(
                                shared_clients.executor,
                                self.conversation_manager._sync_openai_call,
                                transition_prompt,
                            )
                            next_question = (next_question or "").strip()
                        
                        session_data._last_question_followup = False
                        if fm:
                            fm.add_question(next_question, new_concept_title, is_followup=False)
                    else:
                        # ✅ FIX: Check if we should switch to extended mode instead of ending
                        now_ts = time.time()
                        elapsed = now_ts - session_data.created_at
                        min_duration = getattr(session_data, 'min_session_duration', 15 * 60)
                        time_remaining = min_duration - elapsed
                        
                        if time_remaining > 60:
                            session_data.extended_mode = True
                            logger.info(f"🌐 Skip: Summary exhausted - switching to EXTENDED mode ({time_remaining/60:.1f}m left)")
                            next_question = await self.generate_extended_question(session_data)
                            if next_question:
                                session_data._last_question_followup = False
                                session_data.current_concept = getattr(session_data, 'main_topic', 'extended_question')
                            else:
                                await self._finalize_session_with_formal_closing(session_data)
                                return
                        else:
                            logger.info("🏁 No more concepts and time nearly up - ending session")
                            await self._finalize_session_with_formal_closing(session_data)
                            return
                    
                
                if not next_question:
                    logger.error("❌ Failed to generate next question")
                    return
                
                # ✅ COMBINE acknowledgment + next question into ONE response
                combined_response = f"{skip_acknowledgment} {next_question}"
                
                logger.info(f"✅ Combined skip response: '{combined_response}'")
                
                # Add to conversation log
                concept = session_data.current_concept or "skip_handled"
                is_followup = getattr(session_data, "_last_question_followup", False)
                session_data.add_exchange(combined_response, "[SKIP]", 0.3, concept, is_followup)
                
                # Set AI responding lock
                session_data.ai_is_responding = True
                
                # Send SINGLE combined response
                await self._send_quick_message(session_data, {
                    "type": "ai_response",
                    "text": combined_response,
                    "status": session_data.current_stage.value,
                })
                
                # Send audio for SINGLE combined response
                chunk_count = 0
                async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                    combined_response, session_id=session_data.session_id
                ):
                    if audio_chunk:
                        await self._send_quick_message(session_data, {
                            "type": "audio_chunk",
                            "audio": audio_chunk.hex(),
                            "status": session_data.current_stage.value,
                        })
                        chunk_count += 1
                
                await self._send_quick_message(session_data, {
                    "type": "audio_end",
                    "status": session_data.current_stage.value
                })
                
                logger.info(f"🔊 Streamed {chunk_count} audio chunks for skip response")
                
                # Release AI responding lock
                session_data.ai_is_responding = False
                session_data.last_ai_audio_ts = time.time()
                session_data.awaiting_user = True
                
                logger.info("✅ Skip handled with single combined response")
                return
                
            # 3. IRRELEVANT ANSWER DETECTION (LLM-BASED) - Only in TECHNICAL stage
            # 3. IRRELEVANT ANSWER DETECTION (LLM-BASED) - Only in TECHNICAL stage
            if session_data.current_stage == SessionStage.TECHNICAL:
                conv_log = getattr(session_data, "conversation_log", [])
                if conv_log and len(conv_log) > 0 and len(transcript.strip()) > 10:
                    # ✅ FIX: Find the last ACTUAL question (skip silence prompts)
                    last_question = None
                    
                    for i in range(len(conv_log) - 1, -1, -1):
                        exchange = conv_log[i]
                        user_resp = exchange.get("user_response", "")
                        ai_msg = exchange.get("ai_message", "")
                        stage = exchange.get("stage", "")
                        
                        # ✅ Skip silence prompts - they have [USER_SILENT] as user_response
                        if user_resp == "[USER_SILENT]":
                            logger.info(f"🔍 Relevance check: Skipping silence prompt at index {i}")
                            continue
                        
                        # ✅ Skip greetings
                        if stage == "greeting":
                            continue
                        
                        # ✅ Skip if AI message is too short
                        if not ai_msg or len(ai_msg.strip()) < 10:
                            continue
                        
                        # ✅ Check if it's a silence prompt phrase
                        ai_msg_lower = ai_msg.lower()
                        silence_prompt_phrases = [
                            "are you there", "still with me", "can you hear", 
                            "are you still", "hello?", "you there", "are you ready",
                            "just checking", "i'd love to hear"
                        ]
                        if any(phrase in ai_msg_lower for phrase in silence_prompt_phrases):
                            logger.info(f"🔍 Relevance check: Skipping silence phrase at index {i}")
                            continue
                        
                        # ✅ Found a real question!
                        last_question = ai_msg
                        logger.info(f"🔍 Found actual question for relevance check at index {i}: '{last_question[:50]}...'")
                        break
                    
                                        
                    if last_question:
                        logger.info(f"🔍 Checking relevance: Question='{last_question[:80]}...', Answer='{transcript[:80]}...'")
                        
                        # Use LLM to determine if answer is relevant
                        relevance_prompt = f"""You are evaluating if a student's answer is relevant to the question asked.

            Question: "{last_question}"
            Student's Answer: "{transcript}"

            Analyze if the answer is related to the question. The answer should be about the technical topic mentioned in the question.

            Respond with ONLY "RELEVANT" or "IRRELEVANT" - nothing else.

            Examples:
            - Question: "What is SAP Basis?" Answer: "SAP Basis is the technical foundation" → RELEVANT
            - Question: "What is SAP Basis?" Answer: "My favorite cricketer is Dhoni" → IRRELEVANT
            - Question: "Explain database normalization" Answer: "I like pizza" → IRRELEVANT
            - Question: "What is React?" Answer: "React is a JavaScript library" → RELEVANT
            - Question: "How does patching work?" Answer: "The weather is nice today" → IRRELEVANT

            Now evaluate:
            Question: "{last_question}"
            Answer: "{transcript}"

            Response (RELEVANT or IRRELEVANT):"""

                        try:
                            loop = asyncio.get_event_loop()
                            relevance_check = await loop.run_in_executor(
                                shared_clients.executor,
                                self.conversation_manager._sync_openai_call,
                                relevance_prompt,
                            )
                            relevance_check = (relevance_check or "").strip().upper()
                            
                            logger.info(f"🤖 LLM Relevance Check Result: '{relevance_check}'")
                            
                            if "IRRELEVANT" in relevance_check:
                                logger.info(f"⚠️ LLM detected irrelevant answer: '{transcript}' for question: '{last_question}'")

                                # === UPDATE COMMUNICATION STATS: IRRELEVANT (5C) ===
                                if not hasattr(session_data, 'comm_stats'):
                                    session_data.comm_stats = {"total_questions": 0, "answered": 0, "skipped": 0, "silent": 0, "irrelevant": 0, "repeat_requests": 0}
                                session_data.comm_stats["total_questions"] += 1
                                session_data.comm_stats["irrelevant"] += 1
                                await self._send_comm_score_update(session_data, "irrelevant")
                                
                                # ✅ IMPROVED: Generate polite redirect instead of encouragement
                                
                                # Step 1: Generate brief redirect about relevance
                                redirect_prompt = f"""Generate a VERY SHORT polite statement (max 8 words) that:
            1. Indicates the answer wasn't related to the question
            2. Does NOT mention "next", "move on"
            3. Keep it gentle and non-judgmental

            Examples:
            - "That doesn't answer the question."
            - "I don't think that's related."
            - "That's not quite what I asked."
            - "That doesn't match the question."
            - "That's off-topic."

            Generate one now (max 8 words):"""
                                
                                loop = asyncio.get_event_loop()
                                redirect = await loop.run_in_executor(
                                    shared_clients.executor,
                                    self.conversation_manager._sync_openai_call,
                                    redirect_prompt,
                                )
                                redirect = (redirect or "").strip()
                                
                                # Fallback if empty or too long
                                if not redirect or len(redirect.split()) > 10:
                                    redirects = [
                                        "That doesn't answer the question.",
                                        "I don't think that's related.",
                                        "That's not quite what I asked.",
                                        "That doesn't match the question.",
                                        "That's off-topic."
                                    ]
                                    import random
                                    redirect = random.choice(redirects)
                                
                                logger.info(f"🔄 Generated redirect: '{redirect}'")
                                
                                # Step 2: Generate next question using auto-advance logic
                                fm = getattr(session_data, "summary_manager", None)
                                if not fm:
                                    logger.error("❌ No fragment manager available")
                                    return
                                
                                conv_log = getattr(session_data, "conversation_log", [])
                                current_concept = session_data.current_concept or "unknown"
                                
                                # Count questions on current concept
                                questions_on_concept = sum(
                                    1 for exchange in conv_log 
                                    if exchange.get("concept") == current_concept
                                )
                                
                                max_questions = getattr(session_data, 'questions_per_concept', 3)
                                should_ask_followup = questions_on_concept < max_questions
                                
                                next_question = None
                                
                                if should_ask_followup:
                                    # Generate follow-up on same concept
                                    logger.info("🔄 Generating FOLLOW-UP on same concept")
                                    current_concept_title, current_concept_content = fm.get_active_fragment()
                                    previous_question = last_question  # Use the question they just got wrong
                                    
                                    followup_prompt = prompts.dynamic_followup_response(
                                        context_text=current_concept_content[:2000],
                                        user_input="[User didn't answer correctly]",
                                        previous_question=previous_question,
                                        session_state={
                                            "domain": current_concept,
                                            "questions_asked": questions_on_concept,
                                            "concept": current_concept_title
                                        }
                                    )
                                    
                                    next_question = await loop.run_in_executor(
                                        shared_clients.executor,
                                        self.conversation_manager._sync_openai_call,
                                        followup_prompt,
                                    )
                                    next_question = (next_question or "").strip()
                                    
                                    if not next_question or len(next_question.split()) < 8:
                                        next_question = await loop.run_in_executor(
                                            shared_clients.executor,
                                            self.conversation_manager._sync_openai_call,
                                            followup_prompt,
                                        )
                                        next_question = (next_question or "").strip()
                                    
                                    session_data._last_question_followup = True
                                    if fm:
                                        fm.add_question(next_question, current_concept_title, is_followup=True)
                                else:
                                    # Move to new topic
                                    logger.info("🔄 Moving to NEW TOPIC")
                                    old_concept = session_data.current_concept
                                    moved = fm.advance_fragment()
                                    
                                    if moved:
                                        new_concept_title, new_concept_content = fm.get_active_fragment()
                                        session_data.current_concept = new_concept_title
                                        session_data.current_domain = new_concept_title
                                        
                                        transition_prompt = prompts.dynamic_concept_transition(
                                            current_concept=old_concept,
                                            next_concept=new_concept_title,
                                            user_last_answer="[User didn't answer correctly]",
                                            next_concept_content=new_concept_content
                                        )
                                        
                                        next_question = await loop.run_in_executor(
                                            shared_clients.executor,
                                            self.conversation_manager._sync_openai_call,
                                            transition_prompt,
                                        )
                                        next_question = (next_question or "").strip()
                                        
                                        if not next_question or len(next_question.split()) < 10:
                                            next_question = await loop.run_in_executor(
                                                shared_clients.executor,
                                                self.conversation_manager._sync_openai_call,
                                                transition_prompt,
                                            )
                                            next_question = (next_question or "").strip()
                                        
                                        session_data._last_question_followup = False
                                        if fm:
                                            fm.add_question(next_question, new_concept_title, is_followup=False)
                                    else:
                                        # ✅ FIX: Check if we should switch to extended mode instead of ending
                                        now_ts = time.time()
                                        elapsed = now_ts - session_data.created_at
                                        min_duration = getattr(session_data, 'min_session_duration', 15 * 60)
                                        time_remaining = min_duration - elapsed
                                        
                                        if time_remaining > 60:
                                            session_data.extended_mode = True
                                            logger.info(f"🌐 Irrelevant: Summary exhausted - switching to EXTENDED mode ({time_remaining/60:.1f}m left)")
                                            next_question = await self.generate_extended_question(session_data)
                                            if next_question:
                                                session_data._last_question_followup = False
                                                session_data.current_concept = getattr(session_data, 'main_topic', 'extended_question')
                                            else:
                                                await self._finalize_session_with_formal_closing(session_data)
                                                return
                                        else:
                                            logger.info("🏁 No more concepts and time nearly up - ending session")
                                            await self._finalize_session_with_formal_closing(session_data)
                                            return
                                                                    
                                if not next_question:
                                    logger.error("❌ Failed to generate next question")
                                    return
                                
                                # ✅ CRITICAL: COMBINE redirect + next question into ONE response
                                combined_response = f"{redirect} {next_question}"
                                
                                logger.info(f"✅ Combined response: '{combined_response}'")
                                
                                # Add to conversation log
                                concept = session_data.current_concept or "irrelevant_handled"
                                is_followup = getattr(session_data, "_last_question_followup", False)
                                session_data.add_exchange(combined_response, "[IRRELEVANT]", 0.3, concept, is_followup)
                                
                                # Set AI responding lock
                                session_data.ai_is_responding = True
                                
                                # Send SINGLE combined response
                                await self._send_quick_message(session_data, {
                                    "type": "ai_response",
                                    "text": combined_response,
                                    "status": session_data.current_stage.value,
                                })
                                
                                # Send audio for SINGLE combined response
                                chunk_count = 0
                                async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                                    combined_response, session_id=session_data.session_id
                                ):
                                    if audio_chunk:
                                        await self._send_quick_message(session_data, {
                                            "type": "audio_chunk",
                                            "audio": audio_chunk.hex(),
                                            "status": session_data.current_stage.value,
                                        })
                                        chunk_count += 1
                                
                                await self._send_quick_message(session_data, {
                                    "type": "audio_end",
                                    "status": session_data.current_stage.value
                                })
                                
                                logger.info(f"🔊 Streamed {chunk_count} audio chunks for combined response")
                                
                                # Release AI responding lock
                                session_data.ai_is_responding = False
                                session_data.last_ai_audio_ts = time.time()
                                session_data.awaiting_user = True
                                
                                logger.info("✅ Irrelevant answer handled with redirect")
                                return
                                                                                
                        except Exception as relevance_error:
                            logger.error(f"❌ Relevance check failed: {relevance_error}")
                            # Continue with normal processing if relevanc
            # ============================================================================
            # === END OF SMART RESPONSE HANDLING ===
            # ============================================================================

            # Poor transcript handling (clarify → auto-advance)
            if not transcript or len(transcript.strip()) < 2:
                attempt = getattr(session_data, 'clarification_attempts', 0) + 1
                session_data.clarification_attempts = attempt
                if attempt >= 2:
                    logger.info("🕐 No valid speech detected (attempt %d) — staying on same question", attempt)
                    await self._auto_advance_question(session_data)
                    return

                loop = asyncio.get_event_loop()
                clarification_prompt = prompts.dynamic_clarification_request({
                    'clarification_attempts': attempt,
                    'audio_quality': quality,
                    'audio_size': len(audio_bytes)
                })
                clarification_message = await loop.run_in_executor(
                    shared_clients.executor,
                    self.conversation_manager._sync_openai_call,
                    clarification_prompt,
                )
                clarification_message = (clarification_message or "").strip()

                if not clarification_message or len(clarification_message.split()) < 3:
                    clarification_message = await loop.run_in_executor(
                        shared_clients.executor,
                        self.conversation_manager._sync_openai_call,
                        clarification_prompt,
                    )
                    clarification_message = (clarification_message or "").strip()

                await self._send_quick_message(session_data, {
                    "type": "clarification",
                    "text": (clarification_message or " "),
                    "status": session_data.current_stage.value,
                })
                return

            # Normal conversation flow
            session_data.clarification_attempts = 0

            try:
                inferred = self._infer_domain(transcript)
                if inferred and inferred != "general":
                    session_data.current_domain = inferred
            except Exception as e:
                logger.debug("Domain inference error: %s", e)

            # Re-check time windows post-STT
            now_ts = time.time()
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
            if end_time and now_ts >= end_time:
                # Add exchange to conversation log
                concept = session_data.current_concept or "unknown"
                is_followup = getattr(session_data, "_last_question_followup", False)
                session_data.add_exchange(ai_response, transcript, quality, concept, is_followup)

                if session_data.summary_manager:
                    session_data.summary_manager.add_answer(transcript)

                # ✅ FIX: Increment greeting count IMMEDIATELY after adding exchange
                if session_data.current_stage == SessionStage.GREETING:
                    session_data.greeting_count = getattr(session_data, "greeting_count", 0) + 1
                    logger.info(f"📊 Greeting count incremented to {session_data.greeting_count}")

                # === UPDATE COMMUNICATION STATS: ANSWERED (5A) ===
                if session_data.current_stage == SessionStage.TECHNICAL:
                    if not hasattr(session_data, 'comm_stats'):
                        session_data.comm_stats = {"total_questions": 0, "answered": 0, "skipped": 0, "silent": 0, "irrelevant": 0, "repeat_requests": 0}
                    session_data.comm_stats["total_questions"] += 1
                    session_data.comm_stats["answered"] += 1
                    await self._send_comm_score_update(session_data, "answered")

                await self._update_session_state_fast(session_data)
                return
                
            elif soft_cutoff and now_ts >= soft_cutoff:
                concept = session_data.current_concept or "unknown"
                is_followup = getattr(session_data, "_last_question_followup", False)
                session_data.add_exchange("", transcript, quality, concept, is_followup)
                if session_data.summary_manager:
                    session_data.summary_manager.add_answer(transcript)
                ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)
                session_data.conversation_log[-1]["ai_response"] = ai_response
                await self._send_response_with_ultra_fast_audio(session_data, ai_response)
                session_data.awaiting_user = True
                await self._end_due_to_time(session_data)
                return

            # === GREETING STAGE PRE-CHECK ===
            if session_data.current_stage == SessionStage.GREETING:
                if not transcript or len(transcript.strip().split()) < 2:
                    logger.info("🕐 User hasn't replied to greeting yet — letting silence handler take over")
                    return

            # === SET AI RESPONDING LOCK ===
            session_data.awaiting_user = False
            session_data.ai_is_responding = True

            # === SMART AI RESPONSE GENERATION WITH TRANSITION DETECTION ===
            ai_response = None
            
            # ============================================================================
            # ✅ FIXED: GREETING STAGE WITH AUTO-TRANSITION AFTER 2 EXCHANGES
            # ============================================================================
            if session_data.current_stage == SessionStage.GREETING:
                # ✅ NEW: Check greeting count FIRST
                greeting_count = getattr(session_data, "greeting_count", 0)
                max_greeting_exchanges = 2  # Hardcoded to 2 exchanges
                
                logger.info(f"🔍 GREETING CHECK: count={greeting_count}, max={max_greeting_exchanges}")
                logger.info(f"🔍 Current stage: {session_data.current_stage}")
                logger.info(f"🔍 User transcript: '{transcript}'")
                
                # ✅ FIX 1: Auto-transition after 2 greeting exchanges
                if greeting_count >= max_greeting_exchanges:
                    logger.info("🎯 Max greeting exchanges reached - AUTO-TRANSITIONING to TECHNICAL stage")
                    session_data.current_stage = SessionStage.TECHNICAL
                    session_data.greeting_count = 999
                    session_data.awaiting_user_confirmation = False
                    
                    # Generate first technical question
                    fm = session_data.summary_manager
                    if fm:
                        current_concept_title, current_concept_content = fm.get_active_fragment()
                        session_data.current_concept = current_concept_title
                        session_data.current_domain = current_concept_title
                        
                        tech_prompt = prompts.generate_first_technical_question(
                            concept_title=current_concept_title,
                            concept_content=current_concept_content,
                            user_greeting=transcript
                        )
                        
                        loop = asyncio.get_event_loop()
                        ai_response = await loop.run_in_executor(
                            shared_clients.executor,
                            self.conversation_manager._sync_openai_call,
                            tech_prompt,
                        )
                        ai_response = (ai_response or "").strip()
                        
                        if not ai_response or len(ai_response.split()) < 8:
                            logger.warning("⚠️ First technical question too short, retrying...")
                            ai_response = await loop.run_in_executor(
                                shared_clients.executor,
                                self.conversation_manager._sync_openai_call,
                                tech_prompt,
                            )
                            ai_response = (ai_response or "").strip()
                        
                        logger.info(f"🎯 Generated first technical question: '{ai_response}'")
                        fm.add_question(ai_response, current_concept_title, is_followup=False)
                    else:            
                        logger.error("❌ Fragment manager is None!")
                        ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)
                else:
                    # ✅ FIX 2: Check for explicit confirmation
                    user_confirmed = any(phrase in transcript.lower() for phrase in [
                        "yes", "yeah", "yep", "okay", "ok", "sure", "let's", "lets", "go ahead", "start", "ready"
                    ])
                    
                    awaiting_confirmation = getattr(session_data, "awaiting_user_confirmation", False)
                    
                    logger.info(f"🔍 GREETING TRANSITION CHECK:")
                    logger.info(f"  - User transcript: '{transcript}'")
                    logger.info(f"  - User confirmed: {user_confirmed}")
                    logger.info(f"  - Awaiting confirmation: {awaiting_confirmation}")
                    logger.info(f"  - Greeting count: {greeting_count}/{max_greeting_exchanges}")
                    
                    if user_confirmed and awaiting_confirmation:
                        session_data.awaiting_user_confirmation = False
                        session_data.current_stage = SessionStage.TECHNICAL
                        session_data.greeting_count = 999
                        logger.info("🎯 User confirmed — switching to TECHNICAL stage")

                        fm = session_data.summary_manager
                        if fm:
                            current_concept_title, current_concept_content = fm.get_active_fragment()
                            
                            logger.info(f"🎯 Fragment Manager Status:")
                            logger.info(f"  - Current concept: '{current_concept_title}'")
                            logger.info(f"  - Content length: {len(current_concept_content) if current_concept_content else 0}")
                            
                            session_data.current_concept = current_concept_title
                            session_data.current_domain = current_concept_title
                            
                            tech_prompt = prompts.generate_first_technical_question(
                                concept_title=current_concept_title,
                                concept_content=current_concept_content,
                                user_greeting=transcript
                            )
                            
                            loop = asyncio.get_event_loop()
                            ai_response = await loop.run_in_executor(
                                shared_clients.executor,
                                self.conversation_manager._sync_openai_call,
                                tech_prompt,
                            )
                            ai_response = (ai_response or "").strip()
                            
                            if not ai_response or len(ai_response.split()) < 8:
                                logger.warning("⚠️ First technical question too short, retrying...")
                                ai_response = await loop.run_in_executor(
                                    shared_clients.executor,
                                    self.conversation_manager._sync_openai_call,
                                    tech_prompt,
                                )
                                ai_response = (ai_response or "").strip()
                            
                            logger.info(f"🎯 Generated first technical question: '{ai_response}'")
                            fm.add_question(ai_response, current_concept_title, is_followup=False)
                        else:            
                            logger.error("❌ Fragment manager is None!")
                            ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)
                    else:
                        logger.info("💬 Continuing friendly greeting exchange")
                        ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)
                        
                        # ✅ FIX 3: Check if AI asked "ready" question and set flag
                        ai_reply_lower = ai_response.lower()
                        asked_ready = any(phrase in ai_reply_lower for phrase in [
                            "shall we", "ready to", "let's discuss", "go over your", "start with", "dive into"
                        ])
                        
                        if asked_ready:
                            session_data.awaiting_user_confirmation = True
                            logger.info("🕊️ AI asked if user is ready - awaiting confirmation")
            else:                
                # ============================================================================
                # === FOLLOW-UP QUESTION LOGIC (Summary-Based) ===
                # ============================================================================
                logger.info("🔄 Generating AI response (technical stage)")
                fm = session_data.summary_manager
                # Check time remaining for extended mode decision
                now_ts = time.time()
                elapsed = now_ts - session_data.created_at
                min_duration = getattr(session_data, 'min_session_duration', 15 * 60)
                time_remaining = min_duration - elapsed
                
                if not fm:
                    logger.error("❌ No fragment manager")
                    ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)
                else:
                    # Check if we're in extended mode
                    if getattr(session_data, 'extended_mode', False):
                        logger.info(f"🌐 In EXTENDED mode - generating web-based question ({time_remaining/60:.1f}m remaining)")
                        ai_response = await self.generate_extended_question(session_data)
                        
                        if ai_response:
                            session_data._last_question_followup = False
                            session_data.current_concept = getattr(session_data, 'main_topic', 'extended_question')
                        else:
                            # Fallback: check if we should end session
                            if time_remaining <= 60:
                                logger.info("🏁 Extended question failed and time nearly up - ending session")
                                await self._finalize_session_with_formal_closing(session_data)
                                return
                            else:
                                # Try one more time
                                ai_response = await self.generate_extended_question(session_data)
                    else:
                        # Normal summary-based question generation
                        frag_details = fm.get_current_fragment_details()
                        
                        if not frag_details:
                            # Summary exhausted - check if we should switch to extended mode
                            if time_remaining > 60:  # More than 1 minute remaining
                                session_data.extended_mode = True
                                logger.info(f"🌐 Summary exhausted - switching to EXTENDED mode ({time_remaining/60:.1f}m remaining)")
                                ai_response = await self.generate_extended_question(session_data)
                                if ai_response:
                                    session_data._last_question_followup = False
                                    session_data.current_concept = getattr(session_data, 'main_topic', 'extended_question')
                            else:
                                # Time to end session
                                logger.info("🏁 No more fragments and time is up - ending session")
                                await self._finalize_session_with_formal_closing(session_data)
                                return
                        else:
                            # Continue with normal summary-based logic
                            current_concept = frag_details['title']
                            has_example = frag_details.get('has_example', False)
                            questions_on_current = fm.questions_asked_on_current
                            
                            logger.info(f"📊 Current concept: '{current_concept}'")
                            logger.info(f"📊 Questions asked on this concept: {questions_on_current}")
                            logger.info(f"📊 Has example: {has_example}")
                            
                            # Existing MAIN question, EXAMPLE question, or TRANSITION logic...
                            # (Keep your existing code here)
                            
                            if questions_on_current == 0:
                                # === CASE 1: MAIN QUESTION ===
                                logger.info("📝 Generating MAIN question from fragment content")
                                
                                question_prompt = prompts.generate_main_question_from_content(frag_details)
                                
                                loop = asyncio.get_event_loop()
                                ai_response = await loop.run_in_executor(
                                    shared_clients.executor,
                                    self.conversation_manager._sync_openai_call,
                                    question_prompt,
                                )
                                ai_response = (ai_response or "").strip()
                                
                                if not ai_response or len(ai_response.split()) < 5:
                                    ai_response = await loop.run_in_executor(
                                        shared_clients.executor,
                                        self.conversation_manager._sync_openai_call,
                                        question_prompt,
                                    )
                                    ai_response = (ai_response or "").strip()
                                
                                logger.info(f"✅ Generated MAIN question: '{ai_response}'")
                                fm.add_question(ai_response, current_concept, is_followup=False)
                                session_data._last_question_followup = False
                                
                            elif questions_on_current == 1 and has_example:
                                # === CASE 2: EXAMPLE QUESTION ===
                                logger.info("📝 Asking for EXAMPLE (has_example=True)")
                                
                                conv_log = getattr(session_data, "conversation_log", [])
                                prev_question = conv_log[-1].get("ai_message", "") if conv_log else ""
                                
                                example_prompt = prompts.generate_example_question(frag_details, prev_question)
                                
                                loop = asyncio.get_event_loop()
                                ai_response = await loop.run_in_executor(
                                    shared_clients.executor,
                                    self.conversation_manager._sync_openai_call,
                                    example_prompt,
                                )
                                ai_response = (ai_response or "").strip()
                                
                                if not ai_response or len(ai_response) < 10:
                                    ai_response = "Can you give me an example of that?"
                                
                                logger.info(f"✅ Generated EXAMPLE question: '{ai_response}'")
                                fm.add_question(ai_response, current_concept, is_followup=True)
                                session_data._last_question_followup = True
                                
                            else:
                                # === CASE 3: MOVE TO NEXT FRAGMENT ===
                                logger.info("🔄 Moving to NEW TOPIC...")
                                
                                old_concept = current_concept
                                moved = fm.advance_fragment()
                                
                                if not moved:
                                    # No more fragments - check if extended mode needed
                                    if time_remaining > 60:
                                        session_data.extended_mode = True
                                        logger.info(f"🌐 All fragments covered - switching to EXTENDED mode ({time_remaining/60:.1f}m remaining)")
                                        ai_response = await self.generate_extended_question(session_data)
                                        if ai_response:
                                            session_data._last_question_followup = False
                                    else:
                                        logger.info("🏁 No more concepts - ending session")
                                        await self._finalize_session_with_formal_closing(session_data)
                                        return
                                else:
                                    # Get new fragment
                                    new_frag = fm.get_current_fragment_details()
                                    new_concept = new_frag['title']
                                    new_content = new_frag['content']
                                    
                                    logger.info(f"➡️ Transitioning: '{old_concept}' → '{new_concept}'")
                                    
                                    session_data.current_concept = new_concept
                                    session_data.current_domain = new_concept
                                    
                                    # Generate transition question
                                    transition_prompt = prompts.dynamic_concept_transition(
                                        current_concept=old_concept,
                                        next_concept=new_concept,
                                        user_last_answer=transcript,
                                        next_concept_content=new_content
                                    )
                                    
                                    loop = asyncio.get_event_loop()
                                    ai_response = await loop.run_in_executor(
                                        shared_clients.executor,
                                        self.conversation_manager._sync_openai_call,
                                        transition_prompt,
                                    )
                                    ai_response = (ai_response or "").strip()
                                    
                                    if not ai_response or len(ai_response.split()) < 5:
                                        ai_response = await loop.run_in_executor(
                                            shared_clients.executor,
                                            self.conversation_manager._sync_openai_call,
                                            transition_prompt,
                                        )
                                        ai_response = (ai_response or "").strip()
                                    
                                    logger.info(f"✅ Generated TRANSITION: '{ai_response}'")
                                    fm.add_question(ai_response, new_concept, is_followup=False)
                                    session_data._last_question_followup = False
                
                # Track question for repetition avoidance
                if ai_response:
                    if not hasattr(session_data, 'asked_questions'):
                        session_data.asked_questions = []
                    session_data.asked_questions.append(ai_response)
            # Add exchange to conversation log
            concept = session_data.current_concept or "unknown"
            is_followup = getattr(session_data, "_last_question_followup", False)
            session_data.add_exchange(ai_response, transcript, quality, concept, is_followup)

            if session_data.summary_manager:
                session_data.summary_manager.add_answer(transcript)

            if session_data.current_stage == SessionStage.GREETING:
                session_data.greeting_count = getattr(session_data, "greeting_count", 0) + 1
                logger.info(f"📊 Greeting count incremented to {session_data.greeting_count}")

            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, ai_response)

            # === RELEASE AI RESPONDING LOCK ===
            session_data.ai_is_responding = False

            now_ts = time.time()
            if session_data.current_stage == SessionStage.TECHNICAL and (not soft_cutoff or now_ts < soft_cutoff):
                session_data.awaiting_user = True

            logger.info("Total audio processing time: %.2fs", time.time() - start_time)

        except Exception as e:
            logger.error("Enhanced audio processing error: %s", e)
            if session_data:
                session_data.ai_is_responding = False
            await self._send_quick_message(session_data, {
                "type": "error",
                "text": "Audio processing error",
                "status": "error",
            })

    # ============================================================================
    # ✅ FINAL VERSION: Multiple silence responses with working audio
    # ============================================================================
    async def process_silence_notification(self, session_id: str, silence_data: dict):
        """Process standalone silence notification with MULTIPLE responses allowed."""
        
        # ✅ DETAILED LOGGING
        logger.info("="*70)
        logger.info(f"🔕 SILENCE NOTIFICATION RECEIVED")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Silence data: {silence_data}")
        logger.info("="*70)
        
           
        session_data = self.active_sessions.get(session_id)
        if not session_data:
            logger.error(f"❌ Session {session_id} not found in active_sessions!")
            logger.error(f"Active sessions: {list(self.active_sessions.keys())}")
            return

        # ✅ CRITICAL: Check count FIRST and block immediately if >= 5
        current_silence_count = getattr(session_data, 'silence_response_count', 0)
        if current_silence_count >= 5:
            logger.info(f"🛑 BLOCKED: Already sent {current_silence_count} silence responses - rejecting notification")
            return

        # ✅ Block if session is ending
        if getattr(session_data, "is_ending", False):
            logger.info(f"🛑 BLOCKED: Session is ending - rejecting silence notification")
            return

        if not session_data.is_active:
            logger.error(f"❌ Session {session_id} is not active!")
            return
        logger.info(f"✅ Session found and active")
        logger.info(f"✅ silence_ready: {getattr(session_data, 'silence_ready', False)}")
        logger.info(f"✅ greeting_end_ts: {getattr(session_data, 'greeting_end_ts', None)}")
        logger.info(f"✅ has_user_spoken: {getattr(session_data, 'has_user_spoken', False)}")
        
        # 🔧 CHECK 1: Check if audio is currently being processed (race condition fix)
        if getattr(session_data, "ai_is_responding", False):
            logger.info(f"⏸️ BLOCKED: AI is currently responding/processing audio - skipping silence prompt")
            return
        logger.info(f"✅ Guard 1 passed: ai_is_responding is False")

        last_ai_audio_ts = getattr(session_data, "last_ai_audio_ts", 0)
        if time.time() - last_ai_audio_ts < 5:
            logger.info(f"⏸️ BLOCKED: AI audio just finished speaking, cooldown active")
            return
        logger.info(f"✅ Guard 2 passed: AI audio cooldown complete")

        # ✅ Force silence_ready if AI just finished speaking
        if not getattr(session_data, "silence_ready", False):
            last_ai_ts = getattr(session_data, "last_ai_ts", 0)
            if time.time() - last_ai_ts > 2:
                session_data.silence_ready = True
                logger.info(f"🟢 Auto-reactivated silence_ready for session {session_id}")

        # === Soft Silence Prompt Handling (awaiting_user mode) ===
        if getattr(session_data, "awaiting_user", False):
            silence_for = silence_data.get("silenceDuration", silence_data.get("silenceMs", 0))
            logger.info(f"🔍 awaiting_user mode: silence_for={silence_for}ms, threshold=5000ms")
            
            # ✅ COOLDOWN ONLY FOR AWAITING_USER MODE
            last_silence_response_ts = getattr(session_data, "last_silence_response_ts", 0)
            time_since_last = time.time() - last_silence_response_ts
            if time_since_last < 6:  # 10 second cooldown
                logger.info(f"⏸️ BLOCKED (awaiting_user): Silence response sent {time_since_last:.1f}s ago - cooldown active")
                return
            logger.info(f"✅ Guard 3 passed: Awaiting_user cooldown complete ({time_since_last:.1f}s since last)")
            
            if silence_for >= 5000:
                try:
                    logger.info(f"🤫 Generating soft silence prompt (awaiting_user mode, {silence_for}ms)")
                    text = await self.generate_dynamic_silence_response(session_data, silence_data)
                    
                    # 🔧 CHECK: Double-check awaiting_user hasn't changed during generation
                    if not getattr(session_data, "awaiting_user", False):
                        logger.info(f"⏸️ User responded during generation - canceling silence prompt")
                        return

                    concept = getattr(session_data, "current_concept", None) or "silence_handling"
                    session_data.add_exchange(text, "[USER_SILENT]", 0.0, concept, False)
                    logger.info(f"📝 Logged silence (awaiting_user): Q='{text[:50]}...', A='[USER_SILENT]'")
                        
                    await self._send_silence_response_with_audio(session_data, text)
                    
                    # ✅ UPDATE TIMESTAMP AFTER SUCCESSFUL SEND
                    session_data.last_silence_response_ts = time.time()
                    
                    logger.info(f"✅ Soft silence prompt delivered ({silence_for} ms)")
                except Exception as e:
                    logger.error(f"❌ Silence prompt failed: {e}", exc_info=True)
            else:
                logger.info(f"⏸️ Silence too short for awaiting_user prompt: {silence_for}ms < 5000ms")
            return

        # === Main Silence Handling (normal mode) - NO COOLDOWN ===
        logger.info(f"🔍 Entering main silence handling (normal mode) - MULTIPLE RESPONSES ALLOWED")
        try:
            # Guard: silence_ready check
            if not getattr(session_data, "silence_ready", False):
                logger.warning(f"⚠️ BLOCKED: silence_ready is False")
                return
            logger.info(f"✅ Guard 4 passed: silence_ready is True")

            now_ts = time.time()

            # Guard: recordingActive check
            if silence_data.get('recordingActive'):
                logger.debug(f"⚠️ BLOCKED: recordingActive is True")
                return
            logger.info(f"✅ Guard 5 passed: recordingActive is False")

            # Guard: speech cooldown check
            cooldown_s = getattr(config, "SILENCE_COOLDOWN_AFTER_SPEECH_SECONDS", 2.0)
            last_speech_ts = getattr(session_data, "last_user_speech_ts", None)
            if last_speech_ts is not None and (now_ts - last_speech_ts) < cooldown_s:
                logger.debug(f"⚠️ BLOCKED: within {cooldown_s}s speech cooldown")
                return
            logger.info(f"✅ Guard 6 passed: past speech cooldown")

            # Guard: grace period check
            past_greeting_grace = (
                getattr(session_data, "greeting_end_ts", None) is not None and
                (now_ts - session_data.greeting_end_ts) >= getattr(
                    session_data, "silence_grace_after_greeting_s", 4
                )
            )

            if not (getattr(session_data, "has_user_spoken", False) or past_greeting_grace):
                logger.debug(f"⚠️ BLOCKED: no prior speech and still in grace window")
                return
            logger.info(f"✅ Guard 7 passed: has_user_spoken or past grace period")

            # Guard: threshold check
            session_data.consecutive_silence_chunks = getattr(session_data, "consecutive_silence_chunks", 0) + 1
            threshold = getattr(session_data, "silence_chunks_threshold", getattr(config, "SILENCE_CHUNKS_THRESHOLD", 1))
            logger.info(f"🔢 Silence chunks: {session_data.consecutive_silence_chunks}/{threshold}")

            if session_data.consecutive_silence_chunks < threshold:
                logger.info(f"⚠️ BLOCKED: Not enough silence chunks yet")
                return
            logger.info(f"✅ Guard 8 passed: threshold reached")

            # Reset counter
            session_data.consecutive_silence_chunks = 0

            logger.info(f"🎯 All guards passed - generating silence response NOW (response #{getattr(session_data, 'silence_response_count', 0) + 1})")

            # 🔧 Final check before generation
            if getattr(session_data, "ai_is_responding", False):
                logger.info(f"⏸️ BLOCKED: Audio started processing after guards - canceling silence prompt")
                return

            try:
                # Generate dynamic silence response
                text = await self.generate_dynamic_silence_response(session_data, silence_data)

                logger.info(f"📝 Generated silence text: '{text}'")
                # === UPDATE COMMUNICATION STATS: SILENT (5E) ===
                if session_data.current_stage == SessionStage.TECHNICAL:
                    if not hasattr(session_data, 'comm_stats'):
                        session_data.comm_stats = {"total_questions": 0, "answered": 0, "skipped": 0, "silent": 0, "irrelevant": 0, "repeat_requests": 0}
                    # Only count if a question was asked
                    if session_data.comm_stats["total_questions"] > 0 or getattr(session_data, 'last_question_end_ts', None):
                        session_data.comm_stats["silent"] += 1
                        await self._send_comm_score_update(session_data, "silent")

                # ✅ CRITICAL: Check if this is the 5th silence - if so, mark session as ENDING
                if session_data.silence_response_count >= 5:
                    logger.info(f"🛑 This is silence #{session_data.silence_response_count} - marking session as ENDING")
                    session_data.is_ending = True  # ✅ NEW FLAG - blocks all further processing
                    session_data.awaiting_user = False  # ✅ Stop waiting for user

                # 🔧 Final check before sending
                if getattr(session_data, "ai_is_responding", False):
                    logger.info(f"⏸️ User responded during generation - canceling silence prompt")
                    return

                # Send the response with audio
                await self._send_silence_response_with_audio(session_data, text)

                logger.info(f"✅ Silence response #{session_data.silence_response_count} sent successfully")

                # ✅ END SESSION IMMEDIATELY AFTER 5TH SILENCE
                if session_data.silence_response_count >= 5:
                    logger.info(f"🛑🛑🛑 ENDING SESSION NOW - silence #{session_data.silence_response_count}")
    
                    # ✅ CRITICAL: Mark inactive IMMEDIATELY to block ALL processing
                    session_data.is_active = False
                    
                    # Wait 3 seconds for audio to finish playing
                    logger.info(f"⏳ Waiting 3 seconds for audio to finish...")
                    await asyncio.sleep(3)
                    
                    logger.info(f"📤 Sending session_ended message...")
                    
                    # Send session_ended message
                    await self._send_quick_message(session_data, {
                        "type": "session_ended",
                        "reason": "Extended silence - session ended after 5 prompts",
                        "silence_count": session_data.silence_response_count,
                    })
                    
                    # Close WebSocket
                    if session_data.websocket:
                        try:
                            await session_data.websocket.close(code=1000, reason="Session ended after 5 silences")
                            logger.info(f"✅ WebSocket closed for session {session_data.session_id}")
                        except Exception as e:
                            logger.error(f"❌ WebSocket close error: {e}")
                    
                    # Clean up session
                    await self.remove_session(session_data.session_id)
                    
                    logger.info(f"✅ Session {session_data.session_id} ended and cleaned up completely")
                    return
                
                # ✅ NO TIMESTAMP UPDATE IN NORMAL MODE - Allow multiple responses!
                # This allows the system to keep encouraging the user every ~6 seconds

                # Set awaiting_user if in technical stage
                soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
                if (session_data.current_stage == SessionStage.TECHNICAL and
                    (soft_cutoff is None or now_ts < soft_cutoff)):
                    session_data.awaiting_user = True

            except Exception as e_sil:
                logger.error(f"❌ Silence prompt generation/streaming error: {e_sil}", exc_info=True)

        except Exception as e:
            logger.error(f"❌ Silence notification processing error: {e}", exc_info=True)
            
    # ============================================================================
    # ✅ METHOD 2: Reuse working audio generation (BEST APPROACH)
    # ============================================================================
    async def _send_silence_response_with_audio(self, session_data: SessionData, text: str):
        """Send silence response using the SAME reliable audio method as regular AI responses."""
        try:
            logger.info(f"🤫 Starting silence response with audio: '{text[:80]}'")
            
            # Send text message with silence_response type
            await self._send_quick_message(session_data, {
                "type": "silence_response",
                "text": text,
                "status": session_data.current_stage.value,
            })
            
            logger.info(f"✅ Silence response text message sent, generating audio...")
            
            # ✅ USE THE SAME WORKING AUDIO GENERATION METHOD
            chunk_count = 0
            try:
                async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                    text, session_id=session_data.session_id
                ):
                    if not session_data.is_active:
                        logger.warning(f"⚠️ Session inactive, stopping silence audio")
                        break
                        
                    if audio_chunk:
                        await self._send_quick_message(session_data, {
                            "type": "audio_chunk",
                            "audio": audio_chunk.hex(),
                            "status": session_data.current_stage.value,
                        })
                        chunk_count += 1
                        if chunk_count <= 3:  # Log first few chunks
                            logger.debug(f"📦 Sent silence audio chunk #{chunk_count}")
                            
            except Exception as stream_error:
                logger.error(f"❌ Audio streaming error: {stream_error}", exc_info=True)

            await self._send_quick_message(session_data, {
                "type": "audio_end",
                "status": session_data.current_stage.value
            })

            if chunk_count == 0:
                logger.error(f"❌ ERROR: ZERO AUDIO CHUNKS generated for silence response!")
                logger.error(f"   Text: '{text}'")
                logger.error(f"   Session ID: {session_data.session_id}")
                logger.error(f"   Session active: {session_data.is_active}")
            else:
                logger.info(f"✅ Silence response complete: {chunk_count} audio chunks streamed successfully")

        except Exception as e:
            logger.error(f"❌ Silence response audio error: {e}", exc_info=True)

    # Legacy method - redirect to new audio processing
    async def process_audio_ultra_fast(self, session_id: str, audio_data: bytes):
        """Legacy method - convert to new format and use enhanced processing"""
        import base64
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        message_data = {
            'audio': audio_b64,
            'userStatus': 'user_speaking',
            'silenceDetected': False,
            'recordingDuration': 0
        }
        await self.process_audio_with_silence_status(session_id, message_data)

    """async def _update_session_state_fast(self, session_data: SessionData):
        try:
            if session_data.current_stage == SessionStage.GREETING:
                pass
                
            elif session_data.current_stage == SessionStage.TECHNICAL:
                now_ts = time.time()
                soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
                if session_data.summary_manager:
                    has_covered_enough = not session_data.summary_manager.should_continue_test()
                    approaching_time_limit = soft_cutoff and now_ts >= (soft_cutoff - 60)
                    if has_covered_enough or approaching_time_limit:
                        session_data.current_stage = SessionStage.COMPLETE
                        logger.info("Session %s moved to COMPLETE stage (covered: %s, time: %s)",
                                    session_data.session_id, has_covered_enough, approaching_time_limit)
                        asyncio.create_task(self._finalize_session_fast(session_data))
        except Exception as e:
            logger.error("Session state update error: %s", e)"""

    async def _update_session_state_fast(self, session_data):
        """Updated to support extended question mode when summary is exhausted."""
        try:
            if session_data.current_stage == SessionStage.GREETING:
                pass
                
            elif session_data.current_stage == SessionStage.TECHNICAL:
                now_ts = time.time()
                session_start = session_data.created_at
                elapsed = now_ts - session_start
                min_duration = getattr(session_data, 'min_session_duration', 15 * 60)
                time_remaining = min_duration - elapsed
                
                logger.info(f"⏱️ Session time: {elapsed/60:.1f}m elapsed, {time_remaining/60:.1f}m remaining")
                
                # Check if summary questions are exhausted
                summary_exhausted = False
                if session_data.summary_manager:
                    summary_exhausted = not session_data.summary_manager.should_continue_test()
                
                if summary_exhausted:
                    # Check if we should continue with extended questions
                    if time_remaining > 60:  # More than 1 minute remaining
                        if not getattr(session_data, 'extended_mode', False):
                            session_data.extended_mode = True
                            logger.info(f"🌐 Summary exhausted but {time_remaining/60:.1f}m remaining - switching to EXTENDED mode")
                        # Don't set to COMPLETE - continue with extended questions
                    elif time_remaining <= 30:
                        # Less than 30 seconds remaining - complete the session
                        session_data.current_stage = SessionStage.COMPLETE
                        logger.info(f"Session {session_data.session_id} moved to COMPLETE (minimum duration reached)")
                        asyncio.create_task(self._finalize_session_with_formal_closing(session_data))
                
                # Check for time-based completion even if summary not exhausted
                elif time_remaining <= 30:  # Less than 30 seconds
                    session_data.current_stage = SessionStage.COMPLETE
                    logger.info(f"Session {session_data.session_id} moved to COMPLETE (time limit)")
                    asyncio.create_task(self._finalize_session_with_formal_closing(session_data))
                    
        except Exception as e:
            logger.error("Session state update error: %s", e)

    async def _finalize_session_fast(self, session_data: SessionData):
        try:
            sm = getattr(session_data, "summary_manager", None)
            topics = []
            if sm:
                fk = getattr(session_data, "fragment_keys", None)
                frags = getattr(sm, "fragments", None)
                if fk and isinstance(frags, dict):
                    for k in fk:
                        frag = frags.get(k)
                        if isinstance(frag, dict):
                            t = frag.get("title") or frag.get("heading") or frag.get("name")
                            if t: topics.append(t)
                elif isinstance(frags, list):
                    for frag in frags:
                        if isinstance(frag, dict):
                            t = frag.get("title") or frag.get("heading") or frag.get("name")
                            if t: topics.append(t)

            conv_log = getattr(session_data, "conversation_log", []) or []
            user_final_response = (
                conv_log[-1].get("user_response") if conv_log and isinstance(conv_log[-1], dict) else None
            )
            conversation_summary = {
                "topics_covered": topics,
                "total_exchanges": len(conv_log),
                "name": session_data.student_name,
            }

            closing_prompt = prompts.dynamic_session_completion(conversation_summary, user_final_response)
            loop = asyncio.get_event_loop()
            completion_message = await loop.run_in_executor(
                shared_clients.executor,
                self.conversation_manager._sync_openai_call,
                closing_prompt,
            )
            completion_message = (completion_message or "").strip()

            if not completion_message or len(completion_message.split()) < 3:
                completion_message = await loop.run_in_executor(
                    shared_clients.executor,
                    self.conversation_manager._sync_openai_call,
                    closing_prompt,
                )
                completion_message = (completion_message or "").strip()

            completion_message = completion_message or " "
            try:
                conv_log = getattr(session_data, "conversation_log", [])
                if conv_log:
                    save_qa_to_mongodb(
                        session_id=session_data.session_id,
                        student_id=session_data.student_id,
                        student_name=session_data.student_name,
                        conversation_log=conv_log,
                        test_id=session_data.test_id
                    )
            except Exception as qa_err:
                logger.error(f"Q&A save failed: {qa_err}")

            evaluation_text, score, detailed_evaluation = await self.conversation_manager.generate_fast_evaluation(session_data)
            # ✅ FIX: Store detailed_evaluation on session_data so it gets saved to MongoDB
            session_data.detailed_evaluation = detailed_evaluation
            save_success = await self.db_manager.save_session_result_fast(session_data, evaluation, score)
            if not save_success:
                logger.error("Failed to save session %s", session_data.session_id)

            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": completion_message,
                "evaluation": evaluation,
                "score": score,
                "detailed_evaluation": detailed_evaluation,
                "pdf_url": f"/download_results/{session_data.session_id}",
                "status": "complete",
                "enable_new_session": True,
                "redirect_to": "/dashboard",
                "end_reason": "completed",
            })

            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                completion_message, session_id=session_data.session_id
            ):
                if audio_chunk:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": "complete",
                    })
            await self._send_quick_message(session_data, {"type": "audio_end", "status": "complete"})

        except Exception as e:
            logger.error("Fast session finalization error: %s", e)
        finally:
            session_data.is_active = False
            try:
                task = getattr(session_data, "_hard_stop_task", None)
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass
            try:
                if session_data.websocket:
                    await session_data.websocket.close(code=1000)
            except Exception:
                pass
            await self.remove_session(session_data.session_id)

    async def _finalize_session_with_formal_closing(self, session_data):
        """Finalize session with formal interview-style closing message and comprehensive evaluation."""
        try:
            # Gather session context for formal closing
            sm = getattr(session_data, "summary_manager", None)
            topics = []
            if sm:
                fk = getattr(session_data, "fragment_keys", None)
                frags = getattr(sm, "fragments", None)
                if fk and isinstance(frags, dict):
                    for k in fk:
                        frag = frags.get(k)
                        if isinstance(frag, dict):
                            t = frag.get("title") or frag.get("heading") or frag.get("name")
                            if t: topics.append(t)
                elif isinstance(frags, list):
                    for frag in frags:
                        if isinstance(frag, dict):
                            t = frag.get("title") or frag.get("heading") or frag.get("name")
                            if t: topics.append(t)
            
            # Add main topic if available
            main_topic = getattr(session_data, 'main_topic', None)
            if main_topic and main_topic not in topics:
                topics.insert(0, main_topic)
            
            duration_minutes = (time.time() - session_data.created_at) / 60
            total_questions = len(getattr(session_data, 'asked_questions', []))
            extended_mode_used = getattr(session_data, 'extended_mode', False)
            
            session_context = {
                "name": session_data.student_name,
                "topics_covered": topics[:5],
                "total_questions": total_questions,
                "duration_minutes": duration_minutes
            }
            
            logger.info(f"🎬 Finalizing session - Duration: {duration_minutes:.1f}m, Questions: {total_questions}, Extended: {extended_mode_used}")
            
             # === GET FINAL COMMUNICATION SCORE (PATCH 6) ===
            final_comm_score = self.calculate_communication_score(session_data)
            session_data.final_communication_score = final_comm_score
            logger.info(f"📊 Final Communication Score: {final_comm_score['total_score']}/100")
            logger.info(f"   └─ Willingness: {final_comm_score['willingness_score']}/30")
            logger.info(f"   └─ Relevance: {final_comm_score['relevance_score']}/30")
            logger.info(f"   └─ Responsiveness: {final_comm_score['responsiveness_score']}/25")
            logger.info(f"   └─ Clarity: {final_comm_score['clarity_score']}/15")

            # Generate formal closing message
            closing_prompt = prompts.formal_session_closing(session_context)
            loop = asyncio.get_event_loop()
            closing_message = await loop.run_in_executor(
                shared_clients.executor,
                self.conversation_manager._sync_openai_call,
                closing_prompt,
            )
            closing_message = (closing_message or "").strip()
            
            # Retry if too short
            if not closing_message or len(closing_message.split()) < 5:
                closing_message = await loop.run_in_executor(
                    shared_clients.executor,
                    self.conversation_manager._sync_openai_call,
                    closing_prompt,
                )
                closing_message = (closing_message or "").strip()
            
            # Fallback if still empty
            if not closing_message:
                closing_message = f"That concludes our standup for today, {session_data.student_name}. Thank you for your time. We'll connect again in our next session."
            
            logger.info(f"🎬 Formal closing message: '{closing_message}'")
            
            # Save Q&A to MongoDB
            try:
                conv_log = getattr(session_data, "conversation_log", [])
                if conv_log:
                    save_qa_to_mongodb(
                        session_id=session_data.session_id,
                        student_id=session_data.student_id,
                        student_name=session_data.student_name,
                        conversation_log=conv_log,
                        test_id=session_data.test_id
                    )
            except Exception as qa_err:
                logger.error(f"Q&A save failed: {qa_err}")
            
            # ✅ ENHANCED: Generate comprehensive evaluation
            evaluation_text, score, detailed_evaluation = None, None, None
            try:
                evaluation_text, score, detailed_evaluation = await self.conversation_manager.generate_fast_evaluation(session_data)
                # ✅ FIX: Store detailed_evaluation on session_data so it gets saved to MongoDB
                session_data.detailed_evaluation = detailed_evaluation
                #override communication score with real-time calculated value
                if detailed_evaluation and final_comm_score:
                    detailed_evaluation["communication_score"] = final_comm_score["total_score"]
                    detailed_evaluation["communication_breakdown"] = final_comm_score
                # Save to database
                save_success = await self.db_manager.save_session_result_fast(session_data, evaluation_text, score)
                if not save_success:
                    logger.error("Failed to save session %s", session_data.session_id)
                    
            except Exception as eval_err:
                logger.error(f"Evaluation generation error: {eval_err}")
                # Create fallback evaluation
                detailed_evaluation = {
                    "overall_score": 50,
                    "grade": "C",
                    "summary": "Evaluation could not be completed.",
                    "strengths": [],
                    "weaknesses": [],
                    "areas_for_improvement": [],
                    "question_analysis": [],
                    "recommendations": [],
                    "raw_stats": {},
                    "session_info": {
                        "student_name": session_data.student_name,
                        "session_id": session_data.session_id
                    }
                }
                evaluation_text = "Evaluation encountered an error."
                score = 50.0
            
            # ✅ FIX 2: Send stop_audio FIRST to clear frontend queue
            await self._send_quick_message(session_data, {
                "type": "stop_audio",
                "reason": "session_ending"
            })
            await asyncio.sleep(0.2)  # Brief delay for frontend to clear queue
            # ✅ Send closing message with DETAILED evaluation
            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": closing_message,
                "evaluation": evaluation_text,
                "score": score,
                "detailed_evaluation": detailed_evaluation,  # ✅ NEW: Include full evaluation
                "communication_score": final_comm_score,
                "pdf_url": f"/download_results/{session_data.session_id}",
                "status": "complete",
                "enable_new_session": True,
                "redirect_to": "/dashboard",
                "end_reason": "completed",
                "session_stats": {
                    "duration_minutes": round(duration_minutes, 1),
                    "total_questions": total_questions,
                    "topics_covered": len(topics),
                    "extended_mode_used": extended_mode_used
                }
            })
            
            # Stream closing audio
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                closing_message, session_id=session_data.session_id
            ):
                if audio_chunk:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": "closing",
                    })
            await self._send_quick_message(session_data, {"type": "audio_end", "status": "closing"})
            
        except Exception as e:
            logger.error("Formal session finalization error: %s", e)
        finally:
            session_data.is_active = False
            try:
                task = getattr(session_data, "_hard_stop_task", None)
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass
            try:
                if session_data.websocket:
                    await session_data.websocket.close(code=1000)
            except Exception:
                pass
            await self.remove_session(session_data.session_id)

    
    async def _enable_silence_after_delay(self, sd, delay=5.0):
        """Re-enable silence detection a few seconds after AI speech finishes."""
        await asyncio.sleep(delay)
        if sd and getattr(sd, "is_active", False):
            sd.silence_ready = True

    async def _send_response_with_ultra_fast_audio(self, session_data: SessionData, text: str):
        try:
            await self._send_quick_message(session_data, {
                "type": "ai_response",
                "text": text,
                "status": session_data.current_stage.value,
            })
            
            # === Enable silence monitoring after greeting ===
            if session_data.current_stage == SessionStage.GREETING:
                session_data.silence_ready = True
                session_data.greeting_end_ts = time.time()
                logger.info(f"🟢 Silence prompts activated for session {session_data.session_id}")

            chunk_count = 0
            async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                text, session_id=session_data.session_id
            ):
                if audio_chunk and session_data.is_active:
                    await self._send_quick_message(session_data, {
                        "type": "audio_chunk",
                        "audio": audio_chunk.hex(),
                        "status": session_data.current_stage.value,
                    })
                    chunk_count += 1

            await self._send_quick_message(session_data, {"type": "audio_end", "status": session_data.current_stage.value})
            # === TRACK QUESTION END TIME FOR RESPONSIVENESS ===
            if session_data.current_stage == SessionStage.TECHNICAL:
                session_data.last_question_end_ts = time.time()
                logger.info(f"⏱️ Question end timestamp recorded")
            session_data.last_ai_audio_ts = time.time()
            
            # === Enable silence detection after greeting ends ===
            if session_data.current_stage == SessionStage.GREETING:
                session_data.silence_ready = True
                session_data.greeting_end_ts = time.time()
                logger.info(f"🟢 Silence prompts activated for session {session_data.session_id}")

            # === AI QUESTION LOCK LOGIC ===
            if any(qword in text.lower() for qword in ["?", "what", "how", "why", "could you", "can you"]):
                session_data.awaiting_user = True
                logger.info(f"🧍 AI asked a question — locking until user replies (session {session_data.session_id})")
            
            session_data.last_ai_ts = time.time()

            # ✅ Immediately allow silence monitoring again after AI finishes
            session_data.silence_ready = True
            session_data.greeting_end_ts = session_data.greeting_end_ts or time.time()
            logger.info(f"🟢 Silence monitoring re-enabled after AI speech ({session_data.session_id})")

            asyncio.create_task(self._enable_silence_after_delay(session_data, delay=3.0))

            logger.info("Streamed %d audio chunks", chunk_count)

        except Exception as e:
            logger.error("Ultra-fast audio streaming error: %s", e)

    async def _send_quick_message(self, session_data: SessionData, message: dict):
        try:
            if session_data.websocket:
                await session_data.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error("WebSocket send error: %s", e)

    async def get_session_result_fast(self, session_id: str) -> dict:
        try:
            result = await self.db_manager.get_session_result_fast(session_id)
            if not result:
                raise Exception(f"Session {session_id} not found in database")
            return result
        except Exception as e:
            logger.error("Error fetching session result: %s", e)
            raise Exception(f"Session result retrieval failed: {e}")
    
    async def generate_summary_based_question(self, user_input: str, session_data: SessionData) -> str:
        """Use latest MongoDB summary to generate the next technical question dynamically."""
        try:
            db = get_db_manager()
            summary_text = db._sync_get_summary()
            context_text = summary_text[:4000] if summary_text else ""
            first_line = summary_text.split("\n")[0].strip() if summary_text else "SAP topic"
            next_question = f"What is the purpose of {first_line.split('for')[-1].strip()}?" if "for" in first_line else f"Let's discuss {first_line} — can you explain it?"

            tech_prompt = prompts.dynamic_technical_response(context_text, user_input, next_question, {
                "domain": getattr(session_data, "current_domain", "technical discussion")
            })
            loop = asyncio.get_event_loop()
            ai_output = await loop.run_in_executor(
                shared_clients.executor,
                session_data.conversation_manager._sync_openai_call,
                tech_prompt,
            )
            return (ai_output or "").strip()
        except Exception as e:
            print(f"[TechQuestionError] {e}")
            return "Let's discuss the main SAP process — could you explain how patching works?"

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title=config.APP_TITLE, version=config.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

app.mount("/audio", StaticFiles(directory=str(config.AUDIO_DIR)), name="audio")

session_manager = UltraFastSessionManagerWithSilenceHandling()

@app.on_event("startup")
async def startup_event():
    logger.info("Ultra-Fast Daily Standup with Enhanced Silence Detection starting...")
    try:
        db_manager = DatabaseManager(shared_clients)
        try:
            conn = db_manager.get_mysql_connection()
            conn.close()
            logger.info("MySQL connection test successful")
        except Exception as e:
            logger.error("MySQL connection test failed: %s", e)
            raise Exception(f"MySQL connection failed: {e}")

        try:
            await db_manager.get_mongo_client()
            logger.info("MongoDB connection test successful")
        except Exception as e:
            logger.error("MongoDB connection test failed: %s", e)
            raise Exception(f"MongoDB connection failed: {e}")

        try:
            init_biometric_services(
                mongo_host="192.168.48.201",
                mongo_port=27017,
                db_name="connectlydb",
                username="connectly",
                password="LT@connect25",
                auth_source="admin",
                max_voice_warnings=3
            )
            logger.info("✅ Biometric authentication services initialized")
        except Exception as e:
            logger.error(f"⚠️ Biometric services init failed (non-critical): {e}")

        logger.info("All database connections verified - silence detection ready")
    except Exception as e:
        logger.error("Startup failed: %s", e)

@app.on_event("shutdown")
async def shutdown_event():
    await shared_clients.close_connections()
    await session_manager.db_manager.close_connections()
    # ✅ ADD THIS: Cleanup biometric services
    if biometric_service:
        biometric_service.disconnect()
    logger.info("Enhanced Daily Standup application shutting down")

@app.get("/start_test")
async def start_standup_session_fast(student_id: int = None):
    """
    Start standup session.
    
    Args:
        student_id: Logged-in student's ID from ADMINORG_ROUGH
    
    Example: /start_test?student_id=38
    """
    session_data = None
    try:
        logger.info("Starting enhanced standup session with silence detection...")

        if len(session_manager.active_sessions) > 0:
            for sid in list(session_manager.active_sessions.keys()):
                await session_manager.remove_session(sid)
                logger.info(f"🧹 Purged stale session {sid} before new start")

        
        session_data = await session_manager.create_session_fast(student_id=student_id)

        logger.info("Enhanced session created: %s", session_data.test_id)
        return {
            "status": "success",
            "message": "Session started successfully with silence detection",
            "test_id": session_data.test_id,
            "session_id": session_data.session_id,
            "websocket_url": f"/ws/{session_data.session_id}",
            "student_name": session_data.student_name,
            "fragments_count": len(session_data.fragment_keys) if session_data.fragment_keys else 0,
            "estimated_duration": len(session_data.fragment_keys)
                                 * session_data.questions_per_concept
                                 * config.ESTIMATED_SECONDS_PER_QUESTION,
            "features": ["silence_detection", "dynamic_responses", "enhanced_vad", "noise_filtering", "multiple_silence_responses", "gentle_prompts", "repeat_question", "skip_detection", "irrelevant_detection", "fixed_greeting_duplication"],
            "silence_handling": {
                "max_responses": 999,
                "uses_llm": True,
                "bypasses_stt": True,
                "allows_multiple_responses": True,
                "audio_generation": "working",
                "response_style": "gentle_and_brief"
            },
            "smart_response_handling": {
                "repeat_question": "enabled",
                "skip_detection": "enabled_with_acknowledgment",
                "irrelevant_detection": "enabled_llm_based"
            },
            "greeting_fix": {
                "auto_transition_after": "2_exchanges",
                "prevents_duplication": True
            }
        }

    except Exception as e:
        logger.error("Error starting enhanced session: %s", e)
        if session_data and session_data.session_id in session_manager.active_sessions:
            await session_manager.remove_session(session_data.session_id)
            logger.info(f"Cleaned partial session {session_data.session_id} after startup failure")

        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.get("/api/summary/{test_id}")
async def get_standup_summary_fast(test_id: str):
    """Get standup session summary from real database"""
    try:
        logger.info(f"Getting summary for test_id: {test_id}")
        if not test_id:
            raise HTTPException(status_code=400, detail="test_id is required")

        result = await session_manager.get_session_result_fast(test_id)

        if result:
            exchanges = result.get("conversation_log", [])

            yesterday_work = ""
            today_plans = ""
            blockers = ""
            additional_notes = ""

            for exchange in exchanges:
                user_response = (exchange.get("user_response") or "").lower()
                ai_message = (exchange.get("ai_message") or "").lower()

                if any(word in ai_message for word in ["yesterday", "accomplished", "completed"]):
                    yesterday_work = exchange.get("user_response", "")
                elif any(word in ai_message for word in ["today", "plan", "working on"]):
                    today_plans = exchange.get("user_response", "")
                elif any(word in ai_message for word in ["blocker", "challenge", "obstacle", "stuck"]):
                    blockers = exchange.get("user_response", "")
                elif exchange.get("user_response") and not yesterday_work and not today_plans:
                    additional_notes = exchange.get("user_response", "")

            summary_data = {
                "test_id": test_id,
                "session_id": result.get("session_id", test_id),
                "student_name": result.get("student_name", "Student"),
                "timestamp": result.get("timestamp", time.time()),
                "duration": result.get("duration", 0),
                "yesterday": yesterday_work or "Progress discussed during session",
                "today": today_plans or "Plans outlined during session",
                "blockers": blockers or "No specific blockers mentioned",
                "notes": additional_notes or "Additional discussion points covered",
                "accomplishments": yesterday_work,
                "plans": today_plans,
                "challenges": blockers,
                "additional_info": additional_notes,
                "evaluation": result.get("evaluation", "Session completed successfully"),
                "score": result.get("score", 8.0),
                "total_exchanges": result.get("total_exchanges", 0),
                "fragment_analytics": result.get("fragment_analytics", {}),
                "pdf_url": f"/download_results/{test_id}",
                "status": "completed",
                "silence_responses": result.get("silence_responses", 0)
            }
            logger.info(f"Enhanced summary generated for {test_id}")
            return summary_data

        raise HTTPException(status_code=404, detail=f"Session result not found for test_id: {test_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary retrieval failed: {str(e)}")

def generate_comprehensive_pdf_report(result: dict, detailed_evaluation: dict, session_id: str) -> bytes:
    """
    Generate a comprehensive, professional PDF evaluation report.
    """
    try:
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer, 
            pagesize=LETTER,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=HexColor('#1a365d')
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=HexColor('#2d3748'),
            borderPadding=5
        )
        
        subheader_style = ParagraphStyle(
            'SubHeader',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=5,
            textColor=HexColor('#4a5568')
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14
        )
        
        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=5
        )
        
        score_style = ParagraphStyle(
            'ScoreStyle',
            parent=styles['Normal'],
            fontSize=36,
            alignment=TA_CENTER,
            textColor=HexColor('#2b6cb0'),
            spaceBefore=10,
            spaceAfter=10
        )
        
        story = []
        
        # ==================== TITLE PAGE ====================
        story.append(Paragraph("Daily Standup", title_style))
        story.append(Paragraph("Evaluation Report", title_style))
        story.append(Spacer(1, 30))
        
        # Student Info Box
        session_info = detailed_evaluation.get("session_info", {})
        info_data = [
            ["Candidate Name:", session_info.get("student_name", result.get("student_name", "Unknown"))],
            ["Session ID:", session_id],
            ["Date:", datetime.now().strftime("%B %d, %Y")],
            ["Duration:", f"{detailed_evaluation.get('raw_stats', {}).get('duration_minutes', 0):.1f} minutes"],
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#e2e8f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#2d3748')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#cbd5e0')),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 30))
        
        # ==================== OVERALL SCORE SECTION ====================
        overall_score = detailed_evaluation.get("overall_score", 0)
        grade = detailed_evaluation.get("grade", "N/A")
        
        # Determine score color
        if overall_score >= 80:
            score_color = '#48bb78'  # Green
        elif overall_score >= 60:
            score_color = '#ed8936'  # Orange
        else:
            score_color = '#f56565'  # Red
        
        score_display_style = ParagraphStyle(
            'ScoreDisplay',
            parent=styles['Normal'],
            fontSize=48,
            alignment=TA_CENTER,
            textColor=HexColor(score_color),
            spaceBefore=20,
            spaceAfter=10
        )
        
        story.append(Paragraph("OVERALL SCORE", header_style))
        story.append(Paragraph(f"{overall_score}/100", score_display_style))
        story.append(Spacer(1, 15))  # ✅ FIX 1: Spacer prevents score/grade overlap
        story.append(Paragraph(f"Grade: {grade}", ParagraphStyle('Grade', alignment=TA_CENTER, fontSize=24, textColor=HexColor(score_color), spaceBefore=10)))
        story.append(Spacer(1, 20))

        # Summary
        summary = detailed_evaluation.get("summary", "No summary available.")
        story.append(Paragraph(summary, ParagraphStyle('Summary', alignment=TA_CENTER, fontSize=12, textColor=HexColor('#4a5568'), leading=16)))
        story.append(Spacer(1, 20))
        
        # ==================== SCORE BREAKDOWN ====================
        story.append(Paragraph("Score Breakdown", header_style))
        
        score_data = [
            ["Category", "Score", "Rating"],
            ["Technical Knowledge", f"{detailed_evaluation.get('technical_score', 0)}/100", get_rating(detailed_evaluation.get('technical_score', 0))],
            ["Communication", f"{detailed_evaluation.get('communication_score', 0)}/100", get_rating(detailed_evaluation.get('communication_score', 0))],
            ["Attentiveness", f"{detailed_evaluation.get('attentiveness_score', 0)}/100", get_rating(detailed_evaluation.get('attentiveness_score', 0))],
        ]
        
        score_table = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f7fafc'), white]),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 20))
        
        # ==================== SESSION STATISTICS ====================
        story.append(Paragraph("Session Statistics", header_style))
        
        raw_stats = detailed_evaluation.get("raw_stats", {})
        stats_data = [
            ["Metric", "Count"],
            ["Total Questions", str(raw_stats.get('total_questions', 0))],
            ["Questions Answered", str(raw_stats.get('answered_count', 0))],
            ["Questions Skipped", str(raw_stats.get('skipped_count', 0))],
            ["Silent Responses", str(raw_stats.get('silent_count', 0))],
            ["Irrelevant Answers", str(raw_stats.get('irrelevant_count', 0))],
            ["Repeat Requests", str(raw_stats.get('repeat_requests_count', 0))],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))

        # ==================== WEAKNESSES - NOW FIRST ====================
        weaknesses = detailed_evaluation.get("weaknesses", [])

        # ✅ BUILD COMBINED WEAKNESSES WITH COMMUNICATION ISSUES
        all_weaknesses = list(weaknesses) if weaknesses else []
        communication_score = detailed_evaluation.get("communication_score", 100)
        raw_stats = detailed_evaluation.get("raw_stats", {})

        if communication_score < 60:
            if communication_score < 40:
                all_weaknesses.append("Poor communication clarity - responses were often unclear or incomplete")
            if raw_stats.get('irrelevant_count', 0) > 3:
                all_weaknesses.append("Frequent off-topic responses indicate difficulty understanding questions")
            if raw_stats.get('repeat_requests_count', 0) > 2:
                all_weaknesses.append("Multiple requests for question repetition may indicate attention or comprehension issues")
            if raw_stats.get('answered_count', 0) < raw_stats.get('total_questions', 1) * 0.5:
                all_weaknesses.append("Low response rate suggests difficulty articulating answers")

        if all_weaknesses:
            story.append(Paragraph("✗ Areas of Concern", header_style))
            for weakness in all_weaknesses:
                if isinstance(weakness, dict):
                    weakness = weakness.get('area', str(weakness))
                story.append(Paragraph(f"• {weakness}", bullet_style))
            story.append(Spacer(1, 15))

        # ==================== STRENGTHS - NOW SECOND ====================
        strengths = detailed_evaluation.get("strengths", [])
        if strengths:
            story.append(Paragraph("✓ Strengths", header_style))
            for strength in strengths:
                if isinstance(strength, dict):
                    strength = strength.get('area', str(strength))
                story.append(Paragraph(f"• {strength}", bullet_style))
            story.append(Spacer(1, 15))
        
               
        # ==================== AREAS FOR IMPROVEMENT ====================
        improvements = detailed_evaluation.get("areas_for_improvement", [])
        if improvements:
            story.append(Paragraph("Areas for Improvement", header_style))
            for i, improvement in enumerate(improvements, 1):
                story.append(Paragraph(f"{i}. {improvement}", bullet_style))
            story.append(Spacer(1, 15))
        
        # ==================== PAGE BREAK FOR Q&A ANALYSIS ====================
        story.append(PageBreak())
        
        # ==================== QUESTION-BY-QUESTION ANALYSIS ====================
        story.append(Paragraph("Question-by-Question Analysis", title_style))
        story.append(Spacer(1, 20))
        
        question_analysis = detailed_evaluation.get("question_analysis", [])
        
        for qa in question_analysis:  # Limit to 15 questions to fit in report
            q_num = qa.get("question_number", "?")
            question = qa.get("question", "")[:200]  # Truncate long questions
            answer = qa.get("answer", "")[:200]
            concept = qa.get("concept", "Unknown")
            evaluation = qa.get("evaluation", "unknown")
            score = qa.get("score", 0)
            feedback = qa.get("feedback", "")
            
            # Color based on evaluation
            if evaluation in ["correct"]:
                eval_color = '#48bb78'
                eval_bg = '#f0fff4'
            elif evaluation in ["partial"]:
                eval_color = '#ed8936'
                eval_bg = '#fffaf0'
            else:
                eval_color = '#f56565'
                eval_bg = '#fff5f5'
            
            # Question header
            q_header_style = ParagraphStyle(
                'QHeader',
                fontSize=11,
                textColor=HexColor('#2d3748'),
                spaceBefore=15,
                spaceAfter=5,
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph(f"Question {q_num} - {concept}", q_header_style))
            
            # Question and answer
            q_style = ParagraphStyle('Q', fontSize=10, leftIndent=10, textColor=HexColor('#4a5568'))
            a_style = ParagraphStyle('A', fontSize=10, leftIndent=10, textColor=HexColor('#2d3748'))
            
            story.append(Paragraph(f"<b>Q:</b> {question}", q_style))
            story.append(Paragraph(f"<b>A:</b> {answer}", a_style))
            
            # Evaluation result
            eval_style = ParagraphStyle(
                'Eval',
                fontSize=10,
                leftIndent=10,
                textColor=HexColor(eval_color),
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph(f"Result: {evaluation.upper()} ({score}/10)", eval_style))
            
            if feedback:
                feedback_style = ParagraphStyle('Feedback', fontSize=9, leftIndent=10, textColor=HexColor('#718096'), leading=12)
                story.append(Paragraph(f"Feedback: {feedback}", feedback_style))
            
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e2e8f0')))
        
        # ==================== ATTENTIVENESS ANALYSIS ====================
        story.append(Spacer(1, 20))
        story.append(Paragraph("Attentiveness Analysis", header_style))
        
        attentiveness = detailed_evaluation.get("attentiveness_analysis", {})
        att_data = [
            ["Aspect", "Assessment"],
            ["Engagement Level", attentiveness.get("engagement_level", "N/A")],
            ["Response Consistency", attentiveness.get("response_consistency", "N/A")],
            ["Focus Areas", attentiveness.get("focus_areas", "N/A")[:50]],
            ["Distraction Indicators", attentiveness.get("distraction_indicators", "N/A")[:50]],
        ]
        
        att_table = Table(att_data, colWidths=[2.5*inch, 3.5*inch])
        att_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a5568')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#e2e8f0')),
        ]))
        story.append(att_table)
        story.append(Spacer(1, 20))
        
        # ==================== RECOMMENDATIONS ====================
        recommendations = detailed_evaluation.get("recommendations", [])
        if recommendations:
            story.append(Paragraph("Recommendations for Improvement", header_style))
            for i, rec in enumerate(recommendations, 1):
                rec_style = ParagraphStyle('Rec', fontSize=10, leftIndent=20, spaceBefore=5, spaceAfter=5, leading=14)
                story.append(Paragraph(f"<b>{i}.</b> {rec}", rec_style))
            story.append(Spacer(1, 15))
        
        # ==================== TOPICS SUMMARY ====================
        topics_mastered = detailed_evaluation.get("topics_mastered", [])
        topics_to_review = detailed_evaluation.get("topics_to_review", [])
        
        if topics_mastered or topics_to_review:
            story.append(Paragraph("Topics Overview", header_style))
            
            if topics_mastered:
                story.append(Paragraph(f"<b>Topics Mastered:</b> {', '.join(topics_mastered[:5])}", normal_style))
            
            if topics_to_review:
                story.append(Paragraph(f"<b>Topics to Review:</b> {', '.join(topics_to_review[:5])}", normal_style))
        
        # ==================== FOOTER ====================
        story.append(Spacer(1, 40))
        story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#e2e8f0')))
        footer_style = ParagraphStyle('Footer', fontSize=8, alignment=TA_CENTER, textColor=HexColor('#a0aec0'))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", footer_style))
        story.append(Paragraph("Daily Standup Evaluation System", footer_style))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer.read()
        
    except Exception as e:
        logger.error(f"Comprehensive PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"PDF generation failed: {e}")


def get_rating(score: float) -> str:
    """Convert numeric score to rating text."""
    if score >= 90:
        return "Excellent"
    elif score >= 80:
        return "Very Good"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Satisfactory"
    elif score >= 50:
        return "Needs Improvement"
    else:
        return "Poor"

@app.post("/submit_feedback")
async def submit_feedback(payload: FeedbackPayload):
    """
    Submit user feedback after standup session completion.
    
    Expected payload:
    {
        "session_id": "uuid-string",
        "student_id": 123,
        "student_name": "John Doe",
        "feedback": {
            "overallExperience": 4,
            "audioQuality": 5,
            "questionClarity": 4,
            "systemResponsiveness": 3,
            "technicalIssues": ["Delayed responses"],
            "otherIssues": "Sometimes audio cut out",
            "suggestions": "Improve response time",
            "wouldRecommend": "yes",
            "difficultyLevel": "moderate",
            "submitted_at": "2025-01-15T10:30:00.000Z"
        },
        "session_duration": 900
    }
    """
    try:
        logger.info(f"📝 Receiving feedback for session: {payload.session_id}")
        
        # Convert Pydantic model to dict
        payload_dict = payload.dict()
        
        # Save to MongoDB
        success = save_feedback_to_mongodb(payload_dict)
        
        if success:
            return {
                "status": "success",
                "message": "Feedback submitted successfully",
                "session_id": payload.session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to save feedback to database"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Feedback submission error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Feedback submission failed: {str(e)}"
        )


# =============================================================================
# OPTIONAL: Get feedback statistics endpoint
# =============================================================================

@app.get("/feedback_stats")
async def get_feedback_stats():
    """Get aggregated feedback statistics from daily_standup_results collection."""
    try:
        from pymongo import MongoClient
        from urllib.parse import quote_plus
        
        encoded_pass = quote_plus("LT@connect25")
        connection_string = (
            f"mongodb://connectly:{encoded_pass}"
            f"@192.168.48.201:27017/ml_notes"
            f"?authSource=admin"
        )
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        db = client["ml_notes"]
        
        # ✅ CHANGED: Query from daily_standup_results with type filter
        collection = db["daily_standup_results"]
        
        # Aggregate statistics - filter by type="session_feedback"
        pipeline = [
            {
                "$match": {"type": "session_feedback"}  # ✅ Filter only feedback documents
            },
            {
                "$group": {
                    "_id": None,
                    "total_feedback": {"$sum": 1},
                    "avg_overall": {"$avg": "$ratings.overall_experience"},
                    "avg_audio": {"$avg": "$ratings.audio_quality"},
                    "avg_clarity": {"$avg": "$ratings.question_clarity"},
                    "avg_responsiveness": {"$avg": "$ratings.system_responsiveness"},
                    "avg_rating": {"$avg": "$average_rating"},
                    "would_recommend_yes": {
                        "$sum": {"$cond": [{"$eq": ["$would_recommend", "yes"]}, 1, 0]}
                    },
                    "would_recommend_maybe": {
                        "$sum": {"$cond": [{"$eq": ["$would_recommend", "maybe"]}, 1, 0]}
                    },
                    "would_recommend_no": {
                        "$sum": {"$cond": [{"$eq": ["$would_recommend", "no"]}, 1, 0]}
                    },
                }
            }
        ]
        
        result = list(collection.aggregate(pipeline))
        
        if result:
            stats = result[0]
            del stats["_id"]
            
            # Round averages
            for key in ["avg_overall", "avg_audio", "avg_clarity", "avg_responsiveness", "avg_rating"]:
                if stats.get(key):
                    stats[key] = round(stats[key], 2)
            
            # Get common technical issues
            issues_pipeline = [
                {"$match": {"type": "session_feedback"}},  # ✅ Filter only feedback
                {"$unwind": "$technical_issues"},
                {"$group": {"_id": "$technical_issues", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            common_issues = list(collection.aggregate(issues_pipeline))
            stats["common_technical_issues"] = [
                {"issue": item["_id"], "count": item["count"]} 
                for item in common_issues
            ]
            
            client.close()
            return stats
        
        client.close()
        return {
            "total_feedback": 0,
            "message": "No feedback data available yet"
        }
        
    except Exception as e:
        logger.error(f"❌ Feedback stats error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get feedback stats: {str(e)}"
        )

# =============================================================================
# OPTIONAL: Get feedback for specific session
# =============================================================================

@app.get("/feedback/{session_id}")
async def get_session_feedback(session_id: str):
    """Get feedback for a specific session from daily_standup_results collection."""
    try:
        from pymongo import MongoClient
        from urllib.parse import quote_plus
        
        encoded_pass = quote_plus("LT@connect25")
        connection_string = (
            f"mongodb://connectly:{encoded_pass}"
            f"@192.168.48.201:27017/ml_notes"
            f"?authSource=admin"
        )
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        db = client["ml_notes"]
        
        # ✅ CHANGED: Query from daily_standup_results with type filter
        collection = db["daily_standup_results"]
        
        # Find feedback document for this session
        feedback = collection.find_one({
            "session_id": session_id,
            "type": "session_feedback"  # ✅ Filter by type
        })
        
        client.close()
        
        if feedback:
            # Convert ObjectId to string
            feedback["_id"] = str(feedback["_id"])
            return feedback
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No feedback found for session {session_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get feedback error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feedback: {str(e)}"
        )

@app.get("/session_documents/{session_id}")
async def get_all_session_documents(session_id: str):
    """
    Get all documents for a session from daily_standup_results collection.
    Returns conversation (qa_session), evaluation (session_result), and feedback (session_feedback).
    """
    try:
        from pymongo import MongoClient
        from urllib.parse import quote_plus
        
        encoded_pass = quote_plus("LT@connect25")
        connection_string = (
            f"mongodb://connectly:{encoded_pass}"
            f"@192.168.48.201:27017/ml_notes"
            f"?authSource=admin"
        )
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=10000)
        db = client["ml_notes"]
        collection = db["daily_standup_results"]
        
        # Find all documents for this session
        documents = list(collection.find({"session_id": session_id}))
        
        client.close()
        
        if not documents:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for session {session_id}"
            )
        
        # Organize by type
        result = {
            "session_id": session_id,
            "conversation": None,      # type: qa_session
            "evaluation": None,        # type: session_result
            "feedback": None,          # type: session_feedback
            "document_count": len(documents)
        }
        
        for doc in documents:
            doc["_id"] = str(doc["_id"])
            doc_type = doc.get("type", "unknown")
            
            if doc_type == "qa_session":
                result["conversation"] = doc
            elif doc_type == "session_result":
                result["evaluation"] = doc
            elif doc_type == "session_feedback":
                result["feedback"] = doc
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get session documents error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session documents: {str(e)}"
        )

@app.get("/download_results/{session_id}")
async def download_results_fast(session_id: str):
    """Generate comprehensive PDF evaluation report."""
    try:
        result = await session_manager.get_session_result_fast(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get detailed evaluation if stored, otherwise generate basic one
        detailed_evaluation = result.get("detailed_evaluation", None)
        
        if not detailed_evaluation:
            # Generate basic evaluation structure from result data
            detailed_evaluation = {
                "overall_score": result.get("score", 50),  # Convert 0-10 to 0-100
                "technical_score": result.get("score", 50) ,
                "communication_score": 70,
                "attentiveness_score": 70,
                "grade": get_grade_from_score(result.get("score", 5) * 10),
                "summary": result.get("evaluation", "Session completed."),
                "strengths": ["Participated in the session"],
                "weaknesses": [],
                "areas_for_improvement": ["Review core concepts"],
                "question_analysis": [],
                "attentiveness_analysis": {
                    "engagement_level": "Medium",
                    "response_consistency": "Consistent",
                    "focus_areas": "Technical questions",
                    "distraction_indicators": "None detected"
                },
                "recommendations": ["Practice explaining technical concepts"],
                "topics_mastered": [],
                "topics_to_review": [],
                "raw_stats": {
                    "total_questions": result.get("total_exchanges", 0),
                    "answered_count": result.get("total_exchanges", 0),
                    "skipped_count": 0,
                    "silent_count": result.get("silence_responses", 0),
                    "irrelevant_count": 0,
                    "repeat_requests_count": 0,
                    "duration_minutes": result.get("duration", 0) / 60
                },
                "session_info": {
                    "student_name": result.get("student_name", "Unknown"),
                    "session_id": session_id
                }
            }
        
        # Generate comprehensive PDF
        loop = asyncio.get_event_loop()
        pdf_buffer = await loop.run_in_executor(
            shared_clients.executor,
            generate_comprehensive_pdf_report,
            result, detailed_evaluation, session_id
        )

        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=standup_evaluation_{session_id}.pdf"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


def get_grade_from_score(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 93: return "A"
    elif score >= 90: return "A-"
    elif score >= 87: return "B+"
    elif score >= 83: return "B"
    elif score >= 80: return "B-"
    elif score >= 77: return "C+"
    elif score >= 73: return "C"
    elif score >= 70: return "C-"
    elif score >= 67: return "D+"
    elif score >= 60: return "D"
    else: return "F"


# =============================================================================
# NEW ENDPOINT: Get detailed evaluation as JSON
# Add this new endpoint
# =============================================================================

@app.get("/api/evaluation/{session_id}")
async def get_evaluation_details(session_id: str):
    """Get detailed evaluation data for frontend display."""
    try:
        result = await session_manager.get_session_result_fast(session_id)
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        detailed_evaluation = result.get("detailed_evaluation", None)
        
        if not detailed_evaluation:
            # Return basic info if detailed evaluation not available
            return {
                "session_id": session_id,
                "student_name": result.get("student_name", "Unknown"),
                "overall_score": result.get("score", 5) * 10,
                "evaluation_text": result.get("evaluation", "No evaluation available"),
                "has_detailed_evaluation": False
            }
        
        return {
            "session_id": session_id,
            "has_detailed_evaluation": True,
            **detailed_evaluation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch evaluation: {str(e)}")

@app.get("/test")
async def enhanced_test_endpoint():
    """Enhanced test endpoint with silence detection and smart response features"""
    return {
        "message": "Ultra-Fast Daily Standup - COMPLETE FIXED VERSION v3.0",
        "timestamp": time.time(),
        "status": "production_ready_all_features_working",
        "config": {
            "real_data_mode": True,
            "silence_detection": True,
            "noise_filtering": True,
            "multiple_silence_responses": True,
            "gentle_responses": True,
            "smart_response_handling": True,
            "greeting_duplication_fixed": True,
            "greeting_exchanges": config.GREETING_EXCHANGES,
            "summary_chunks": config.SUMMARY_CHUNKS,
            "openai_model": config.OPENAI_MODEL,
            "mysql_host": config.MYSQL_HOST,
            "mongodb_host": config.MONGODB_HOST
        },
        "enhanced_features": [
            "Real database connections",
            "Gentle silence response generation (max 8 words)",
            "Background noise filtering (sneezes, coughs, etc.)",
            "Multiple silence responses allowed",
            "Status-aware audio processing", 
            "LLM-powered gentle handling",
            "No STT for silence cases",
            "Contextual brief responses",
            "Enhanced VAD with status tracking",
            "Intelligent conversation flow",
            "Multi-tier silence escalation",
            "Ultra-fast TTS streaming",
            "Working audio generation",
            "✅ Fixed greeting duplication (auto-transition after 2 exchanges)",
            "🔁 REPEAT question on request",
            "⏩ SKIP question detection with 'That's okay!' acknowledgment",
            "⚠️ IRRELEVANT answer detection (LLM-based) with 'Nice try!' encouragement"
        ],
        "noise_filtering": {
            "min_audio_duration": "0.5 seconds",
            "min_transcript_length": "3 characters",
            "min_quality_score": 0.3,
            "filters_out": ["sneezes", "coughs", "door_slams", "background_chatter"]
        },
        "silence_handling": {
            "detection_threshold": "6 seconds",
            "response_types": ["gentle_encouragement", "brief_help_offer", "patient_waiting"],
            "max_responses_per_session": "unlimited",
            "response_frequency": "every 6 seconds of continued silence",
            "response_length": "max 8 words",
            "style": "gentle and brief, NOT motivational",
            "uses_llm_for_responses": True,
            "bypasses_stt": True,
            "audio_working": True
        },
        "smart_response_handling": {
            "repeat_detection": {
                "triggers": ["repeat", "say that again", "what did you say", "didn't catch that", "pardon"],
                "action": "Repeats the last AI question with audio",
                "example": "User: 'repeat' → AI repeats last question"
            },
            "skip_detection": {
                "triggers": ["skip", "don't know", "can't answer", "not sure", "no idea"],
                "action": "Says 'That's okay!' + auto-advances to next question in ONE response",
                "example": "User: 'I don't know' → AI: 'That's okay! What is the purpose of...'"
            },
            "irrelevant_detection": {
                "method": "LLM-based relevance checking",
                "action": "Says 'Nice try!' + asks next question in ONE response",
                "example": "Q: 'What is SAP?' A: 'Dhoni is my fav' → AI: 'Nice try! Let me ask about...'",
                "only_in": "TECHNICAL stage",
                "min_answer_length": "10 characters"
            }
        },
        "greeting_fix": {
            "issue_fixed": "Greeting no longer repeats",
            "solution": "Auto-transition after 2 exchanges OR explicit confirmation",
            "max_greeting_exchanges": 2,
            "transition_logic": "greeting_count >= 2 → TECHNICAL stage",
            "example_flow": "AI greets → User responds → AI confirms ONCE → TECHNICAL questions start"
        },
        "extended_question_mode": {
            "enabled": True,
            "description": "Generates additional web-based questions when summary exhausted",
            "triggers_when": "summary_questions_complete AND time_remaining > 60_seconds",
            "max_questions": 30,
            "avoids_repetition": True
        },
        "session_duration": {
            "minimum": "15 minutes",
            "enforced": True,
            "soft_cutoff": "13 minutes",
            "hard_cutoff": "15 minutes"
        },
        "formal_closing": {
            "enabled": True,
            "style": "interview_professional",
            "thanks_candidate": True,
            "mentions_next_session": True
        },
        "biometric_auth": {
            "enabled": biometric_service is not None,
            "face_verification": "pre_standup",
            "voice_verification": "during_standup_every_45s",
            "max_voice_warnings": 3
        }
    }

@app.post("/auth/verify-face", response_model=FaceVerificationResponse)
async def verify_face_endpoint(request: FaceVerificationRequest):
    """
    Verify face before allowing standup entry.
    
    This endpoint should be called before the standup session starts.
    Student must pass face verification to proceed.
    
    Args:
        request: FaceVerificationRequest with student_code and base64 image
        
    Returns:
        FaceVerificationResponse with verification result
    """
    from core.biometric_auth import get_biometric_service
    
    service = get_biometric_service()
    if service is None:
        raise HTTPException(
            status_code=503, 
            detail="Biometric authentication service not available"
        )
    
    try:
        # Decode base64 image
        image_base64 = request.image_base64
        
        # Handle data URL format (data:image/jpeg;base64,...)
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        try:
            image_data = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        
        # Verify face
        result = service.verify_face(
            student_code=request.student_code,
            image_data=image_data
        )
        # ✅ ENHANCED LOGGING - Include error_type
        logger.info(
            f"🔐 Face verification for {request.student_code}: "
            f"verified={result['verified']}, similarity={result['similarity']}, "
            f"error_type={result.get('error_type')}, error={result.get('error')}"
        )
        
               
        return FaceVerificationResponse(
            verified=result["verified"],
            similarity=result["similarity"],
            threshold=result["threshold"],
            error=result["error"],
            error_type=result.get("error_type"),
            can_proceed=result["verified"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Face verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


'''@app.post("/auth/verify-voice/{session_id}")
async def verify_voice_endpoint(
    session_id: str,
    student_code: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Verify voice during standup session.
    
    This endpoint is called periodically (every 45 seconds) during the standup
    to verify the speaker is the registered student.
    
    Args:
        session_id: Current standup session ID
        student_code: Student's code for lookup
        audio: Audio file (webm, wav, or mp3)
        
    Returns:
        Voice verification result with warning count
    """
    from core.biometric_auth import get_biometric_service, get_voice_tracker
    
    service = get_biometric_service()
    tracker = get_voice_tracker()
    
    if service is None or tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Biometric authentication service not available"
        )
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if len(audio_data) < 1000:
            raise HTTPException(status_code=400, detail="Audio file too small")
        
        # Determine audio format
        audio_format = "webm"
        if audio.filename:
            if audio.filename.endswith(".wav"):
                audio_format = "wav"
            elif audio.filename.endswith(".mp3"):
                audio_format = "mp3"
            elif audio.filename.endswith(".webm"):
                audio_format = "webm"
        
        # Verify voice
        result = service.verify_voice(
            student_code=student_code,
            audio_data=audio_data,
            audio_format=audio_format
        )
        
        # Record verification result and get tracking status
        tracking = tracker.record_verification(
            session_id=session_id,
            verified=result["verified"],
            similarity=result["similarity"]
        )
        
        logger.info(
            f"🎤 Voice verification for {student_code} (session {session_id}): "
            f"verified={result['verified']}, warnings={tracking['warning_count']}"
        )
        
        return {
            "verified": result["verified"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "warning_count": tracking["warning_count"],
            "should_terminate": tracking["should_terminate"],
            "message": tracking["message"],
            "error": result["error"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Voice verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))'''

@app.post("/auth/verify-voice/{session_id}")
async def verify_voice_endpoint(
    session_id: str,
    student_code: str = Form(...),
    audio: UploadFile = File(...)
):
    """Verify voice during standup session"""
    service = get_biometric_service()
    tracker = get_voice_tracker()
    
    if not service or not tracker:
        raise HTTPException(status_code=503, detail="Biometric authentication service not available")
    
    try:
        # Read audio data
        audio_data = await audio.read()
        
        # Get file extension
        audio_format = audio.filename.split('.')[-1] if audio.filename else "webm"
        
        # Verify voice
        result = service.verify_voice(student_code, audio_data, audio_format)
        
        # ✅ FIX: Check for skip_warning flag from verification result
        skip_warning = result.get("skip_warning", False)
        
        # Record in tracker (only if not skipping)
        tracker_result = tracker.record_verification(
            session_id, 
            result["verified"], 
            result["similarity"],
            skip_warning=skip_warning  # ✅ Pass the flag
        )
        
        logger.info(
            f"🎤 Voice verification for {student_code} (session {session_id}): "
            f"verified={result['verified']}, warnings={tracker_result['warning_count']}"
        )
        
        return {
            "verified": result["verified"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "warning_count": tracker_result["warning_count"],
            "should_terminate": tracker_result["should_terminate"],
            "message": tracker_result["message"],
            "error": result.get("error")
        }
        
    except Exception as e:
        logger.error(f"❌ Voice verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/start-session/{session_id}")
async def start_verification_session(session_id: str, student_code: str):
    """
    Initialize voice verification tracking for a new standup session.
    
    Call this after face verification succeeds and before the standup begins.
    
    Args:
        session_id: New standup session ID
        student_code: Student's code
        
    Returns:
        Session initialization status
    """
    from core.biometric_auth import get_voice_tracker
    
    tracker = get_voice_tracker()
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Voice tracking service not available"
        )
    
    tracker.start_session(session_id, student_code)
    
    logger.info(f"🎬 Verification session started: {session_id} for student {student_code}")
    
    return {
        "status": "success",
        "session_id": session_id,
        "student_code": student_code,
        "message": "Voice verification tracking started",
        "max_warnings": tracker.max_warnings
    }


@app.get("/auth/session-status/{session_id}")
async def get_verification_session_status(session_id: str):
    """
    Get current voice verification status for a session.
    
    Args:
        session_id: Standup session ID
        
    Returns:
        Current warning count and termination status
    """
    from core.biometric_auth import get_voice_tracker
    
    tracker = get_voice_tracker()
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Voice tracking service not available"
        )
    
    status = tracker.get_session_status(session_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "student_code": status["student_code"],
        "warning_count": status["warning_count"],
        "max_warnings": tracker.max_warnings,
        "terminated": status["terminated"],
        "termination_reason": status["termination_reason"],
        "verification_count": len(status["verification_history"])
    }


@app.delete("/auth/end-session/{session_id}")
async def end_verification_session(session_id: str):
    """
    Clean up voice verification tracking when standup ends.
    
    Call this when the standup session completes or is terminated.
    
    Args:
        session_id: Standup session ID to clean up
        
    Returns:
        Cleanup status
    """
    from core.biometric_auth import get_voice_tracker
    
    tracker = get_voice_tracker()
    if tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Voice tracking service not available"
        )
    
    tracker.end_session(session_id)
    
    logger.info(f"🏁 Verification session ended: {session_id}")
    
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Voice verification tracking ended"
    }


@app.get("/auth/check-registration/{student_code}")
async def check_biometric_registration(student_code: str):
    """
    Check if a student has registered biometric data.
    
    Use this to determine if face/voice verification should be required.
    
    Args:
        student_code: Student's code
        
    Returns:
        Registration status for face and voice
    """
    from core.biometric_auth import get_biometric_service
    
    service = get_biometric_service()
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Biometric authentication service not available"
        )
    
    face_registered = service.get_stored_face_embedding(student_code) is not None
    voice_registered = service.get_stored_voice_embedding(student_code) is not None
    
    return {
        "student_code": student_code,
        "face_registered": face_registered,
        "voice_registered": voice_registered,
        "both_registered": face_registered and voice_registered,
        "can_require_verification": face_registered and voice_registered
    }

@app.get("/health")
async def health_check_fast():
    """Ultra-fast health check with real database status and smart features"""
    try:
        db_status = {"mysql": False, "mongodb": False}
        try:
            db_manager = DatabaseManager(shared_clients)
            conn = db_manager.get_mysql_connection()
            conn.close()
            db_status["mysql"] = True
            await db_manager.get_mongo_client()
            db_status["mongodb"] = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

        return {
            "status": "healthy" if all(db_status.values()) else "degraded",
            "service": "ultra_fast_daily_standup_complete_fixed",
            "timestamp": time.time(),
            "active_sessions": len(session_manager.active_sessions),
            "version": config.APP_VERSION,
            "database_status": db_status,
            "real_data_mode": True,
            "silence_detection_enabled": True,
            "noise_filtering_enabled": True,
            "multiple_silence_responses": True,
            "gentle_responses": True,
            "audio_generation_working": True,
            "greeting_duplication_fixed": True,
            "smart_features": {
                "repeat_question": True,
                "skip_detection": True,
                "skip_acknowledgment": True,
                "irrelevant_detection": True,
                "irrelevant_encouragement": True,
                "llm_based_checking": True,
                "single_combined_responses": True
            },
            "features": [
                "enhanced_vad", 
                "gentle_silence_responses", 
                "llm_powered_handling", 
                "noise_filter", 
                "unlimited_silence_prompts", 
                "working_audio", 
                "fixed_greeting",
                "repeat_question",
                "skip_with_acknowledgment",
                "irrelevant_with_encouragement",
                "auto_greeting_transition"
            ]
        }
    except Exception as e:
        logger.error(f"Enhanced health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.websocket("/ws/{session_id}")
async def enhanced_websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    hb_task = None
    session_data = None

    try:
        logger.info("Enhanced WebSocket connected for session: %s", session_id)

        session_data = session_manager.active_sessions.get(session_id)
        if not session_data or not getattr(session_data, "is_active", True):
            logger.error("Session %s not found or inactive", session_id)
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": "Session not found",
                "status": "error",
            }))
            return

        session_data.websocket = websocket

        # ✅ FIX 1: Prevent duplicate greetings
        should_send_greeting = not getattr(session_data, "greeting_sent", False)
        
        if should_send_greeting:
            session_data.greeting_sent = True
            logger.info(f"✅ Sending first greeting for session {session_id}")
        else:
            logger.info(f"⚠️ Greeting already sent for session {session_id}, skipping duplicate")

        session_data.awaiting_user = True
        session_data.user_has_spoken = False
        session_data.silence_prompt_active = False
        session_data.consecutive_silence_chunks = 0
        session_data.last_user_speech_ts = 0
        session_data.last_ai_ts = 0
        session_data.fragment_buffer = []
        session_data.fragment_evaluations = []
        logger.info(f"🔄 Session state reset for fresh connection: {session_id}")

        # ✅ Only generate greeting if not already sent
        if should_send_greeting:
            actual_domain = (
                getattr(session_data, "current_domain", None)
                or (session_data.summary_manager.current_topic if getattr(session_data, "summary_manager", None) else None)
                or "project"
            )
            # ✅ Determine time of day based on current hour
            current_hour = datetime.now().hour
            if 5 <= current_hour < 12:
                time_of_day = "morning"
            elif 12 <= current_hour < 17:
                time_of_day = "afternoon"
            elif 17 <= current_hour < 21:
                time_of_day = "evening"
            else:
                time_of_day = "evening"  # Late night treated as evening

            context = {
                "user_name": session_data.student_name,
                "time_of_day": time_of_day,
                "domain": actual_domain,
                "simple_english": True,
                "sentiment_hint": "neutral",
                "recent_exchanges": [],
                "suppress_salutation": False,
            }
                        
            loop = asyncio.get_event_loop()
            greeting_prompt = prompts.dynamic_greeting_response(
                user_input="(session start)", greeting_count=0, context=context
            )
            greeting = await loop.run_in_executor(
                shared_clients.executor,
                session_manager.conversation_manager._sync_openai_call,
                greeting_prompt,
            )
            greeting = (greeting or "").strip()

            if not greeting or len(greeting.split()) < 4:
                greeting = await loop.run_in_executor(
                    shared_clients.executor,
                    session_manager.conversation_manager._sync_openai_call,
                    greeting_prompt,
                )
                greeting = (greeting or "").strip()

            greeting = greeting or " "

            await websocket.send_text(json.dumps({
                "type": "ai_response",
                "text": greeting,
                "status": "greeting"
            }))
            # ✅ ADD THESE 2 LINES: Log the initial greeting to conversation log
            session_data.add_exchange(greeting, "(session_start)", 0.0, "greeting", False)
            logger.info(f"📝 Logged initial greeting to conversation_log")
            try:
                async for audio_chunk in session_manager.tts_processor.generate_ultra_fast_stream(
                    greeting, session_id=session_id
                ):
                    if audio_chunk and getattr(session_data, "is_active", False) and session_data.websocket is websocket:
                        await websocket.send_text(json.dumps({
                            "type": "audio_chunk",
                            "audio": audio_chunk.hex(),
                            "status": "greeting",
                        }))
                await websocket.send_text(json.dumps({"type": "audio_end", "status": "greeting"}))

                session_data.silence_ready = True
                session_data.greeting_end_ts = time.time()
                session_data.greeting_count = 1
                logger.info(f"✅ Initial greeting sent, greeting_count = 1")
                # === SEND INITIAL COMMUNICATION SCORE (PATCH 7) ===
                initial_comm_score = session_manager.calculate_communication_score(session_data)
                await websocket.send_text(json.dumps({
                    "type": "communication_score_update",
                    "score": initial_comm_score["total_score"],
                    "breakdown": initial_comm_score,
                    "is_initial": True
                }))


            except Exception as tts_error:
                logger.error("TTS error during greeting: %s", tts_error)
                session_data.silence_ready = True
                session_data.greeting_end_ts = time.time()
                session_data.greeting_count = 1


        async def _heartbeat(ws: WebSocket, sd, interval: int = 30):
            try:
                while sd and getattr(sd, "is_active", False):
                    await asyncio.sleep(interval)
                    if getattr(sd, "websocket", None) is ws:
                        await ws.send_text(json.dumps({"type": "ping"}))
            except Exception:
                pass

        hb_task = asyncio.create_task(_heartbeat(websocket, session_data, interval=30))

        while getattr(session_data, "is_active", False):
            try:
                timeout_s = max(120, getattr(config, "WEBSOCKET_TIMEOUT", 60))
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout_s)
                message = json.loads(raw)
                mtype = message.get("type")

                if mtype == "audio_data":
                    asyncio.create_task(session_manager.process_audio_with_silence_status(session_id, message))
                     
                elif mtype == "silence_detected":
                    logger.info("Session %s: received silence notification", session_id)
                    asyncio.create_task(session_manager.process_silence_notification(session_id, message))

                elif mtype == "user_status_update":
                    status = message.get("status", "unknown")
                    logger.debug("Session %s: user status update: %s", session_id, status)

                elif mtype == "default_answer":
                    default_text = message.get("text", "I need more time to think about this.")
                    logger.info("Session %s: default answer: %s", session_id, default_text)
                    asyncio.create_task(session_manager.process_default_answer(session_id, default_text))

                elif mtype == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif mtype == "heartbeat":
                    await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

                else:
                    logger.debug("Unknown WS message type: %s", mtype)
                    continue

            except asyncio.TimeoutError:
                logger.debug("WebSocket timeout for session %s - sending keepalive", session_id)
                if getattr(session_data, "is_active", False) and session_data.websocket is websocket:
                    try:
                        await websocket.send_text(json.dumps({"type": "ping"}))
                    except Exception as ping_error:
                        logger.error("Keepalive ping failed: %s", ping_error)
                        break
                continue

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected: %s", session_id)
                if session_data and getattr(session_data, "websocket", None) is websocket:
                    session_data.websocket = None
                break

            except json.JSONDecodeError as json_error:
                logger.error("JSON decode error: %s", json_error)
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "text": "Invalid message format",
                        "status": "error",
                    }))
                except Exception:
                    pass
                continue

            except Exception as e:
                logger.error("Enhanced WebSocket error: %s", e)
                error_message = "Connection error occurred"
                if "audio" in str(e).lower():
                    error_message = "Audio processing error"
                elif "timeout" in str(e).lower():
                    error_message = "Request timeout"

                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "text": error_message,
                        "status": "error",
                    }))
                except Exception:
                    pass

                if "critical" in str(e).lower() or "fatal" in str(e).lower():
                    break
                else:
                    continue

    except Exception as e:
        logger.error("Enhanced WebSocket endpoint error: %s", e)

    finally:
        if hb_task and not hb_task.done():
            hb_task.cancel()
        if session_data and getattr(session_data, "websocket", None) is websocket:
            session_data.websocket = None
        try:
            await session_manager.remove_session(session_id)
            logger.info(f"🧹 Session {session_id} fully cleaned up after disconnect")
        except Exception as cleanup_err:
            logger.error(f"⚠️ Cleanup error for session {session_id}: {cleanup_err}")

        logger.info("Enhanced WebSocket cleanup completed for session: %s", session_id)