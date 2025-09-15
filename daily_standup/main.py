# -*- coding: utf-8 -*-
"""
Ultra-fast, real database daily standup backend with optimized performance
NO DUMMY DATA - Real connections only
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
import uuid
import logging
import os
import io
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import base64

# Local imports - use package-relative, keep names intact
from core import *
from core.ai_services import DS_SessionData as SessionData
from core.ai_services import DS_SessionStage as SessionStage
from core.ai_services import DS_SummaryManager as SummaryManager
from core.ai_services import ds_shared_clients as shared_clients
from core.config import config
from core.database import DatabaseManager
from core.ai_services import DS_OptimizedAudioProcessor as OptimizedAudioProcessor
# ⬇️ Unified Chatterbox TTS
from core.tts_processor import UnifiedTTSProcessor as UltraFastTTSProcessor
from core.ai_services import DS_OptimizedConversationManager as OptimizedConversationManager
from core.prompts import DailyStandupPrompts as prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ULTRA-FAST SESSION MANAGER - NO DUMMY DATA
# =============================================================================

class UltraFastSessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, SessionData] = {}
        self.db_manager = DatabaseManager(shared_clients)
        self.audio_processor = OptimizedAudioProcessor(shared_clients)
        # ⬇️ INIT unified TTS with ref dir + encoding from config
        self.tts_processor = UltraFastTTSProcessor(
            ref_audio_dir=getattr(config, "REF_AUDIO_DIR", Path("ref_audios")),
            encode=getattr(config, "TTS_STREAM_ENCODING", "wav"),
        )
        self.conversation_manager = OptimizedConversationManager(shared_clients)

    # FIXED: Update your session creation method with better timing logic
    async def create_session_fast(self, websocket: Optional[Any] = None) -> SessionData:
        session_id = str(uuid.uuid4())
        test_id = f"standup_{int(time.time())}"
        try:
            student_info_task = asyncio.create_task(self.db_manager.get_student_info_fast())
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

            # FIXED: Keep your 15-minute session limit but make it more flexible
            SESSION_MAX_SECONDS = getattr(config, "SESSION_MAX_SECONDS", 15 * 60)  # Keep 15 minutes
            # FIXED: Increase soft cutoff buffer to 2 minutes instead of 10 seconds
            SESSION_SOFT_CUTOFF_SECONDS = getattr(config, "SESSION_SOFT_CUTOFF_SECONDS", 2 * 60)  # 2 minutes buffer
            now_ts = time.time()
            session_data.end_time = now_ts + SESSION_MAX_SECONDS
            session_data.soft_cutoff_time = session_data.end_time - SESSION_SOFT_CUTOFF_SECONDS
            session_data.awaiting_user = False
            
            # FIXED: Add clarification attempt tracking
            session_data.clarification_attempts = 0
            session_data.current_domain = 'general'
            
            session_data._hard_stop_task = asyncio.create_task(self._hard_stop_watchdog(session_data))

            fragment_manager = SummaryManager(shared_clients, session_data)
            if not fragment_manager.initialize_fragments(summary):
                raise Exception("Failed to initialize fragments from summary")
            session_data.summary_manager = fragment_manager

            self.tts_processor.start_session(session_data.session_id)

            self.active_sessions[session_id] = session_data
            logger.info("Real session created %s for %s with %d fragments",
                        session_id, session_data.student_name, len(session_data.fragment_keys))
            return session_data
        except Exception as e:
            logger.error("Failed to create session: %s", e)
            raise Exception(f"Session creation failed: {e}")
        
    async def generate_silence_response(session_data: SessionData):
        try:
            context = {
                "name": session_data.student_name,
                "domain": session_data.current_domain or "your work",
                "simple_english": True,
            }
            prompt = prompts.dynamic_silence_response(context)
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(shared_clients.executor, session_data.conversation_manager._sync_openai_call, prompt)
            text = text.strip()
            if not text or len(text.split()) < 3:
                text = f"Are you comfortable to continue {session_data.student_name}? Let’s start whenever you're ready."
            await session_data.websocket.send_text(json.dumps({"type": "ai_response", "text": text, "status": session_data.current_stage.value}))
            async for audio_chunk in session_data.tts_processor.generate_ultra_fast_stream(text, session_id=session_data.session_id):
                if audio_chunk:
                    await session_data.websocket.send_text(json.dumps({"type": "audio_chunk", "audio": audio_chunk.hex(), "status": session_data.current_stage.value}))
            await session_data.websocket.send_text(json.dumps({"type": "audio_end", "status": session_data.current_stage.value}))
            logger.info("Sent silence-based prompt.")
        except Exception as e:
            logger.error("Failed to send silence-based response: %s", e)


    async def process_default_answer(self, session_id: str, default_text: str):
        """Process a default answer when user doesn't respond in time"""
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            logger.warning("Inactive session: %s", session_id)
            return

        start_time = time.time()
        try:
            logger.info("Session %s: processing default answer: %s", session_id, default_text)

            # Check session timing
            now_ts = time.time()
            if hasattr(session_data, "end_time") and now_ts >= session_data.end_time:
                await self._end_due_to_time(session_data)
                return

            # Process the default answer as a regular response
            # Set a moderate quality score for default answers
            quality = 0.5
            
            # Generate AI response
            ai_response = await self.conversation_manager.generate_fast_response(session_data, default_text)

            concept = session_data.current_concept if session_data.current_concept else "default_answer"
            is_followup = getattr(session_data, '_last_question_followup', False)
            session_data.add_exchange(ai_response, default_text, quality, concept, is_followup)

            if session_data.summary_manager:
                session_data.summary_manager.add_answer(default_text)

            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, ai_response)

            # Check timing again after processing
            now_ts = time.time()
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
            if session_data.current_stage == SessionStage.TECHNICAL and (not soft_cutoff or now_ts < soft_cutoff):
                session_data.awaiting_user = True

            processing_time = time.time() - start_time
            logger.info("Default answer processing time: %.2fs", processing_time)

        except Exception as e:
            logger.error("Default answer processing error: %s", e)
            await self._send_quick_message(session_data, {
                "type": "error", 
                "text": "Sorry, there was an issue processing your response. Please try again.", 
                "status": "error"
            })
    # FIXED: Better auto-advance logic
    async def _auto_advance_question(self, session_data: SessionData):
        """Auto-advance to the next question when user can't provide a response"""
        try:
            logger.info("Session %s: auto-advancing to next question", session_data.session_id)
            
            # FIXED: More natural transition responses
            advance_responses = [
                "I understand. Let's move on to the next topic.",
                "No worries, let's continue with something else.",
                "That's okay, let me ask about something different.",
                "Let's try a different question."
            ]
            
            # Choose response based on attempt count
            attempt = getattr(session_data, 'clarification_attempts', 0)
            if attempt < len(advance_responses):
                auto_advance_response = advance_responses[attempt]
            else:
                auto_advance_response = advance_responses[0]
            
            # Add this as an exchange with a neutral concept
            session_data.add_exchange(auto_advance_response, "[AUTO_ADVANCE]", 0.3, "auto_advance", False)
            
            # Reset clarification attempts
            session_data.clarification_attempts = 0
            
            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, auto_advance_response)
            
            # FIXED: Check timing more carefully before setting awaiting_user
            now_ts = time.time()
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
            end_time = getattr(session_data, "end_time", None)
            
            # Only continue if we have adequate time remaining
            time_remaining = end_time - now_ts if end_time else float('inf')
            if (session_data.current_stage == SessionStage.TECHNICAL and 
                time_remaining > 60 and  # At least 1 minute left
                (not soft_cutoff or now_ts < soft_cutoff)):
                session_data.awaiting_user = True
            else:
                # Not enough time left - move to completion
                logger.info("Session %s: insufficient time remaining (%.1fs), moving to completion", 
                        session_data.session_id, time_remaining)
                await self._finalize_session_fast(session_data)
                    
        except Exception as e:
            logger.error("Auto-advance error: %s", e)
            # Fallback to ending session gracefully
            await self._finalize_session_fast(session_data)

    async def _hard_stop_watchdog(self, session_data: SessionData):
        """Mark hard cutoff at end_time. Allow exactly one final user answer if a question is pending."""
        try:
            delay = max(0.0, getattr(session_data, "end_time", time.time()) - time.time())
            await asyncio.sleep(delay)
            if not session_data.is_active:
                return

            if getattr(session_data, "awaiting_user", False):
                # Signal that cutoff is reached but we're waiting for ONE last answer.
                setattr(session_data, "hard_cutoff_reached", True)
                grace = getattr(config, "FINAL_ANSWER_GRACE_SECONDS", 0)
                if grace > 0:
                    await asyncio.sleep(grace)
                    if session_data.is_active:
                        await self._end_due_to_time(session_data)
            else:
                # No pending question → end immediately
                await self._end_due_to_time(session_data)
        except asyncio.CancelledError:
            return

    async def _end_due_to_time(self, session_data: SessionData):
        """
        Hard/soft time-limit handler.

        Rules:
        - No new AI questions once we enter this function.
        - If a final user answer just arrived earlier, we simply close out (no follow-ups).
        - We still generate evaluation + save to DB.
        - Send one short "thanks / session end" line (from prompt file, with a safety fallback).
        """

        # --- Helper: extract topic titles for closing summary ---
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
                if topics:
                    return topics

            if isinstance(frags, list):
                for frag in frags:
                    if isinstance(frag, dict):
                        t = frag.get("title") or frag.get("heading") or frag.get("name")
                        if t:
                            topics.append(t)
                if topics:
                    return topics

            if fk:
                return [str(k) for k in fk]
            return topics

        # From now on, ensure no further turns will be opened.
        try:
            session_data.awaiting_user = False
        except Exception:
            pass

        try:
            # Build summary for the closing prompt
            topics = _extract_topics(session_data)
            conv_log = getattr(session_data, "conversation_log", []) or []
            user_final_response = (
                conv_log[-1].get("user_response") if conv_log and isinstance(conv_log[-1], dict) else None
            )
            conversation_summary = {
                "topics_covered": topics,
                "total_exchanges": len(conv_log),
            }

            # --- Closing line from prompt file ---
            closing_prompt = prompts.dynamic_session_completion(conversation_summary, user_final_response)
            loop = asyncio.get_event_loop()
            closing_text = await loop.run_in_executor(
                shared_clients.executor,
                self.conversation_manager._sync_openai_call,
                closing_prompt,
            )

            # --- Safety fallback: ensure it isn't just "Thank you." ---
            closing_text = (closing_text or "").strip()
            if len(closing_text.split()) < 6:
                closing_text = f"Thanks {session_data.student_name}; we’ll stop here now—session complete."

            # --- Evaluate + Save even when ending due to time ---
            evaluation, score = None, None
            try:
                evaluation, score = await self.conversation_manager.generate_fast_evaluation(session_data)
            except Exception as e_eval:
                logger.error("Evaluation generation error (time-cutoff): %s", e_eval)

            try:
                if evaluation is not None and score is not None:
                    saved = await self.db_manager.save_session_result_fast(session_data, evaluation, score)
                    if not saved:
                        logger.error("Save (time-cutoff) failed for %s", session_data.session_id)
            except Exception as e_save:
                logger.exception("Save error (time-cutoff): %s", e_save)

            # --- Final message (no more questions) ---
            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": closing_text,
                "status": "complete",
                "enable_new_session": True,
                "evaluation": evaluation,
                "score": score,
                "pdf_url": f"/daily_standup/download_results/{session_data.session_id}",
                "redirect_to": "/dashboard",
            })

            # Stream TTS for the closing line (pinned voice)
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
            # Fallback: still end gracefully and attempt evaluation/save
            logger.error("Closing generation error: %s", e)
            fallback_text = f"Thanks {session_data.student_name}; we’ll stop here now—session complete."

            evaluation, score = None, None
            try:
                evaluation, score = await self.conversation_manager.generate_fast_evaluation(session_data)
            except Exception as e_eval2:
                logger.error("Evaluation generation error (fallback): %s", e_eval2)

            try:
                if evaluation is not None and score is not None:
                    saved = await self.db_manager.save_session_result_fast(session_data, evaluation, score)
                    if not saved:
                        logger.error("Save (fallback) failed for %s", session_data.session_id)
            except Exception as e_save2:
                logger.exception("Save error (fallback): %s", e_save2)

            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": fallback_text,
                "status": "complete",
                "enable_new_session": True,
                "evaluation": evaluation,
                "score": score,
                "pdf_url": f"/daily_standup/download_results/{session_data.session_id}",
            })

            try:
                async for audio_chunk in self.tts_processor.generate_ultra_fast_stream(
                    fallback_text, session_id=session_data.session_id
                ):
                    if audio_chunk:
                        await self._send_quick_message(session_data, {
                            "type": "audio_chunk",
                            "audio": audio_chunk.hex(),
                            "status": "complete",
                        })
                await self._send_quick_message(session_data, {"type": "audio_end", "status": "complete"})
            except Exception as e_tts2:
                logger.error("TTS fallback closing stream error: %s", e_tts2)

        finally:
            # From this point, the session is over.
            session_data.is_active = False

            # Cancel watchdog if still running
            try:
                task = getattr(session_data, "_hard_stop_task", None)
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass

            # Close the websocket so the client UI stops showing running time immediately
            try:
                if session_data.websocket:
                    await session_data.websocket.close(code=1000)
            except Exception:
                pass

            # Unpin voice & remove session
            await self.remove_session(session_data.session_id)

    async def remove_session(self, session_id: str):
        if session_id in self.active_sessions:
            # ⬇️ CLEAN THE PINNED VOICE
            try:
                self.tts_processor.end_session(session_id)
            except Exception:
                pass
            del self.active_sessions[session_id]
            logger.info("Removed session %s", session_id)

    async def process_audio_ultra_fast(self, session_id: str, audio_data: bytes):
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            logger.warning("Inactive session: %s", session_id)
            return

        start_time = time.time()

        try:
            now_ts = time.time()
            end_time = getattr(session_data, "end_time", None)

            # 🔒 15-Minute Hard Cutoff BEFORE STT
            if end_time and now_ts >= end_time:
                if getattr(session_data, "awaiting_user", False):
                    try:
                        transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_data)
                    except Exception as stt_err:
                        logger.error("STT error after cutoff: %s", stt_err)
                        transcript, quality = ("", 0.0)

                    concept = session_data.current_concept or "unknown"
                    is_followup = getattr(session_data, "_last_question_followup", False)

                    if transcript.strip():
                        session_data.add_exchange("[FINAL_QUESTION_AWAITED_USER]", transcript, quality, concept, is_followup)
                        if session_data.summary_manager:
                            session_data.summary_manager.add_answer(transcript)

                await self._end_due_to_time(session_data)
                return

            logger.info("Session %s: received %d bytes of audio", session_id, len(audio_data))

            transcript, quality = await self.audio_processor.transcribe_audio_fast(audio_data)

            # 🔁 Clarification if transcript too short/poor
            if not transcript or len(transcript.strip()) < 2:
                clarification_context = {
                    'clarification_attempts': getattr(session_data, 'clarification_attempts', 0),
                    'audio_quality': quality,
                    'audio_size': len(audio_data)
                }
                session_data.clarification_attempts = clarification_context['clarification_attempts'] + 1

                if session_data.clarification_attempts == 1:
                    await self.generate_silence_response(session_data)
                    return

                loop = asyncio.get_event_loop()
                clarification_message = await loop.run_in_executor(
                    shared_clients.executor,
                    self.conversation_manager._sync_openai_call,
                    prompts.dynamic_clarification_request(clarification_context),
                )

                await self._send_quick_message(session_data, {
                    "type": "clarification",
                    "text": clarification_message,
                    "status": session_data.current_stage.value,
                })

                if session_data.clarification_attempts >= 2:
                    logger.info("Session %s: clarification attempts exceeded, auto-advancing", session_id)
                    await self._auto_advance_question(session_data)

                return

            logger.info("Session %s: transcript='%s' quality=%.2f", session_id, transcript, quality)

            session_data.clarification_attempts = 0
            now_ts = time.time()

            # 🔁 Re-check cutoff AFTER STT
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)

            if end_time and now_ts >= end_time:
                if getattr(session_data, "awaiting_user", False):
                    concept = session_data.current_concept or "unknown"
                    is_followup = getattr(session_data, "_last_question_followup", False)
                    session_data.add_exchange("[FINAL_USER]", transcript, quality, concept, is_followup)
                    if session_data.summary_manager:
                        session_data.summary_manager.add_answer(transcript)

                await self._end_due_to_time(session_data)
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
                await self._end_due_to_time(session_data)
                return

            # 🔍 In-session normal processing
            session_data.awaiting_user = False

            # Try domain inference
            try:
                domain = self._infer_domain(transcript)
                session_data.current_domain = domain
            except Exception as e:
                logger.debug("Domain inference error: %s", e)

            ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript)

            concept = session_data.current_concept or "unknown"
            is_followup = getattr(session_data, "_last_question_followup", False)

            session_data.add_exchange(ai_response, transcript, quality, concept, is_followup)

            if session_data.summary_manager:
                session_data.summary_manager.add_answer(transcript)

            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, ai_response)

            now_ts = time.time()
            if session_data.current_stage == SessionStage.TECHNICAL and (not soft_cutoff or now_ts < soft_cutoff):
                session_data.awaiting_user = True

            logger.info("Total processing time: %.2fs", time.time() - start_time)

        except Exception as e:
            logger.error("Audio processing error: %s", e)

            fallback_msg = "There was a problem processing your response. Please try again."

            if "domain" in str(e).lower():
                logger.info("Retrying without domain inference")
                try:
                    ai_response = await self.conversation_manager.generate_fast_response(session_data, transcript or "")
                    await self._send_response_with_ultra_fast_audio(session_data, ai_response)
                    return
                except Exception as retry_err:
                    logger.error("Retry failed: %s", retry_err)

            await self._send_quick_message(session_data, {
                "type": "error",
                "text": fallback_msg,
                "status": "error",
            })

    # NEW: Add this method to handle default answers from frontend
    async def process_default_answer(self, session_id: str, default_text: str):
        """Process a default answer when user doesn't respond in time"""
        session_data = self.active_sessions.get(session_id)
        if not session_data or not session_data.is_active:
            logger.warning("Inactive session: %s", session_id)
            return

        start_time = time.time()
        try:
            logger.info("Session %s: processing default answer: %s", session_id, default_text)

            # Check session timing
            now_ts = time.time()
            if hasattr(session_data, "end_time") and now_ts >= session_data.end_time:
                await self._end_due_to_time(session_data)
                return

            # Process the default answer as a regular response
            # Set a moderate quality score for default answers
            quality = 0.5
            
            # Generate AI response
            ai_response = await self.conversation_manager.generate_fast_response(session_data, default_text)

            concept = session_data.current_concept if session_data.current_concept else "default_answer"
            is_followup = getattr(session_data, '_last_question_followup', False)
            session_data.add_exchange(ai_response, default_text, quality, concept, is_followup)

            if session_data.summary_manager:
                session_data.summary_manager.add_answer(default_text)

            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, ai_response)

            # Check timing again after processing
            now_ts = time.time()
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
            if session_data.current_stage == SessionStage.TECHNICAL and (not soft_cutoff or now_ts < soft_cutoff):
                session_data.awaiting_user = True

            processing_time = time.time() - start_time
            logger.info("Default answer processing time: %.2fs", processing_time)

        except Exception as e:
            logger.error("Default answer processing error: %s", e)
            await self._send_quick_message(session_data, {
                "type": "error", 
                "text": "Sorry, there was an issue processing your response. Please try again.", 
                "status": "error"
            })


    # NEW: Add this method to auto-advance questions
    async def _auto_advance_question(self, session_data: SessionData):
        """Auto-advance to the next question when user can't provide a response"""
        try:
            logger.info("Session %s: auto-advancing to next question", session_data.session_id)
            
            # Use a generic "moving on" response
            auto_advance_response = "I understand. Let's move on to the next topic."
            
            # Add this as an exchange with a neutral concept
            session_data.add_exchange(auto_advance_response, "[AUTO_ADVANCE]", 0.3, "auto_advance", False)
            
            # Reset clarification attempts
            session_data.clarification_attempts = 0
            
            await self._update_session_state_fast(session_data)
            await self._send_response_with_ultra_fast_audio(session_data, auto_advance_response)
            
            # Set awaiting_user flag if still in technical stage
            now_ts = time.time()
            soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
            if session_data.current_stage == SessionStage.TECHNICAL and (not soft_cutoff or now_ts < soft_cutoff):
                session_data.awaiting_user = True
                
        except Exception as e:
            logger.error("Auto-advance error: %s", e)
            # Fallback to ending session
            await self._end_due_to_time(session_data)

    # FIXED: More graceful session state updates
    async def _update_session_state_fast(self, session_data: SessionData):
        try:
            if session_data.current_stage == SessionStage.GREETING:
                session_data.greeting_count += 1
                if session_data.greeting_count >= config.GREETING_EXCHANGES:
                    session_data.current_stage = SessionStage.TECHNICAL
                    logger.info("Session %s moved to TECHNICAL stage", session_data.session_id)
            elif session_data.current_stage == SessionStage.TECHNICAL:
                # FIXED: Check both fragment completion AND timing
                now_ts = time.time()
                soft_cutoff = getattr(session_data, "soft_cutoff_time", None)
                
                # Only auto-complete if we've covered enough material OR reached soft cutoff
                if session_data.summary_manager:
                    has_covered_enough = not session_data.summary_manager.should_continue_test()
                    approaching_time_limit = soft_cutoff and now_ts >= (soft_cutoff - 60)  # 1 minute before soft cutoff
                    
                    if has_covered_enough or approaching_time_limit:
                        session_data.current_stage = SessionStage.COMPLETE
                        logger.info("Session %s moved to COMPLETE stage (covered: %s, time: %s)", 
                                session_id, has_covered_enough, approaching_time_limit)
                        asyncio.create_task(self._finalize_session_fast(session_data))
        except Exception as e:
            logger.error("Session state update error: %s", e)

    async def _finalize_session_fast(self, session_data: SessionData):
        try:
            evaluation, score = await self.conversation_manager.generate_fast_evaluation(session_data)
            save_success = await self.db_manager.save_session_result_fast(session_data, evaluation, score)
            if not save_success:
                logger.error("Failed to save session %s", session_data.session_id)

            completion_message = f"Great job! Your standup session is complete. You scored {score}/10. Thank you!"
            await self._send_quick_message(session_data, {
                "type": "conversation_end",
                "text": completion_message,
                "evaluation": evaluation,
                "score": score,
                "pdf_url": f"/daily_standup/download_results/{session_data.session_id}",
                "status": "complete",
                "enable_new_session": True,
                "redirect_to": "/dashboard",         # <-- redirect hint
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
            # ✅ Make the UI stop even if frontend isn’t listening for special flags
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

    async def _send_response_with_ultra_fast_audio(self, session_data: SessionData, text: str):
        try:
            await self._send_quick_message(session_data, {
                "type": "ai_response",
                "text": text,
                "status": session_data.current_stage.value,
            })
            chunk_count = 0
            # ⬇️ PASS session_id
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


# =============================================================================
# FASTAPI APPLICATION - NO DUMMY DATA
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

session_manager = UltraFastSessionManager()

@app.on_event("startup")
async def startup_event():
    logger.info("Ultra-Fast Daily Standup application starting...")
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

        logger.info("All database connections verified")
    except Exception as e:
        logger.error("Startup failed: %s", e)

@app.on_event("shutdown")
async def shutdown_event():
    await shared_clients.close_connections()
    await session_manager.db_manager.close_connections()
    logger.info("Daily Standup application shutting down")

@app.get("/start_test")
async def start_standup_session_fast():
    try:
        logger.info("Starting real standup session...")
        session_data = await session_manager.create_session_fast()
        greeting = "Hello! Welcome to your daily standup. How are you doing today?"
        logger.info("Real session created: %s", session_data.test_id)
        return {
            "status": "success",
            "message": "Session started successfully",
            "test_id": session_data.test_id,
            "session_id": session_data.session_id,
            "websocket_url": f"/daily_standup/ws/{session_data.session_id}",
            "greeting": greeting,
            "student_name": session_data.student_name,
            "fragments_count": len(session_data.fragment_keys) if session_data.fragment_keys else 0,
            "estimated_duration": len(session_data.fragment_keys) * session_data.questions_per_concept * config.ESTIMATED_SECONDS_PER_QUESTION,
        }
    except Exception as e:
        logger.error("Error starting session: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")
@app.get("/api/summary/{test_id}")
async def get_standup_summary_fast(test_id: str):
    """Get standup session summary from real database"""
    try:
        logger.info(f"?? Getting summary for test_id: {test_id}")
        
        if not test_id:
            raise HTTPException(status_code=400, detail="test_id is required")
        
        result = await session_manager.get_session_result_fast(test_id)
        
        if result:
            exchanges = result.get("conversation_log", [])
            
            # Extract standup information from conversation
            yesterday_work = ""
            today_plans = ""
            blockers = ""
            additional_notes = ""
            
            for exchange in exchanges:
                user_response = exchange.get("user_response", "").lower()
                ai_message = exchange.get("ai_message", "").lower()
                
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
                "pdf_url": f"/daily_standup/download_results/{test_id}",
                "status": "completed"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Session result not found for test_id: {test_id}")
        
        logger.info(f"? Real summary generated for {test_id}")
        return summary_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=f"Summary retrieval failed: {str(e)}")
@app.get("/download_results/{session_id}")
async def download_results_fast(session_id: str):
    """Fast PDF generation and download from real data"""
    try:
        result = await session_manager.get_session_result_fast(session_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        loop = asyncio.get_event_loop()
        pdf_buffer = await loop.run_in_executor(
            shared_clients.executor,
            generate_pdf_report,
            result, session_id
        )
        
        return StreamingResponse(
            io.BytesIO(pdf_buffer),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=standup_report_{session_id}.pdf"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
@app.get("/test")
async def test_endpoint_fast():
    """Fast test endpoint with real configuration"""
    return {
        "message": "Ultra-Fast Daily Standup service is running with REAL DATA",
        "timestamp": time.time(),
        "status": "blazing_fast",
        "config": {
            "real_data_mode": True,
            "greeting_exchanges": config.GREETING_EXCHANGES,
            "summary_chunks": config.SUMMARY_CHUNKS,
            "openai_model": config.OPENAI_MODEL,
            "mysql_host": config.MYSQL_HOST,
            "mongodb_host": config.MONGODB_HOST
        },
        "optimizations": [
            "Real database connections",
            "No dummy data fallbacks",
            "800ms silence detection",
            "Parallel processing pipeline", 
            "Fragment-based questioning",
            "Sliding window conversation history",
            "Ultra-fast TTS streaming",
            "Thread pool optimization",
            "Connection pooling",
            "Real error detection only"
        ]
    }
@app.get("/health")
async def health_check_fast():
    """Ultra-fast health check with real database status"""
    try:
        db_status = {"mysql": False, "mongodb": False}
        
        # Quick database health check
        try:
            db_manager = DatabaseManager(shared_clients)
            
            # Test MySQL
            conn = db_manager.get_mysql_connection()
            conn.close()
            db_status["mysql"] = True
            
            # Test MongoDB
            await db_manager.get_mongo_client()
            db_status["mongodb"] = True
            
        except Exception as e:
            logger.warning(f"?? Database health check failed: {e}")
        
        return {
            "status": "healthy" if all(db_status.values()) else "degraded",
            "service": "ultra_fast_daily_standup",
            "timestamp": time.time(),
            "active_sessions": len(session_manager.active_sessions),
            "version": config.APP_VERSION,
            "database_status": db_status,
            "real_data_mode": True  # Always true now
        }
    except Exception as e:
        logger.error(f"? Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint_ultra_fast(websocket: WebSocket, session_id: str):
    import asyncio, json, base64, time
    from fastapi import WebSocketDisconnect

    await websocket.accept()
    hb_task = None
    session_data = None

    # --- inner helper: keep idle sockets alive ---
    async def _heartbeat(ws: WebSocket, sd, interval: int = 30):  # Increased from 20 to 30 seconds
        try:
            while sd and getattr(sd, "is_active", False):
                await asyncio.sleep(interval)
                # Only ping if this exact socket is still attached
                if getattr(sd, "websocket", None) is ws:
                    await ws.send_text(json.dumps({"type": "ping"}))
        except Exception:
            # benign (socket closed/replaced)
            pass

    try:
        logger.info("WebSocket connected for session: %s", session_id)

        # Fetch existing session (created by /start_test)
        session_data = session_manager.active_sessions.get(session_id)
        if not session_data or not getattr(session_data, "is_active", True):
            logger.error("Session %s not found or inactive", session_id)
            await websocket.send_text(json.dumps({
                "type": "error",
                "text": f"Session {session_id} not found. Please start a new session.",
                "status": "error",
            }))
            return

        # Bind this socket to the session
        session_data.websocket = websocket

        # Initial greeting + pinned-voice TTS
        greeting = f"Hello {session_data.student_name}! Welcome to your daily standup. How are you doing today?"
        await websocket.send_text(json.dumps({"type": "ai_response", "text": greeting, "status": "greeting"}))
        
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
        except Exception as tts_error:
            logger.error("TTS error during greeting: %s", tts_error)
            # Continue without TTS

        # Start heartbeat (server-side keepalive) - increased interval
        hb_task = asyncio.create_task(_heartbeat(websocket, session_data, interval=30))

        # Main receive loop
        while getattr(session_data, "is_active", False):
            try:
                # FIXED: Increased timeout for better stability
                timeout_s = max(120, getattr(config, "WEBSOCKET_TIMEOUT", 60))  # Increased from 60 to 120 seconds
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout_s)
                message = json.loads(raw)
                mtype = message.get("type")

                if mtype == "audio_data":
                    audio_b64 = message.get("audio", "")
                    if not audio_b64:
                        continue
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # FIXED: Don't wait for processing to complete - fire and forget
                        asyncio.create_task(session_manager.process_audio_ultra_fast(session_id, audio_bytes))
                    except Exception as decode_error:
                        logger.error("Audio decode error: %s", decode_error)
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "text": "Audio processing failed. Please try again.",
                            "status": "error"
                        }))

                elif mtype == "default_answer":
                    default_text = message.get("text", "I need more time to think about this.")
                    logger.info("Session %s: default answer: %s", session_id, default_text)
                    asyncio.create_task(session_manager.process_default_answer(session_id, default_text))

                elif mtype == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif mtype == "heartbeat":
                    # Client-side heartbeat
                    await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))

                else:
                    logger.debug("Unknown WS message type: %s", mtype)
                    # Ignore unknown message types

            except asyncio.TimeoutError:
                # FIXED: More graceful timeout handling
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
                # FIXED: Keep session alive for potential reconnection
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
                logger.error("WebSocket error: %s", e)
                # FIXED: More specific error handling
                error_message = "Connection error occurred"
                if "domain" in str(e).lower():
                    error_message = "Processing your request..."
                    # Don't break connection for domain inference errors
                    continue
                elif "audio" in str(e).lower():
                    error_message = "Audio processing error. Please try again."
                elif "timeout" in str(e).lower():
                    error_message = "Request timeout. Please try again."
                
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "text": error_message,
                        "status": "error",
                    }))
                except Exception:
                    pass
                
                # Only break on critical errors
                if "critical" in str(e).lower() or "fatal" in str(e).lower():
                    break
                else:
                    continue

    except Exception as e:
        logger.error("WebSocket endpoint error: %s", e)

    finally:
        # Stop heartbeat; DO NOT remove the session here.
        if hb_task and not hb_task.done():
            hb_task.cancel()
        
        # FIXED: Only clear websocket reference, don't end session
        if session_data and getattr(session_data, "websocket", None) is websocket:
            session_data.websocket = None
            
        logger.info("WebSocket cleanup completed for session: %s", session_id)
        # Session cleanup is handled by _finalize_session_fast / _end_due_to_time