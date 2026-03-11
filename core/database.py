"""
Unified Database Management Module
==================================

This merges the three previous database.py implementations from:
- daily_standup
- weekend_mocktest
- weekly_interview

Handles:
- MySQL connections
- MongoDB connections (async + sync)
- Summaries, test results, interview results, session results
- Q&A storage in ml_notes.daily_standup_results
- HR question tracking (cross-session)

Each method retains its original name and behavior to avoid breaking imports.
"""

import os
import time
import random
import logging
import asyncio
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime, timedelta
import mysql.connector
from mysql.connector import errorcode
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus

import pymongo
from pymongo import MongoClient

from .config import config

logger = logging.getLogger(__name__)


# ============================================================================
# CORE CLASS
# ============================================================================
class DatabaseManager:
    """Unified Database Manager supporting Daily Standup, Mock Tests, and Weekly Interviews."""

    def __init__(self, client_manager=None):
        self.client_manager = client_manager
        self._mongo_client = None
        self._mongo_db = None

        # weekend_mocktest style mongo client
        self.mongo_client = None
        self.db = None
        self.summaries_collection = None
        self.test_results_collection = None

        # Q&A storage MongoDB client (ml_notes database) — Daily Standup
        self._qa_mongo_client = None
        self._qa_mongo_db = None
        self._qa_collection = None

        # HR questions cache — Weekly Interview
        self._hr_questions_cache = {}  # Dict to cache by candidate_type
        self._hr_questions_cache_time = None
        self._HR_CACHE_TTL = 3600  # Cache for 1 hour

    # ------------------------------------------------------------------------
    # CONFIGURATION PROPERTIES
    # ------------------------------------------------------------------------
    @property
    def mysql_config(self) -> Dict[str, Any]:
        return {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': config.MYSQL_DATABASE,
            'USER': config.MYSQL_USER,
            'PASSWORD': config.MYSQL_PASSWORD,
            'HOST': config.MYSQL_HOST,
            'PORT': config.MYSQL_PORT,
        }

    @property
    def mongo_config(self) -> Dict[str, Any]:
        return {
            "username": config.MONGODB_USERNAME,
            "password": config.MONGODB_PASSWORD,
            "host": config.MONGODB_HOST,
            "port": config.MONGODB_PORT,
            "database": config.MONGODB_DATABASE,
            "auth_source": config.MONGODB_AUTH_SOURCE,
        }

    # ========================================================================
    # MYSQL CONNECTION
    # ========================================================================
    def get_mysql_connection(self):
        """Get MySQL connection using environment configuration"""
        try:
            db_config = self.mysql_config
            conn = mysql.connector.connect(
                user=db_config['USER'],
                password=db_config['PASSWORD'],
                host=db_config['HOST'],
                database=db_config['NAME'],
                port=db_config['PORT'],
                connection_timeout=5
            )
            return conn

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                logger.error("❌ MySQL: Wrong username or password")
                raise Exception("MySQL authentication failed")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                logger.error(f"❌ MySQL: Database '{db_config['NAME']}' does not exist")
                raise Exception(f"MySQL database '{db_config['NAME']}' not found")
            else:
                logger.error(f"❌ MySQL connection error: {err}")
                raise Exception(f"MySQL connection failed: {err}")
        except Exception as e:
            logger.error(f"❌ MySQL connection failed: {e}")
            raise Exception(f"MySQL connection failed: {e}")

    # ========================================================================
    # ASYNC MONGO CONNECTION (used by daily_standup + weekly_interview)
    # ========================================================================
    async def get_mongo_client(self) -> AsyncIOMotorClient:
        """Get MongoDB client with connection pooling"""
        if self._mongo_client is None:
            mongo_cfg = self.mongo_config
            username = quote_plus(mongo_cfg['username'])
            password = quote_plus(mongo_cfg['password'])
            mongo_uri = f"mongodb://{username}:{password}@{mongo_cfg['host']}:{mongo_cfg['port']}/{mongo_cfg['auth_source']}"

            self._mongo_client = AsyncIOMotorClient(
                mongo_uri,
                maxPoolSize=config.MONGO_MAX_POOL_SIZE,
                serverSelectionTimeoutMS=config.MONGO_SERVER_SELECTION_TIMEOUT
            )

            try:
                await self._mongo_client.admin.command('ping')
                logger.info("✅ MongoDB client initialized and tested")
            except Exception as e:
                logger.error(f"❌ MongoDB connection failed: {e}")
                self._mongo_client = None
                raise Exception(f"MongoDB connection failed: {e}")

        return self._mongo_client

    async def get_mongo_db(self):
        """Get MongoDB database instance"""
        if self._mongo_db is None:
            client = await self.get_mongo_client()
            self._mongo_db = client[self.mongo_config["database"]]
        return self._mongo_db

    # ========================================================================
    #  DAILY STANDUP — Q&A Storage MongoDB (ml_notes.daily_standup_results)
    # ========================================================================
    def _init_qa_mongodb(self):
        """Initialize MongoDB connection for Q&A storage in ml_notes.daily_standup_results"""
        try:
            # Skip if already initialized
            if self._qa_mongo_client is not None:
                return

            # HARDCODED settings for ml_notes database
            qa_host = "192.168.48.201"
            qa_port = 27017
            qa_user = "connectly"
            qa_pass = "LT@connect25"
            qa_db = "ml_notes"
            qa_auth = "admin"
            qa_collection_name = "daily_standup_results"

            encoded_pass = quote_plus(qa_pass)
            qa_connection_string = (
                f"mongodb://{qa_user}:{encoded_pass}"
                f"@{qa_host}:{qa_port}/{qa_db}"
                f"?authSource={qa_auth}"
            )

            logger.info(f"🔌 Q&A: connecting to {qa_db}.{qa_collection_name}")

            self._qa_mongo_client = MongoClient(
                qa_connection_string,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
            )

            # Test connection
            self._qa_mongo_client.admin.command('ping')

            self._qa_mongo_db = self._qa_mongo_client[qa_db]
            self._qa_collection = self._qa_mongo_db[qa_collection_name]

            logger.info(f"✅ Q&A ready: {qa_db}.{qa_collection_name}")

        except Exception as e:
            logger.error(f"❌ Q&A MongoDB connection failed: {e}")
            self._qa_mongo_client = None
            self._qa_mongo_db = None
            self._qa_collection = None
            raise Exception(f"Q&A MongoDB connection failed: {e}")

    def save_qa_exchange(self, session_id: str, student_id: int, student_name: str,
                         ai_question: str, user_answer: str, concept: str,
                         is_followup: bool = False, quality_score: float = 0.0,
                         stage: str = "technical", test_id: str = None) -> bool:
        """Save a single Q&A exchange to MongoDB"""
        try:
            if self._qa_collection is None:
                self._init_qa_mongodb()

            qa_document = {
                "session_id": session_id,
                "test_id": test_id,
                "student_id": student_id,
                "student_name": student_name,
                "question": ai_question,
                "answer": user_answer,
                "concept": concept,
                "is_followup": is_followup,
                "quality_score": quality_score,
                "stage": stage,
                "timestamp": datetime.now(),
                "created_at": datetime.utcnow(),
                "type": "qa_exchange",
            }

            result = self._qa_collection.insert_one(qa_document)

            if result.inserted_id:
                logger.info(f"✅ Q&A saved: session={session_id[:8]}..., concept={concept}")
                return True
            else:
                logger.error(f"❌ Failed to save Q&A: no inserted_id returned")
                return False

        except Exception as e:
            logger.error(f"❌ Error saving Q&A to MongoDB: {e}")
            return False

    async def save_qa_exchange_async(self, session_id: str, student_id: int, student_name: str,
                                      ai_question: str, user_answer: str, concept: str,
                                      is_followup: bool = False, quality_score: float = 0.0,
                                      stage: str = "technical", test_id: str = None) -> bool:
        """Async wrapper for save_qa_exchange"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self.save_qa_exchange,
                session_id, student_id, student_name,
                ai_question, user_answer, concept,
                is_followup, quality_score, stage, test_id
            )
        except Exception as e:
            logger.error(f"❌ Async Q&A save error: {e}")
            return False

    def save_session_qa_batch(self, session_id: str, student_id: int, student_name: str,
                              conversation_log: list, test_id: str = None) -> bool:
        """Save conversation with CORRECT Q&A pairing"""
        try:
            if self._qa_collection is None:
                self._init_qa_mongodb()

            if not conversation_log:
                logger.warning("No conversation log to save")
                return True

            logger.info(f"📝 Processing {len(conversation_log)} exchanges for session {session_id}")

            paired_exchanges = []
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

                if not ai_message or len(ai_message.strip()) < 2:
                    continue

                # ✅ GET ANSWER FROM NEXT EXCHANGE (this is the fix!)
                user_answer = ""
                quality_score = 0.0

                if idx + 1 < len(conversation_log):
                    next_exchange = conversation_log[idx + 1]
                    user_answer = next_exchange.get("user_response", "")
                    quality_score = next_exchange.get("quality", 0.0)
                else:
                    user_answer = "(Session ended - no answer)"

                # Determine response type
                response_type = "answered"

                if not user_answer or user_answer.strip() == "":
                    response_type = "no_response"
                elif user_answer == "(session_start)":
                    response_type = "session_start"
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

                paired_exchanges.append({
                    "index": len(paired_exchanges) + 1,
                    "question": ai_message,
                    "answer": user_answer,
                    "response_type": response_type,
                    "stage": stage,
                    "concept": concept,
                    "quality_score": quality_score,
                    "is_followup": is_followup
                })

            logger.info(f"📊 Paired {len(paired_exchanges)} Q&A (answered={answered}, silent={silent}, repeat={repeat_requests})")

            session_document = {
                "session_id": session_id,
                "test_id": test_id,
                "student_id": student_id,
                "student_name": student_name,
                "total_exchanges": len(paired_exchanges),
                "answered_count": answered,
                "skipped_count": skipped,
                "silent_count": silent,
                "irrelevant_count": irrelevant,
                "repeat_requests_count": repeat_requests,
                "auto_advanced_count": auto_advanced,
                "conversation": paired_exchanges,
                "timestamp": datetime.now(),
                "created_at": datetime.utcnow(),
                "type": "qa_session"
            }

            result = self._qa_collection.insert_one(session_document)
            logger.info(f"✅ Saved {len(paired_exchanges)} correctly paired Q&A exchanges")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving session: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_session_qa(self, session_id: str) -> dict:
        """Retrieve Q&A session document for a session"""
        try:
            if self._qa_collection is None:
                self._init_qa_mongodb()

            doc = self._qa_collection.find_one(
                {"session_id": session_id, "type": "qa_session"}
            )

            return doc if doc else {}

        except Exception as e:
            logger.error(f"❌ Error retrieving Q&A: {e}")
            return {}

    def get_student_qa_history(self, student_id: int, limit: int = 100) -> list:
        """Retrieve Q&A session history for a student"""
        try:
            if self._qa_collection is None:
                self._init_qa_mongodb()

            cursor = self._qa_collection.find(
                {"student_id": student_id, "type": "qa_session"}
            ).sort("timestamp", -1).limit(limit)

            return list(cursor)

        except Exception as e:
            logger.error(f"❌ Error retrieving student Q&A history: {e}")
            return []

    # ========================================================================
    #  DAILY STANDUP — Session Results (evaluation storage + PDF generation)
    # ========================================================================
    async def save_session_result_fast(self, session_data, evaluation: str, score: float) -> bool:
        """Save session evaluation result to MongoDB for later PDF generation."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_save_session_result,
                session_data,
                evaluation,
                score
            )
        except Exception as e:
            logger.error(f"❌ Async session result save error: {e}")
            return False

    def _sync_save_session_result(self, session_data, evaluation: str, score: float) -> bool:
        """Synchronous method to save session result to MongoDB with CORRECT Q&A pairing."""
        try:
            if self._qa_collection is None:
                self._init_qa_mongodb()

            # Get raw conversation log
            raw_conversation_log = getattr(session_data, 'conversation_log', [])

            logger.info(f"📝 Processing {len(raw_conversation_log)} raw conversation entries for storage")

            # ✅ TRANSFORM TO CORRECT PAIRING - Same logic as save_qa_to_mongodb
            corrected_conversation_log = []

            for idx in range(len(raw_conversation_log)):
                entry = raw_conversation_log[idx]

                ai_message = entry.get("ai_message", "")
                stage = entry.get("stage", "unknown")
                concept = entry.get("concept", "unknown")
                is_followup = entry.get("is_followup", False)
                timestamp = entry.get("timestamp", 0)

                # Skip entries without meaningful AI message
                if not ai_message or len(ai_message.strip()) < 5:
                    continue

                # ✅ GET ANSWER FROM NEXT ENTRY (the fix!)
                user_answer = ""
                quality_score = 0.0

                if idx + 1 < len(raw_conversation_log):
                    next_entry = raw_conversation_log[idx + 1]
                    user_answer = next_entry.get("user_response", "")
                    quality_score = next_entry.get("quality", 0.0)
                else:
                    user_answer = "(Session ended - no answer)"

                # Skip placeholder entries
                if user_answer == "(session_start)":
                    continue

                corrected_conversation_log.append({
                    "timestamp": timestamp,
                    "stage": stage,
                    "ai_message": ai_message,
                    "user_response": user_answer,  # ✅ Now correctly paired!
                    "quality": quality_score,
                    "concept": concept,
                    "is_followup": is_followup
                })

            logger.info(f"✅ Transformed to {len(corrected_conversation_log)} correctly paired exchanges")

            # Log first few pairs for verification
            for i, pair in enumerate(corrected_conversation_log[:3]):
                logger.info(f"📊 Pair {i+1}: Q='{pair['ai_message'][:40]}...' A='{pair['user_response'][:40]}...'")

            # Extract detailed evaluation if available
            detailed_evaluation = None
            if hasattr(session_data, 'detailed_evaluation'):
                detailed_evaluation = session_data.detailed_evaluation

            # Build session result document with CORRECTED conversation log
            result_document = {
                "session_id": session_data.session_id,
                "test_id": session_data.test_id,
                "student_id": session_data.student_id,
                "student_name": session_data.student_name,
                "evaluation": evaluation,
                "score": score,
                "detailed_evaluation": detailed_evaluation,
                "duration": time.time() - session_data.created_at,
                "total_exchanges": len(corrected_conversation_log),
                "conversation_log": corrected_conversation_log,  # ✅ CORRECTED VERSION!
                "silence_responses": getattr(session_data, 'silence_response_count', 0),
                "fragment_analytics": {
                    "total_fragments": len(getattr(session_data, 'fragment_keys', [])),
                    "concepts_covered": list(getattr(session_data, 'concept_question_counts', {}).keys()),
                },
                "timestamp": time.time(),
                "created_at": datetime.utcnow(),
                "type": "session_result"
            }

            # Upsert - update if exists, insert if not
            result = self._qa_collection.update_one(
                {"session_id": session_data.session_id, "type": "session_result"},
                {"$set": result_document},
                upsert=True
            )

            if result.upserted_id or result.modified_count > 0:
                logger.info(f"✅ Session result saved: {session_data.session_id}")
                return True
            else:
                logger.warning(f"⚠️ Session result unchanged: {session_data.session_id}")
                return True  # Still consider success if document existed

        except Exception as e:
            logger.error(f"❌ Error saving session result: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def get_session_result_fast(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session result for PDF generation."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_session_result,
                session_id
            )
        except Exception as e:
            logger.error(f"❌ Async session result retrieval error: {e}")
            return None

    def _sync_get_session_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous method to retrieve session result from MongoDB."""
        try:
            if self._qa_collection is None:
                self._init_qa_mongodb()

            # First try to find session_result document
            result = self._qa_collection.find_one(
                {"session_id": session_id, "type": "session_result"}
            )

            if result:
                logger.info(f"✅ Found session_result for: {session_id}")
                return result

            # Fallback: try to find qa_session document and build result from it
            qa_session = self._qa_collection.find_one(
                {"session_id": session_id, "type": "qa_session"}
            )

            if qa_session:
                logger.info(f"✅ Found qa_session for: {session_id}, building result")
                # Build a result document from qa_session
                return {
                    "session_id": session_id,
                    "test_id": qa_session.get("test_id"),
                    "student_id": qa_session.get("student_id"),
                    "student_name": qa_session.get("student_name"),
                    "evaluation": "Session completed.",
                    "score": 70,  # Default score
                    "detailed_evaluation": None,
                    "duration": 0,
                    "total_exchanges": qa_session.get("total_exchanges", 0),
                    "conversation_log": qa_session.get("conversation", []),
                    "silence_responses": qa_session.get("silent_count", 0),
                    "timestamp": qa_session.get("timestamp"),
                    "type": "session_result"
                }

            logger.warning(f"⚠️ No session data found for: {session_id}")
            return None

        except Exception as e:
            logger.error(f"❌ Error retrieving session result: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ========================================================================
    #  DAILY STANDUP — Student Info + Summary
    # ========================================================================
    async def get_student_info_fast(self, student_id: int = None) -> Tuple[int, str, str, str]:
        """Fetch student info - specific student if ID provided, random otherwise."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_get_student_info,
            student_id
        )

    def _sync_get_student_info(self, student_id: int = None) -> Tuple[int, str, str, str]:
        try:
            conn = self.get_mysql_connection()
            cursor = conn.cursor(dictionary=True)

            # if student_id:
            #     # ✅ Fetch specific logged-in student
            #     logger.info(f"📋 Fetching student with ID: {student_id}")
            #     cursor.execute("""
            #         SELECT ID, First_Name, Last_Name 
            #         FROM tbl_Student 
            #         WHERE ID = %s
            #         LIMIT 1
            #     """, (student_id,))

            if student_id:
                # ✅ Fetch specific logged-in student (validated active)
                logger.info(f"📋 Fetching student with ID: {student_id}")
                cursor.execute("""
                    SELECT ID, First_Name, Last_Name 
                    FROM tbl_Student 
                    WHERE ID = %s AND status = 1 AND Status_code = 'st'
                    LIMIT 1
                """, (student_id,))
                
            else:
                # ⚠️ Fallback to random (testing only)
                logger.warning("⚠️ No student_id provided - random student mode")
                cursor.execute("""
                    SELECT ID, First_Name, Last_Name 
                    FROM tbl_Student 
                    WHERE ID IS NOT NULL AND First_Name IS NOT NULL AND Last_Name IS NOT NULL
                    ORDER BY RAND()
                    LIMIT 1
                """)

            row = cursor.fetchone()
            cursor.close()
            conn.close()

            if not row:
                error_msg = f"No student found with ID: {student_id}" if student_id else "No students found"
                raise Exception(error_msg)

            session_key = f"SESSION_{int(time.time())}"
            logger.info(f"✅ Found student: {row['First_Name']} {row['Last_Name']} (ID: {row['ID']})")
            return (row['ID'], row['First_Name'], row['Last_Name'], session_key)

        except Exception as e:
            logger.error(f"❌ Error fetching student info: {e}")
            raise

    async def get_summary_fast(self) -> str:
        """Fetch summary from MongoDB (daily_standup style)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor,
                self._sync_get_summary
            )
        except Exception as e:
            raise

    def _sync_get_summary(self) -> str:
        """Fetch latest summary from MongoDB by timestamp"""
        try:
            mongo_cfg = self.mongo_config
            username = quote_plus(mongo_cfg['username'])
            password = quote_plus(mongo_cfg['password'])
            mongo_uri = f"mongodb://{username}:{password}@{mongo_cfg['host']}:{mongo_cfg['port']}/{mongo_cfg['auth_source']}"

            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            db = client[mongo_cfg['database']]
            collection = db[config.SUMMARIES_COLLECTION]

            # Sort by timestamp DESC, look for summary_text field
            doc = collection.find_one(
                {"summary_text": {"$exists": True, "$ne": None, "$ne": ""}},
                sort=[("timestamp", -1)]
            )

            # Fallback to old field name if needed
            if not doc:
                doc = collection.find_one(
                    {"summary": {"$exists": True, "$ne": None, "$ne": ""}},
                    sort=[("timestamp", -1)]
                )

            client.close()

            if not doc:
                raise Exception("No valid summary found in MongoDB")

            # Try both field names
            summary_text = doc.get("summary_text") or doc.get("summary")

            if not summary_text:
                raise Exception("Summary field is empty")

            # Log what we fetched
            logger.info(f"✅ Fetched summary: {len(summary_text)} chars")
            logger.info(f"✅ Contains RFC: {'RFC' in summary_text}")
            logger.info(f"✅ First 100 chars: {summary_text[:100]}")

            return summary_text.strip()

        except Exception as e:
            logger.error(f"❌ Error fetching summary: {e}")
            raise

    # ========================================================================
    #  WEEKEND MOCKTEST SPECIFIC
    # ========================================================================
    def _init_mongodb_weekend(self):
        """Initialize MongoDB (weekend mocktest style)"""
        self.mongo_client = pymongo.MongoClient(
            config.MONGO_CONNECTION_STRING,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            maxPoolSize=50,
            minPoolSize=5
        )
        self.db = self.mongo_client[config.MONGO_DB_NAME]
        self.summaries_collection = self.db[config.SUMMARIES_COLLECTION]
        self.test_results_collection = self.db[config.TEST_RESULTS_COLLECTION]

    def get_recent_summaries(self, limit: int = None) -> List[Dict[str, Any]]:
        """Fetch recent summaries (weekend_mocktest)"""
        if not self.summaries_collection:
            self._init_mongodb_weekend()
        if limit is None:
            limit = config.RECENT_SUMMARIES_COUNT
        cursor = self.summaries_collection.find(
            {"summary": {"$exists": True, "$ne": "", "$type": "string"}},
            {"summary": 1, "timestamp": 1, "date": 1}
        ).sort("_id", pymongo.DESCENDING).limit(limit)
        return list(cursor)

    def save_test_results(self, test_id: str, test_data: Dict[str, Any], evaluation_result: Dict[str, Any]) -> bool:
        if not self.test_results_collection:
            self._init_mongodb_weekend()
        doc = {
            "test_id": test_id,
            "timestamp": time.time(),
            "evaluation_report": evaluation_result.get("evaluation_report", ""),
            "total_questions": test_data.get("total_questions"),
            "score": evaluation_result.get("total_correct", 0),
        }
        result = self.test_results_collection.insert_one(doc)
        return bool(result.inserted_id)

    # ========================================================================
    #  WEEKLY INTERVIEW — Summary Retrieval
    # ========================================================================
    async def get_recent_summaries_fast(self, days: int = None, limit: int = None) -> List[Dict[str, Any]]:
        """Weekly Interview: fetch recent summaries with 7-day filter"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.client_manager.executor,
            self._sync_get_recent_summaries,
            days or config.RECENT_SUMMARIES_DAYS,
            limit or config.SUMMARIES_LIMIT
        )

    def _sync_get_recent_summaries(self, days: int, limit: int) -> List[Dict[str, Any]]:
        """Synchronous 7-day summaries retrieval with smart filtering"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db[config.SUMMARIES_COLLECTION]

            # Calculate 7-day window
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_timestamp = start_date.timestamp()

            # Multiple query strategies for maximum 7-day coverage
            query_strategies = [
                {
                    "name": "timestamp_based_7day",
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"},
                        "timestamp": {"$gte": start_timestamp},
                        "$expr": {"$gt": [{"$strLenCP": "$summary"}, config.MIN_CONTENT_LENGTH]}
                    },
                    "sort": [("timestamp", -1)]
                },
                {
                    "name": "date_based_7day",
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"},
                        "date": {"$gte": start_date.strftime("%Y-%m-%d")},
                        "$expr": {"$gt": [{"$strLenCP": "$summary"}, config.MIN_CONTENT_LENGTH]}
                    },
                    "sort": [("date", -1)]
                },
                {
                    "name": "recent_quality_summaries",
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"},
                        "$expr": {"$gt": [{"$strLenCP": "$summary"}, config.MIN_CONTENT_LENGTH * 2]}
                    },
                    "sort": [("_id", -1)]
                },
                {
                    "name": "fallback_any_summaries",
                    "filter": {
                        "summary": {"$exists": True, "$ne": "", "$type": "string"}
                    },
                    "sort": [("_id", -1)]
                }
            ]

            summaries = []

            for strategy in query_strategies:
                try:
                    logger.info(f"🔍 Trying strategy: {strategy['name']}")

                    cursor = collection.find(
                        strategy["filter"],
                        {
                            "summary": 1,
                            "timestamp": 1,
                            "date": 1,
                            "session_id": 1,
                            "_id": 1
                        }
                    ).sort(strategy["sort"]).limit(limit)

                    summaries = list(cursor)

                    if summaries:
                        logger.info(f"✅ Retrieved {len(summaries)} summaries using {strategy['name']}")
                        break

                except Exception as e:
                    logger.warning(f"⚠️ Strategy {strategy['name']} failed: {e}")
                    continue

            client.close()

            if not summaries:
                raise Exception("No valid summaries found in database for 7-day interview processing")

            # Log sample for verification
            if summaries:
                first_summary = summaries[0]["summary"]
                sample_length = min(len(first_summary), 200)
                logger.info(f"📄 Sample summary ({sample_length} chars): {first_summary[:sample_length]}...")
                logger.info(f"📊 Total summaries for interview: {len(summaries)}")

            return summaries

        except Exception as e:
            logger.error(f"❌ Sync 7-day summary retrieval error: {e}")
            raise Exception(f"MongoDB 7-day summary retrieval failed: {e}")

    # ========================================================================
    #  WEEKLY INTERVIEW — HR Question Tracking (cross-session)
    # ========================================================================
    async def get_hr_questions_asked(self, student_id: int, limit: int = 50) -> List[str]:
        """Get list of HR questions previously asked to this student across all sessions"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_hr_questions_asked,
                student_id,
                limit
            )
        except Exception as e:
            logger.error(f"❌ Error getting HR questions: {e}")
            return []

    def _sync_get_hr_questions_asked(self, student_id: int, limit: int) -> List[str]:
        """Sync helper to get previously asked HR questions"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["hr_questions_history"]

            cursor = collection.find(
                {"student_id": student_id},
                {"question": 1, "_id": 0}
            ).sort("asked_at", -1).limit(limit)

            questions = [doc["question"] for doc in cursor if doc.get("question")]

            client.close()
            logger.info(f"✅ Found {len(questions)} previous HR questions for student {student_id}")
            return questions

        except Exception as e:
            logger.error(f"❌ Error fetching HR questions: {e}")
            return []

    async def store_hr_question_asked(self, student_id: int, question: str, session_id: str = None) -> bool:
        """Store an HR question that was asked to prevent future repetition"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_store_hr_question,
                student_id,
                question,
                session_id
            )
        except Exception as e:
            logger.error(f"❌ Error storing HR question: {e}")
            return False

    def _sync_store_hr_question(self, student_id: int, question: str, session_id: str) -> bool:
        """Sync helper to store HR question"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["hr_questions_history"]

            doc = {
                "student_id": student_id,
                "question": question,
                "session_id": session_id,
                "asked_at": datetime.utcnow()
            }

            result = collection.insert_one(doc)
            client.close()

            if result.inserted_id:
                logger.info(f"✅ Stored HR question for student {student_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"❌ Error storing HR question: {e}")
            return False

    async def clear_hr_questions_for_student(self, student_id: int) -> bool:
        """Clear HR question history for a student (for testing/reset)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_clear_hr_questions,
                student_id
            )
        except Exception as e:
            logger.error(f"❌ Error clearing HR questions: {e}")
            return False

    def _sync_clear_hr_questions(self, student_id: int) -> bool:
        """Sync helper to clear HR questions"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["hr_questions_history"]

            result = collection.delete_many({"student_id": student_id})
            client.close()

            logger.info(f"✅ Cleared {result.deleted_count} HR questions for student {student_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error clearing HR questions: {e}")
            return False

    # ========================================================================
    #  WEEKLY INTERVIEW — HR Questions from DB (with caching)
    # ========================================================================
    async def get_hr_questions_from_db(self, limit: int = 100, candidate_type: str = "fresher") -> List[str]:
        """
        Fetch HR questions from MongoDB collection 'HR&Managerial_Interview_Quest'.

        Args:
            limit: Maximum number of questions to return
            candidate_type: "fresher" or "experienced"
        """
        cache_key = f"hr_questions_{candidate_type}"

        # Check cache
        if hasattr(self, '_hr_questions_cache') and self._hr_questions_cache:
            if self._hr_questions_cache.get(cache_key) and self._hr_questions_cache_time:
                cache_age = time.time() - self._hr_questions_cache_time
                if cache_age < 3600:  # 1 hour cache
                    return self._hr_questions_cache[cache_key][:limit]

        try:
            loop = asyncio.get_event_loop()
            questions = await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_hr_questions_from_db,
                limit,
                candidate_type
            )

            # Update cache
            if not hasattr(self, '_hr_questions_cache'):
                self._hr_questions_cache = {}
            self._hr_questions_cache[cache_key] = questions
            self._hr_questions_cache_time = time.time()

            return questions
        except Exception as e:
            logger.error(f"Error fetching HR questions: {e}")
            return []

    def _sync_get_hr_questions_from_db(self, limit: int, candidate_type: str = "fresher") -> List[str]:
        """Extract questions from nested document structure."""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client["ml_notes"]
            collection = db["HR&Managerial_Interview_Questions"]

            # Find document by candidate_type
            doc = collection.find_one({"candidate_type": candidate_type})

            if not doc:
                doc = collection.find_one({})  # Fallback to any document
                if not doc:
                    return []

            questions = []

            # Extract from behavioral, leadership, introduction (most relevant for HR)
            for category in ["behavioral", "leadership", "introduction"]:
                if category in doc and isinstance(doc[category], dict):
                    if "questions" in doc[category]:
                        for q_obj in doc[category]["questions"]:
                            if isinstance(q_obj, dict) and "text" in q_obj:
                                q_text = q_obj["text"].strip()
                                if len(q_text) > 10 and q_text not in questions:
                                    questions.append(q_text)

            client.close()
            logger.info(f"[HR] Loaded {len(questions)} questions from MongoDB")
            return questions[:limit]

        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    # ========================================================================
    #  WEEKLY INTERVIEW — Save/Retrieve Interview Results
    # ========================================================================
    async def save_interview_result_fast(self, interview_data: Dict[str, Any]) -> bool:
        """Save weekly interview result to MongoDB asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_save_interview_result,
                interview_data
            )
        except Exception as e:
            logger.error(f"❌ Error saving interview result: {e}")
            return False

    def _sync_save_interview_result(self, interview_data: Dict[str, Any]) -> bool:
        """Synchronous helper to save interview result to MongoDB"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["weekly_interview_results"]

            interview_data["created_at"] = datetime.utcnow()
            interview_data["updated_at"] = datetime.utcnow()
            interview_data["timestamp"] = time.time()

            result = collection.insert_one(interview_data)
            client.close()

            if result.inserted_id:
                logger.info(f"✅ Interview result saved with ID: {result.inserted_id}")
                logger.info(f"✅ Saved test_id: {interview_data.get('test_id')}, session_id: {interview_data.get('session_id')}")
                return True
            else:
                logger.error("❌ Failed to save interview result - no inserted_id returned")
                return False

        except Exception as e:
            logger.error(f"❌ Error saving interview result to MongoDB: {e}")
            return False

    async def get_interview_result_fast(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific interview result by test_id (FIXED - searches by test_id)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_interview_result_by_test_id,
                test_id
            )
        except Exception as e:
            logger.error(f"❌ Error retrieving interview result: {e}")
            return None

    def _sync_get_interview_result_by_test_id(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous helper to retrieve interview result by test_id"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["weekly_interview_results"]

            logger.info(f"🔍 Searching for interview result with test_id: {test_id}")

            # Try to find by test_id first
            result = collection.find_one({"test_id": test_id})

            # If not found by test_id, try session_id
            if not result:
                logger.info(f"🔍 Not found by test_id, trying session_id: {test_id}")
                result = collection.find_one({"session_id": test_id})

            client.close()

            if result:
                result["_id"] = str(result["_id"])
                logger.info(f"✅ Retrieved interview result for test_id: {test_id}")
                logger.info(f"✅ Result has {len(result.get('conversation_log', []))} exchanges")
            else:
                logger.warning(f"⚠️ No interview result found for test_id: {test_id}")

            return result

        except Exception as e:
            logger.error(f"❌ Error retrieving interview result from MongoDB: {e}")
            return None

    async def get_interview_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific interview result by session_id (original method kept for compatibility)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_interview_result,
                session_id
            )
        except Exception as e:
            logger.error(f"❌ Error retrieving interview result: {e}")
            return None

    def _sync_get_interview_result(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous helper to retrieve interview result"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["weekly_interview_results"]

            result = collection.find_one({"session_id": session_id})
            client.close()

            if result:
                result["_id"] = str(result["_id"])
                logger.info(f"✅ Retrieved interview result for session: {session_id}")

            return result

        except Exception as e:
            logger.error(f"❌ Error retrieving interview result from MongoDB: {e}")
            return None

    async def get_all_interview_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all interview results (for debugging/admin)"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_all_interview_results,
                limit
            )
        except Exception as e:
            logger.error(f"❌ Error retrieving all interview results: {e}")
            return []

    def _sync_get_all_interview_results(self, limit: int) -> List[Dict[str, Any]]:
        """Synchronous helper to retrieve all interview results"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["weekly_interview_results"]

            cursor = collection.find().sort("timestamp", -1).limit(limit)

            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)

            client.close()

            logger.info(f"✅ Retrieved {len(results)} interview results")
            return results

        except Exception as e:
            logger.error(f"❌ Error retrieving all interview results from MongoDB: {e}")
            return []

    async def get_student_interview_history(self, student_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve interview history for a specific student."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.client_manager.executor if self.client_manager else None,
                self._sync_get_student_interview_history,
                student_id,
                limit
            )
        except Exception as e:
            logger.error(f"❌ Error retrieving student interview history: {e}")
            return []

    def _sync_get_student_interview_history(self, student_id: int, limit: int) -> List[Dict[str, Any]]:
        """Synchronous helper to retrieve student's interview history"""
        try:
            client = MongoClient(config.mongodb_connection_string, serverSelectionTimeoutMS=5000)
            db = client[config.MONGODB_DATABASE]
            collection = db["weekly_interview_results"]

            cursor = collection.find(
                {"student_id": student_id}
            ).sort("timestamp", -1).limit(limit)

            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)

            client.close()

            logger.info(f"✅ Retrieved {len(results)} interview records for student: {student_id}")
            return results

        except Exception as e:
            logger.error(f"❌ Error retrieving student interview history from MongoDB: {e}")
            return []

    # ========================================================================
    #  COMMON UTILITIES
    # ========================================================================
    async def close_connections(self):
        if self._mongo_client:
            self._mongo_client.close()
        if self.mongo_client:
            self.mongo_client.close()
        if self._qa_mongo_client:
            self._qa_mongo_client.close()
        logger.info("🔌 Database connections closed")


# ============================================================================
# SINGLETON HELPERS (weekend_mocktest style)
# ============================================================================
_db_manager = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def close_db_manager():
    global _db_manager
    if _db_manager:
        _db_manager.close_connections()
        _db_manager = None