"""
Biometric Authentication Service for Daily Standup - CLEAN VERSION
===================================================================
Handles face verification (pre-standup) and voice verification (during standup)

Architecture:
  - Client-side: BlazeFace (face count) + COCO-SSD (prohibited objects)
  - Server-side: InsightFace (identity verification + pose check)
  - NO YOLO - all object/person detection handled client-side

Place this file in: EDU-APP/core/biometric_auth.py
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime
import logging
import tempfile
import os
import base64
from pymongo import MongoClient
from urllib.parse import quote_plus
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class BiometricAuthService:
    """Service for biometric authentication during standup sessions"""
    
    def __init__(self, mongo_host: str = "192.168.48.201", mongo_port: int = 27017,
                 db_name: str = "connectlydb", username: str = "connectly", 
                 password: str = "LT@connect25", auth_source: str = "admin"):
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port
        self.db_name = db_name
        self.username = username
        self.password = password
        self.auth_source = auth_source
        
        self._client: Optional[MongoClient] = None
        self._db = None
        
        # Similarity thresholds
        self.FACE_SIMILARITY_THRESHOLD = 0.5
        self.VOICE_SIMILARITY_THRESHOLD = 0.45
        
        # Models (lazy loaded)
        self._face_analyzer = None
        self._voice_encoder = None
        
    def _get_connection_string(self) -> str:
        """Build MongoDB connection string"""
        encoded_pass = quote_plus(self.password)
        return (
            f"mongodb://{self.username}:{encoded_pass}"
            f"@{self.mongo_host}:{self.mongo_port}/{self.db_name}"
            f"?authSource={self.auth_source}"
        )
        
    def connect(self):
        """Initialize MongoDB connection"""
        if self._client is None:
            self._client = MongoClient(
                self._get_connection_string(),
                serverSelectionTimeoutMS=10000
            )
            self._db = self._client[self.db_name]
            logger.info("✅ BiometricAuthService connected to MongoDB")
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("🔌 BiometricAuthService disconnected from MongoDB")
    
    @property
    def face_analyzer(self):
        """Lazy load InsightFace analyzer"""
        if self._face_analyzer is None:
            try:
                from insightface.app import FaceAnalysis
                self._face_analyzer = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("✅ Face analyzer (InsightFace buffalo_l) loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load face analyzer: {e}")
                raise
        return self._face_analyzer
    
    @property
    def voice_encoder(self):
        """Lazy load SpeechBrain ECAPA-TDNN encoder"""
        if self._voice_encoder is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._voice_encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                logger.info("✅ Voice encoder (ECAPA-TDNN) loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load voice encoder: {e}")
                raise
        return self._voice_encoder
    
    # ================== DATABASE OPERATIONS ==================

    def get_stored_face_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """Retrieve stored face embedding for a student from student_photos collection"""
        self.connect()
        
        try:
            query = {"status": "active"}
            try:
                query["student_id"] = int(student_id)
            except (ValueError, TypeError):
                query["student_code"] = str(student_id)
            
            logger.info(f"🔍 Querying student_photos with: {query}")
            
            doc = self._db.student_photos.find_one(
                query,
                sort=[("uploaded_at", -1)]
            )
            
            if doc and doc.get("face_embedding", {}).get("has_embedding"):
                embedding = doc["face_embedding"]["embedding"]
                logger.info(f"✅ Found face embedding for student {student_id}")
                return np.array(embedding, dtype=np.float32)
            
            if doc and doc.get("embedding"):
                embedding = doc["embedding"]
                logger.info(f"✅ Found face embedding (alt field) for student {student_id}")
                return np.array(embedding, dtype=np.float32)
            
            logger.warning(f"⚠️ No face embedding found for student {student_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving face embedding: {e}")
            return None
    
    def get_stored_voice_embedding(self, student_id: str) -> Optional[np.ndarray]:
        """Retrieve stored voice embedding for a student from student_voice collection"""
        self.connect()
        
        try:
            query = {"status": "active"}
            try:
                query["student_id"] = int(student_id)
            except (ValueError, TypeError):
                query["student_code"] = str(student_id)
            
            logger.info(f"🔍 Querying student_voice with: {query}")
            
            doc = self._db.student_voice.find_one(
                query,
                sort=[("uploaded_at", -1)]
            )
            
            if doc and doc.get("voice_embedding", {}).get("has_embedding"):
                embedding = doc["voice_embedding"]["embedding"]
                logger.info(f"✅ Found voice embedding for student {student_id}")
                return np.array(embedding, dtype=np.float32)
            
            if doc and doc.get("embedding"):
                embedding = doc["embedding"]
                logger.info(f"✅ Found voice embedding (alt field) for student {student_id}")
                return np.array(embedding, dtype=np.float32)
            
            logger.warning(f"⚠️ No voice embedding found for student {student_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving voice embedding: {e}")
            return None

    
    # ================== EMBEDDING EXTRACTION ==================
    
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], str, str]:
        """
        Extract face embedding with attention detection.
        Uses InsightFace only — no YOLO.
        """
        try:
            import cv2

            if len(image_data) < 1000:
                return None, "Camera capture failed - image too small", "extraction_error"
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Failed to decode image", "extraction_error"

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width = img_rgb.shape[:2]
            
            faces = self.face_analyzer.get(img_rgb)
            
            # ==================== NO FACE ====================
            if len(faces) == 0:
                return None, "👤 No face detected - please look at the camera", "no_face"
            
            # ==================== MULTIPLE FACES ====================
            if len(faces) > 1:
                return None, f"👥 Multiple faces ({len(faces)}) - only you should be visible", "multiple_faces"
            
            face = faces[0]
            
            # ==================== FACE SIZE ====================
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            image_area = img_width * img_height
            face_ratio = face_area / image_area
            
            if face_ratio < 0.02:
                return None, "📏 Face too small - move closer to camera", "face_too_small"
            
            if face_ratio > 0.85:
                return None, "📏 Face too close - move back slightly", "face_too_close"
            
            # ==================== DETECTION CONFIDENCE ====================
            det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
            if det_score < 0.4:
                return None, "🔅 Face unclear - improve lighting", "poor_quality"
            
            # ==================== ATTENTION DETECTION ====================
            pose = getattr(face, 'pose', None)
            
            if pose is not None and len(pose) >= 3:
                raw_pitch = float(pose[0])
                raw_yaw = float(pose[1])
                raw_roll = float(pose[2])
                
                logger.info(f"👀 Raw pose: pitch={raw_pitch:.1f}, yaw={raw_yaw:.1f}, roll={raw_roll:.1f}")
                
                abs_yaw = abs(raw_yaw)
                abs_pitch = abs(raw_pitch)
                abs_roll = abs(raw_roll)
                
                # ===== YAW: Looking LEFT/RIGHT =====
                YAW_THRESHOLD = 20
                if abs_yaw > YAW_THRESHOLD:
                    direction = "left" if raw_yaw > 0 else "right"
                    logger.warning(f"👀 LOOKING {direction.upper()}: yaw={raw_yaw:.1f}°")
                    return None, f"👀 Looking {direction} - please face the camera", "not_looking_at_camera"
                
                # ===== PITCH: Looking UP/DOWN =====
                PITCH_DOWN_THRESHOLD = 15
                PITCH_UP_THRESHOLD = 20
                
                if raw_pitch > PITCH_DOWN_THRESHOLD:
                    logger.warning(f"👀 LOOKING DOWN: pitch={raw_pitch:.1f}° - possible reading!")
                    return None, "👀 Looking down detected - please look at camera", "looking_down"
                
                if raw_pitch < -PITCH_UP_THRESHOLD:
                    logger.warning(f"👀 LOOKING UP: pitch={raw_pitch:.1f}°")
                    return None, "👀 Looking up - please look straight at camera", "not_looking_at_camera"
                
                # ===== ROLL: Head TILT =====
                ROLL_THRESHOLD = 25
                if abs_roll > ROLL_THRESHOLD:
                    logger.warning(f"🔄 HEAD TILTED: roll={raw_roll:.1f}°")
                    return None, "🔄 Head tilted - please keep head straight", "head_tilted"
                
                logger.info(f"✅ Pose OK: pitch={raw_pitch:.1f}°, yaw={raw_yaw:.1f}°, roll={raw_roll:.1f}°")
                
            else:
                # Pose not available - try alternative method using landmarks
                logger.warning("⚠️ Pose data not available, checking landmarks...")
                
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    landmarks = face.landmark_2d_106
                    
                    try:
                        nose_tip = landmarks[54] if len(landmarks) > 54 else None
                        
                        face_center_x = (bbox[0] + bbox[2]) / 2
                        face_center_y = (bbox[1] + bbox[3]) / 2
                        
                        if nose_tip is not None:
                            nose_offset_x = (nose_tip[0] - face_center_x) / face_width
                            nose_offset_y = (nose_tip[1] - face_center_y) / face_height
                            
                            logger.info(f"👃 Nose offset: x={nose_offset_x:.2f}, y={nose_offset_y:.2f}")
                            
                            if abs(nose_offset_x) > 0.15:
                                direction = "left" if nose_offset_x < 0 else "right"
                                logger.warning(f"👀 LOOKING {direction.upper()} (landmark-based)")
                                return None, f"👀 Looking {direction} - please face the camera", "not_looking_at_camera"
                            
                            if nose_offset_y > 0.1:
                                logger.warning(f"👀 LOOKING DOWN (landmark-based)")
                                return None, "👀 Looking down - please look at camera", "looking_down"
                                
                    except Exception as landmark_err:
                        logger.warning(f"Landmark analysis failed: {landmark_err}")
            
            # ==================== EXTRACT EMBEDDING ====================
            if face.embedding is None:
                return None, "Could not extract face features", "extraction_error"
            
            logger.info(f"✅ Face embedding extracted successfully")
            return face.embedding, "", ""
            
        except Exception as e:
            logger.error(f"❌ Face processing error: {e}")
            import traceback
            traceback.print_exc()
            return None, f"Face processing error: {str(e)}", "extraction_error"

    def extract_voice_embedding(self, audio_data: bytes, audio_format: str = "webm") -> Tuple[Optional[np.ndarray], str]:
        """Extract voice embedding from audio bytes"""
        try:
            import torch
            import torchaudio
            
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            try:
                waveform, sample_rate = torchaudio.load(tmp_path)
                
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Minimum 0.5 seconds at 16kHz
                if waveform.shape[1] < 8000:
                    return None, "Audio too short - need at least 0.5 seconds"
                
                embedding = self.voice_encoder.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()
                
                logger.info(f"✅ Extracted voice embedding: shape={embedding.shape}")
                return embedding, ""
                
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"❌ Voice embedding extraction error: {e}")
            return None, f"Voice processing error: {str(e)}"
    
    # ================== SIMILARITY CALCULATION ==================
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        similarity = 1 - cosine(emb1, emb2)
        return float(similarity)
    
    # ================== VERIFICATION METHODS ==================
    
    def verify_face(self, student_code: str, image_data: bytes) -> Dict[str, Any]:
        """
        Verify face against stored embedding using InsightFace.
        Checks: face presence, attention/pose, identity match.
        NO YOLO — client-side BlazeFace + COCO-SSD handle person/object detection.
        
        Returns dict with:
        - verified: bool
        - similarity: float
        - threshold: float
        - error: str or None
        - can_proceed: bool
        - error_type: str
        """
        stored_embedding = self.get_stored_face_embedding(student_code)
        
        if stored_embedding is None:
            logger.warning(f"❌ No registered face for student {student_code}")
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": self.FACE_SIMILARITY_THRESHOLD,
                "error": "No registered face found for this student. Please complete face registration first.",
                "can_proceed": False,
                "error_type": "no_registration"
            }
        
        current_embedding, error, error_type = self.extract_face_embedding(image_data)
        
        if current_embedding is None:
            logger.warning(f"❌ Face extraction failed for {student_code}: {error} (type: {error_type})")
            
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": self.FACE_SIMILARITY_THRESHOLD,
                "error": error,
                "can_proceed": False,
                "error_type": error_type
            }
            
        similarity = self.cosine_similarity(stored_embedding, current_embedding)
        verified = similarity >= self.FACE_SIMILARITY_THRESHOLD
        
        logger.info(
            f"🔐 Face verification for {student_code}: "
            f"similarity={similarity:.4f}, threshold={self.FACE_SIMILARITY_THRESHOLD}, "
            f"verified={verified}"
        )
        
        if not verified:
            return {
                "verified": False,
                "similarity": round(similarity, 4),
                "threshold": self.FACE_SIMILARITY_THRESHOLD,
                "error": "Face does not match registered profile - unauthorized person detected",
                "can_proceed": False,
                "error_type": "face_mismatch"
            }
        
        return {
            "verified": True,
            "similarity": round(similarity, 4),
            "threshold": self.FACE_SIMILARITY_THRESHOLD,
            "error": None,
            "can_proceed": True,
            "error_type": None
        }

    def verify_face_identity_only(self, student_code: str, image_data: bytes) -> Dict[str, Any]:
        """
        Lightweight identity verification using InsightFace ONLY.
        
        Used by ProctoringMonitor for periodic server-side identity checks
        during active sessions (every 4s). Client-side BlazeFace + COCO-SSD
        handles face presence, pose, and prohibited object detection.
        
        Key differences from verify_face():
        - RELAXED pose thresholds (yaw<35°, pitch<30° vs yaw<20°, pitch<15°)
        - Simpler error handling (client already catches most attention issues)
        
        GPU time: ~50-100ms per check
        
        Returns:
            dict with keys: verified, similarity, error, error_type
        """
        # Step 1: Get stored embedding
        stored_embedding = self.get_stored_face_embedding(student_code)
        
        if stored_embedding is None:
            return {
                "verified": False,
                "similarity": 0.0,
                "error": "No registered face found for this student",
                "error_type": "no_registration"
            }
        
        # Step 2: Extract face embedding with RELAXED thresholds
        try:
            import cv2
            
            if len(image_data) < 1000:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "error": "Camera capture failed - image too small",
                    "error_type": "extraction_error"
                }
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "error": "Failed to decode image",
                    "error_type": "extraction_error"
                }
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img_rgb)
            
            # No face detected
            if len(faces) == 0:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "error": "No face detected",
                    "error_type": "no_face"
                }
            
            # Multiple faces
            if len(faces) > 1:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "error": f"Multiple faces detected ({len(faces)})",
                    "error_type": "multiple_faces"
                }
            
            face = faces[0]
            
            # Detection confidence check
            det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
            if det_score < 0.3:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "error": "Face unclear - improve lighting",
                    "error_type": "poor_quality"
                }
            
            # RELAXED pose check (client-side BlazeFace handles strict pose)
            pose = getattr(face, 'pose', None)
            if pose is not None and len(pose) >= 3:
                raw_pitch = float(pose[0])
                raw_yaw = float(pose[1])
                
                # Relaxed thresholds: only reject extreme head turns
                YAW_THRESHOLD = 35
                PITCH_THRESHOLD = 30
                
                if abs(raw_yaw) > YAW_THRESHOLD:
                    direction = "left" if raw_yaw > 0 else "right"
                    logger.info(f"👀 Identity check: face turned {direction} (yaw={raw_yaw:.1f}°)")
                    return {
                        "verified": False,
                        "similarity": 0.0,
                        "error": f"Face turned {direction}",
                        "error_type": "face_turned"
                    }
                
                if abs(raw_pitch) > PITCH_THRESHOLD:
                    direction = "down" if raw_pitch > 0 else "up"
                    logger.info(f"👀 Identity check: looking {direction} (pitch={raw_pitch:.1f}°)")
                    return {
                        "verified": False,
                        "similarity": 0.0,
                        "error": f"Looking {direction}",
                        "error_type": "face_turned"
                    }
            
            # Extract embedding
            if face.embedding is None:
                return {
                    "verified": False,
                    "similarity": 0.0,
                    "error": "Could not extract face features",
                    "error_type": "extraction_error"
                }
            
            current_embedding = face.embedding
            
        except Exception as e:
            logger.error(f"❌ Identity extraction error: {e}")
            return {
                "verified": False,
                "similarity": 0.0,
                "error": f"Face processing error: {str(e)}",
                "error_type": "extraction_error"
            }
        
        # Step 3: Compare embeddings
        similarity = self.cosine_similarity(stored_embedding, current_embedding)
        verified = similarity >= self.FACE_SIMILARITY_THRESHOLD
        
        logger.info(
            f"🔐 Identity-only verification for {student_code}: "
            f"similarity={similarity:.4f}, threshold={self.FACE_SIMILARITY_THRESHOLD}, "
            f"verified={verified}"
        )
        
        if not verified:
            return {
                "verified": False,
                "similarity": round(similarity, 4),
                "error": "Face does not match registered identity",
                "error_type": "face_mismatch"
            }
        
        return {
            "verified": True,
            "similarity": round(similarity, 4),
            "error": None,
            "error_type": None
        }


    def verify_voice(self, student_code: str, audio_data: bytes, audio_format: str = "webm") -> Dict[str, Any]:
        """Verify voice against stored embedding"""
        stored_embedding = self.get_stored_voice_embedding(student_code)
        
        if stored_embedding is None:
            logger.warning(f"⚠️ No voice embedding found for student {student_code} - skipping verification")
            return {
                "verified": True,
                "similarity": -1.0,
                "threshold": self.VOICE_SIMILARITY_THRESHOLD,
                "error": "Voice embedding not available - skipping verification",
                "is_error": True,
                "skip_warning": True  # Don't penalize student for missing embedding
            }
        
        current_embedding, error = self.extract_voice_embedding(audio_data, audio_format)
        
        if current_embedding is None:
            # Extraction errors should NOT count as warnings - technical issue
            logger.warning(f"⚠️ Voice extraction failed: {error} - skipping verification")
            return {
                "verified": True,  # Don't count as failure
                "similarity": -1.0,
                "threshold": self.VOICE_SIMILARITY_THRESHOLD,
                "error": error,
                "is_extraction_error": True,
                "skip_warning": True  # Skip warning for technical errors
            }
        
        similarity = self.cosine_similarity(stored_embedding, current_embedding)
        verified = similarity >= self.VOICE_SIMILARITY_THRESHOLD
        
        logger.info(
            f"🎤 Voice verification for {student_code}: "
            f"similarity={similarity:.4f}, threshold={self.VOICE_SIMILARITY_THRESHOLD}, "
            f"verified={verified}"
        )
        
        return {
            "verified": verified,
            "similarity": round(similarity, 4),
            "threshold": self.VOICE_SIMILARITY_THRESHOLD,
            "error": None if verified else "Voice does not match registered profile",
            "is_extraction_error": False,
            "skip_warning": False  # Normal verification - count warning if failed
        }


class VoiceVerificationTracker:
    """Tracks voice verification warnings during a standup session"""
    
    def __init__(self, max_warnings: int = 3):
        self.max_warnings = max_warnings
        self.sessions: Dict[str, Dict] = {}
    
    def start_session(self, session_id: str, student_code: str):
        """Initialize tracking for a new session"""
        self.sessions[session_id] = {
            "student_code": student_code,
            "warning_count": 0,
            "verification_history": [],
            "started_at": datetime.utcnow(),
            "terminated": False,
            "termination_reason": None,
            "consecutive_failures": 0,
            "last_verified_at": None
        }
        logger.info(f"🎬 Voice verification tracking started for session {session_id}")
    
    def record_verification(self, session_id: str, verified: bool, similarity: float, 
                           skip_warning: bool = False) -> Dict[str, Any]:
        """
        Record a verification result and return current status
        
        Args:
            session_id: The session ID
            verified: Whether voice was verified
            similarity: The similarity score
            skip_warning: If True, don't increment warning (for extraction errors)
        """
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Session {session_id} not found in tracker")
            return {
                "warning_count": 0,
                "should_terminate": False,
                "message": "Session not found"
            }
        
        session = self.sessions[session_id]
        
        # Check if session already terminated
        if session["terminated"]:
            return {
                "warning_count": session["warning_count"],
                "should_terminate": True,
                "message": "Session already terminated"
            }
        
        # Record in history
        session["verification_history"].append({
            "timestamp": datetime.utcnow(),
            "verified": verified,
            "similarity": similarity,
            "skip_warning": skip_warning
        })
        
        # Only increment warning if NOT verified AND NOT skipping
        if not verified and not skip_warning:
            session["warning_count"] += 1
            session["consecutive_failures"] += 1
            warning_count = session["warning_count"]
            
            logger.info(f"🔴 Voice MISMATCH for session {session_id}: "
                       f"similarity={similarity:.4f}, warning {warning_count}/{self.max_warnings}")
            
            if warning_count >= self.max_warnings:
                session["terminated"] = True
                session["termination_reason"] = "voice_verification_failed"
                logger.warning(f"🛑 Session {session_id} TERMINATED: {warning_count} voice failures")
                return {
                    "warning_count": warning_count,
                    "should_terminate": True,
                    "message": f"Session terminated: Voice verification failed {warning_count} times"
                }
            else:
                remaining = self.max_warnings - warning_count
                logger.warning(f"⚠️ Session {session_id} warning {warning_count}/{self.max_warnings}")
                return {
                    "warning_count": warning_count,
                    "should_terminate": False,
                    "message": f"Warning {warning_count}/{self.max_warnings}: Voice mismatch detected. {remaining} warning(s) remaining."
                }
        
        elif verified:
            # Voice matched - reset consecutive failures but NOT total warnings
            session["consecutive_failures"] = 0
            session["last_verified_at"] = datetime.utcnow()
            logger.info(f"✅ Voice VERIFIED for session {session_id}: similarity={similarity:.4f}")
        
        elif skip_warning:
            # Extraction error - log but don't count
            logger.info(f"⏭️ Skipping warning for session {session_id} (extraction error)")
        
        return {
            "warning_count": session["warning_count"],
            "should_terminate": False,
            "message": "Voice verified" if verified else "Skipped (extraction error)"
        }
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a session"""
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str):
        """Clean up session tracking"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🏁 Voice verification tracking ended for session {session_id}")


class FaceVerificationTracker:
    """Tracks face verification warnings during a standup session"""
    
    def __init__(self, max_warnings: int = 3):
        self.max_warnings = max_warnings
        self.sessions: Dict[str, Dict] = {}
    
    def start_session(self, session_id: str, student_code: str):
        """Initialize tracking for a new session"""
        self.sessions[session_id] = {
            "student_code": student_code,
            "warning_count": 0,
            "verification_history": [],
            "started_at": datetime.utcnow(),
            "terminated": False,
            "termination_reason": None,
            "last_verified_at": None,
            "error_counts": {
                "no_face": 0,
                "multiple_faces": 0,
                "face_turned": 0,
                "face_obstructed": 0,
                "face_mismatch": 0,
                "other": 0
            }
        }
        logger.info(f"🎬 Face verification tracking started for session {session_id}")
    
    def record_verification(self, session_id: str, verified: bool, similarity: float,
                           error_type: str = None, error_message: str = None) -> Dict[str, Any]:
        """
        Record a face verification result and return current status
        """
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Session {session_id} not found in face tracker")
            return {
                "warning_count": 0,
                "should_terminate": False,
                "message": "Session not found"
            }
        
        session = self.sessions[session_id]
        
        # Check if session already terminated
        if session["terminated"]:
            return {
                "warning_count": session["warning_count"],
                "should_terminate": True,
                "message": "Session already terminated"
            }
        
        # Record in history
        session["verification_history"].append({
            "timestamp": datetime.utcnow(),
            "verified": verified,
            "similarity": similarity,
            "error_type": error_type,
            "error_message": error_message
        })
        
        if not verified:
            session["warning_count"] += 1
            warning_count = session["warning_count"]
            
            # Track error types
            if error_type and error_type in session["error_counts"]:
                session["error_counts"][error_type] += 1
            else:
                session["error_counts"]["other"] += 1
            
            logger.info(f"🔴 Face verification FAILED for session {session_id}: "
                       f"type={error_type}, warning {warning_count}/{self.max_warnings}")
            
            if warning_count >= self.max_warnings:
                session["terminated"] = True
                session["termination_reason"] = f"face_verification_failed: {error_type}"
                logger.warning(f"🛑 Session {session_id} TERMINATED: {warning_count} face verification failures")
                return {
                    "warning_count": warning_count,
                    "should_terminate": True,
                    "message": f"Session terminated: Face verification failed {warning_count} times",
                    "error_type": error_type,
                    "error_counts": session["error_counts"]
                }
            else:
                remaining = self.max_warnings - warning_count
                return {
                    "warning_count": warning_count,
                    "should_terminate": False,
                    "message": f"Warning {warning_count}/{self.max_warnings}: {error_message}. {remaining} warning(s) remaining.",
                    "error_type": error_type,
                    "error_counts": session["error_counts"]
                }
        
        else:
            # Face verified successfully
            session["last_verified_at"] = datetime.utcnow()
            logger.info(f"✅ Face VERIFIED for session {session_id}: similarity={similarity:.4f}")
            return {
                "warning_count": session["warning_count"],
                "should_terminate": False,
                "message": "Face verified successfully"
            }
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a session"""
        return self.sessions.get(session_id)
    
    def end_session(self, session_id: str):
        """Clean up session tracking"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🏁 Face verification tracking ended for session {session_id}")


# ================== GLOBAL INSTANCES ==================
biometric_service: Optional[BiometricAuthService] = None
voice_tracker: Optional[VoiceVerificationTracker] = None
face_tracker: Optional[FaceVerificationTracker] = None


def init_biometric_services(
    mongo_host: str = "192.168.48.201",
    mongo_port: int = 27017,
    db_name: str = "connectlydb",
    username: str = "connectly",
    password: str = "LT@connect25",
    auth_source: str = "admin",
    max_voice_warnings: int = 3,
    max_face_warnings: int = 3
) -> Tuple[BiometricAuthService, VoiceVerificationTracker, FaceVerificationTracker]:
    """Initialize biometric services - call this at app startup"""
    global biometric_service, voice_tracker, face_tracker
    
    biometric_service = BiometricAuthService(
        mongo_host=mongo_host,
        mongo_port=mongo_port,
        db_name=db_name,
        username=username,
        password=password,
        auth_source=auth_source
    )
    biometric_service.connect()
    
    voice_tracker = VoiceVerificationTracker(max_warnings=max_voice_warnings)
    face_tracker = FaceVerificationTracker(max_warnings=max_face_warnings)
    
    logger.info("✅ Biometric services initialized successfully")
    return biometric_service, voice_tracker, face_tracker


def get_biometric_service() -> Optional[BiometricAuthService]:
    """Get the global biometric service instance"""
    return biometric_service


def get_voice_tracker() -> Optional[VoiceVerificationTracker]:
    """Get the global voice tracker instance"""
    return voice_tracker


def get_face_tracker() -> Optional[FaceVerificationTracker]:
    """Get the global face tracker instance"""
    return face_tracker