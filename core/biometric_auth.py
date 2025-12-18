"""
Biometric Authentication Service for Daily Standup - FIXED VERSION
===================================================================
Handles face verification (pre-standup) and voice verification (during standup)

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


# Global service instances (initialized by init_biometric_services)
biometric_service: Optional["BiometricAuthService"] = None
voice_tracker: Optional["VoiceVerificationTracker"] = None

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
    
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], str ,str]:
        """
        Extract face embedding from image bytes with enhanced security checks:
        - No face detected
        - Multiple faces detected
        - Face too small / far from camera
        - Face turned away (pose detection)
        - Face obstructed (landmark detection)
        - Poor detection confidence
        """
        try:
            import cv2

            # Debug 1: Log received data size
            logger.info(f"📊 Received image data: {len(image_data)} bytes")
            
            # Debug 2: Check if data is too small (likely blank/corrupted)
            if len(image_data) < 1000:
                logger.error(f"❌ Image data too small: {len(image_data)} bytes - likely blank frame or corrupted data")
                return None, "Camera capture failed - invalid image data (too small)", "extraction_error"
            
            # Debug 3: Check if data is suspiciously small (might be blank JPEG)
            if len(image_data) < 5000:
                logger.warning(f"⚠️ Image data is quite small: {len(image_data)} bytes - may be low quality or partially blank")
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return None, "Failed to decode image", "extraction_error"

            logger.info(f"📐 Decoded image dimensions: {img.shape[1]}x{img.shape[0]} (WxH)")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width = img_rgb.shape[:2]
            
            # Detect all faces in the image
            faces = self.face_analyzer.get(img_rgb)
            
            # Check 1: No face detected
            if len(faces) == 0:
                logger.warning("❌ No face detected in image")
                return None, "No face detected in image - please ensure your face is clearly visible", "no_face"
            
            # Check 2: Multiple faces detected
            if len(faces) > 1:
                logger.warning(f"❌ Multiple faces detected: {len(faces)} faces found")
                return None, f"Multiple faces detected ({len(faces)} faces) - only you should be visible", "multiple_faces"
            face = faces[0]
            
            # Check 3: Face bounding box and size
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            image_area = img_width * img_height
            face_ratio = face_area / image_area
            
            logger.info(f"📐 Face size: {face_width}x{face_height}, ratio: {face_ratio:.3f}")
            
            # Face too small (less than 3% of image) - too far or partially visible
            if face_ratio < 0.03:
                logger.warning(f"❌ Face too small: ratio={face_ratio:.3f}")
                return None, "Face too small or too far - please move closer to the camera", "face_too_small"
            
            # Face too large (more than 80% of image) - too close
            if face_ratio > 0.80:
                logger.warning(f"❌ Face too large: ratio={face_ratio:.3f}")
                return None, "Face too close to camera - please move back slightly", "face_too_close"
            
            # Check 4: Detection confidence score
            if hasattr(face, 'det_score'):
                det_score = float(face.det_score)
                logger.info(f"📊 Detection confidence: {det_score:.3f}")
                
                if det_score < 0.5:
                    logger.warning(f"❌ Low detection confidence: {det_score:.3f}")
                    return None, "Face not clearly visible - please ensure good lighting", "poor_quality"
                
                if det_score < 0.7:
                    logger.warning(f"⚠️ Medium detection confidence: {det_score:.3f}")
                    # Continue but log warning - might be partial obstruction
            
            # Check 5: Face pose/orientation (yaw = left/right, pitch = up/down, roll = tilt)
            if hasattr(face, 'pose') and face.pose is not None:
                pose = face.pose
                if len(pose) >= 3:
                    pitch, yaw, roll = pose[0], pose[1], pose[2]
                    logger.info(f"📐 Face pose: pitch={pitch:.1f}°, yaw={yaw:.1f}°, roll={roll:.1f}°")
                    
                    # Check if face is turned too much horizontally (looking left/right)
                    if abs(yaw) > 35:
                        direction = "left" if yaw > 0 else "right"
                        logger.warning(f"❌ Face turned {direction}: yaw={yaw:.1f}°")
                        return None, f"Face turned {direction} - please look at the camera", "face_turned"
                    
                    # Check if face is tilted too much vertically (looking up/down)
                    if abs(pitch) > 35:
                        direction = "down" if pitch > 0 else "up"
                        logger.warning(f"❌ Face tilted {direction}: pitch={pitch:.1f}°")
                        return None, f"Face tilted {direction} - please look straight at camera", "face_turned"
                    
                    # Check if face is rotated/tilted sideways
                    if abs(roll) > 30:
                        logger.warning(f"❌ Face tilted sideways: roll={roll:.1f}°")
                        return None, "Head tilted sideways - please keep head straight", "face_turned"
            
            # Check 6: Facial landmarks for obstruction detection
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                landmarks = face.landmark_2d_106
                
                # Check if key facial regions are properly detected
                # 106-point landmark model: 
                # - Eyes: points 33-42 (left eye), 87-96 (right eye)
                # - Nose: points 52-71
                # - Mouth: points 72-86
                
                try:
                    # Check eye regions
                    left_eye_points = landmarks[33:43]
                    right_eye_points = landmarks[87:97]
                    
                    # Check if eye landmarks are within face bounding box
                    left_eye_valid = np.all((left_eye_points[:, 0] >= bbox[0]) & 
                                           (left_eye_points[:, 0] <= bbox[2]) &
                                           (left_eye_points[:, 1] >= bbox[1]) & 
                                           (left_eye_points[:, 1] <= bbox[3]))
                    
                    right_eye_valid = np.all((right_eye_points[:, 0] >= bbox[0]) & 
                                            (right_eye_points[:, 0] <= bbox[2]) &
                                            (right_eye_points[:, 1] >= bbox[1]) & 
                                            (right_eye_points[:, 1] <= bbox[3]))
                    
                    if not left_eye_valid or not right_eye_valid:
                        logger.warning("❌ Eye landmarks invalid - possible obstruction")
                        return None, "Eyes not visible - remove objects covering your face", "face_obstructed"
                    # Check nose region
                    nose_points = landmarks[52:72]
                    nose_center = np.mean(nose_points, axis=0)
                    
                    # Nose should be roughly in the center of the face
                    face_center_x = (bbox[0] + bbox[2]) / 2
                    nose_offset = abs(nose_center[0] - face_center_x) / face_width
                    
                    if nose_offset > 0.3:
                        logger.warning(f"❌ Nose offset too high: {nose_offset:.2f}")
                        return None, "Face partially obstructed - ensure full face is visible", "face_obstructed"
                except Exception as landmark_error:
                    logger.warning(f"⚠️ Landmark analysis error: {landmark_error}")
                    # Continue without landmark checks if they fail
            
            # Check 7: Verify embedding exists
            if face.embedding is None:
                logger.warning("❌ Could not extract face embedding")
                return None, "Could not extract face features - please try again", "extraction_error"
            embedding = face.embedding
            logger.info(f"✅ Face embedding extracted successfully: shape={embedding.shape}")
            return embedding, "", ""
            
        except Exception as e:
            logger.error(f"❌ Face embedding extraction error: {e}")
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
        Verify face against stored embedding with enhanced security checks.
        
        Returns dict with:
        - verified: bool - whether face matches stored profile
        - similarity: float - cosine similarity score
        - threshold: float - minimum similarity required
        - error: str or None - specific error message for UI display
        - can_proceed: bool - whether user can continue (always False if not verified)
        - error_type: str - category of error for frontend handling
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
                "error_type": error_type  # Now comes directly from extract_face_embedding
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
    
    def verify_voice(self, student_code: str, audio_data: bytes, audio_format: str = "webm") -> Dict[str, Any]:
        """Verify voice against stored embedding"""
        stored_embedding = self.get_stored_voice_embedding(student_code)
        
        if stored_embedding is None:
            logger.warning(f"⚠️ No voice embedding found for student {student_code} - counting as failure")
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": self.VOICE_SIMILARITY_THRESHOLD,
                "error": "No registered voice found for this student.",
                "is_error": True,
                "skip_warning": False  # ✅ Count as warning - no registered voice
            }
        
        current_embedding, error = self.extract_voice_embedding(audio_data, audio_format)
        
        if current_embedding is None:
            # ✅ Extraction errors should NOT count as warnings - technical issue
            logger.warning(f"⚠️ Voice extraction failed: {error} - skipping verification")
            return {
                "verified": True,  # Don't count as failure
                "similarity": -1.0,
                "threshold": self.VOICE_SIMILARITY_THRESHOLD,
                "error": error,
                "is_extraction_error": True,
                "skip_warning": True  # ✅ Skip warning for technical errors
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
            "skip_warning": False  # ✅ Normal verification - count warning if failed
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
        
        # ✅ FIX: Only increment warning if NOT verified AND NOT skipping
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