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
import cv2
from pymongo import MongoClient
from urllib.parse import quote_plus
from scipy.spatial.distance import cosine

# ================== PERSON DETECTION (YOLO) ==================
class PersonDetector:
    """
    ULTRA-SENSITIVE detection for exam security.
    Uses YOLOv8 with aggressive settings to catch even tiny objects.
    """
    
    def __init__(self):
        self._model = None
        
        # COCO class IDs for detection
        self.DETECTION_CLASSES = {
            0: {"name": "person", "type": "person", "emoji": "👤", "min_conf": 0.20, "min_area": 0.005},
            67: {"name": "cell phone", "type": "prohibited", "emoji": "📱", "min_conf": 0.40, "min_area": 0.003},
            63: {"name": "laptop", "type": "prohibited", "emoji": "💻", "min_conf": 0.20, "min_area": 0.01},
            62: {"name": "tv/monitor", "type": "prohibited", "emoji": "🖥️", "min_conf": 0.25, "min_area": 0.02},
            73: {"name": "book", "type": "prohibited", "emoji": "📖", "min_conf": 0.25, "min_area": 0.01},
            74: {"name": "clock/watch", "type": "prohibited", "emoji": "⌚", "min_conf": 0.20, "min_area": 0.001},
            65: {"name": "remote", "type": "prohibited", "emoji": "📱", "min_conf": 0.15, "min_area": 0.0005},
            # Additional classes that might be phones
            77: {"name": "cell phone", "type": "prohibited", "emoji": "📱", "min_conf": 0.10, "min_area": 0.0005},  # Sometimes detected as this
        }
        
    @property
    def model(self):
        """Lazy load YOLOv8 model - use MEDIUM for better small object detection"""
        if self._model is None:
            try:
                from ultralytics import YOLO
                # Use medium model for better accuracy on small objects
                # Try 'm' first, fall back to 's' if not available
                try:
                    self._model = YOLO('yolov8m.pt')
                    logger.info("✅ YOLOv8m (medium) detector loaded")
                except:
                    self._model = YOLO('yolov8s.pt')
                    logger.info("✅ YOLOv8s (small) detector loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load YOLOv8: {e}")
                raise
        return self._model
    
    def detect_persons_and_objects(self, image_data: bytes) -> Dict[str, Any]:
        """
        Run detection with multiple strategies to catch everything:
        1. Normal detection
        2. Enhanced contrast detection
        3. Multi-scale detection
        """
        try:
            import cv2
            
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"error": "Failed to decode image", "person_count": 0}
            
            img_height, img_width = img.shape[:2]
            img_area = img_height * img_width
            
            logger.info(f"🖼️ Processing image: {img_width}x{img_height}")
            
            # ==================== STRATEGY 1: Normal Detection ====================
            all_detections = self._run_detection(img, img_area, "normal")
            
            # ==================== STRATEGY 2: Enhanced Image ====================
            # Increase contrast and brightness to catch dark objects
            enhanced = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
            enhanced_detections = self._run_detection(enhanced, img_area, "enhanced")
            
            # Merge detections (remove duplicates)
            all_detections = self._merge_detections(all_detections, enhanced_detections)
            
            # ==================== STRATEGY 3: Edge Detection Focus ====================
            # Convert to grayscale and enhance edges (helps with phones)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # Blend with original
            blended = cv2.addWeighted(img, 0.7, edges_colored, 0.3, 0)
            edge_detections = self._run_detection(blended, img_area, "edge_enhanced")
            
            all_detections = self._merge_detections(all_detections, edge_detections)
            
            # ==================== Separate persons and objects ====================
            persons = []
            prohibited_objects = []
            
            for det in all_detections:
                if det["type"] == "person":
                    persons.append(det)
                elif det["type"] == "prohibited":
                    prohibited_objects.append(det)
            
            # Sort by area
            persons.sort(key=lambda p: p["area_ratio"], reverse=True)
            prohibited_objects.sort(key=lambda o: o["confidence"], reverse=True)
            
            # Mark main user
            if persons:
                persons[0]["is_main_user"] = True
            
            # Build result
            return self._build_result(persons, prohibited_objects, all_detections)
            
        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "person_count": 0,
                "persons": [],
                "has_multiple_persons": False,
                "has_unauthorized_presence": False,
                "prohibited_objects": [],
                "has_prohibited_objects": False,
                "warning_message": None,
                "violation_type": None,
                "error": str(e)
            }
    
    def _run_detection(self, img, img_area: int, strategy: str) -> List[Dict]:
        """Run YOLO detection with very low confidence threshold"""
        detections = []
        
        try:
            # Run with VERY LOW confidence to catch everything
            results = self.model(img, verbose=False, conf=0.05, iou=0.3)
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    
                    # Skip unknown classes
                    if class_id not in self.DETECTION_CLASSES:
                        continue
                    
                    class_config = self.DETECTION_CLASSES[class_id]
                    confidence = float(box.conf[0])
                    
                    # Apply class-specific confidence threshold
                    if confidence < class_config["min_conf"]:
                        continue
                    
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = bbox
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / img_area
                    
                    # Apply class-specific area threshold
                    if area_ratio < class_config["min_area"]:
                        continue
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    det = {
                        "class_id": class_id,
                        "class_name": class_config["name"],
                        "type": class_config["type"],
                        "emoji": class_config["emoji"],
                        "confidence": round(confidence, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int(center_x), int(center_y)],
                        "area_ratio": round(area_ratio, 5),
                        "strategy": strategy
                    }
                    
                    detections.append(det)
                    
                    if class_config["type"] == "prohibited":
                        logger.warning(
                            f"🚨 [{strategy}] {class_config['emoji']} {class_config['name']}: "
                            f"conf={confidence:.0%}, area={area_ratio:.2%}"
                        )
                    
        except Exception as e:
            logger.error(f"Detection strategy '{strategy}' failed: {e}")
        
        return detections
    
    def _merge_detections(self, list1: List[Dict], list2: List[Dict]) -> List[Dict]:
        """Merge two detection lists, removing duplicates based on IoU"""
        merged = list(list1)
        
        for det2 in list2:
            is_duplicate = False
            
            for det1 in merged:
                if det1["class_id"] == det2["class_id"]:
                    # Check IoU
                    iou = self._calculate_iou(det1["bbox"], det2["bbox"])
                    if iou > 0.3:  # 30% overlap = same object
                        # Keep the one with higher confidence
                        if det2["confidence"] > det1["confidence"]:
                            det1.update(det2)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                merged.append(det2)
        
        return merged
    
    def _calculate_iou(self, box1: List, box2: List) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _build_result(self, persons: List, prohibited_objects: List, all_detections: List) -> Dict:
        """Build final result"""
        person_count = len(persons)
        has_multiple = person_count > 1
        has_prohibited = len(prohibited_objects) > 0
        
        warning_message = None
        violation_type = None
        unauthorized = False
        
        # Priority 1: Prohibited objects
        if has_prohibited:
            unauthorized = True
            violation_type = "prohibited_object"
            
            obj = prohibited_objects[0]
            if len(prohibited_objects) == 1:
                warning_message = f"{obj['emoji']} {obj['class_name'].title()} detected in frame"
            else:
                names = [f"{o['emoji']} {o['class_name']}" for o in prohibited_objects[:3]]
                warning_message = f"Prohibited items: {', '.join(names)}"
        
        # Priority 2: Multiple persons
        elif has_multiple:
            unauthorized = True
            violation_type = "multiple_persons"
            warning_message = f"👥 {person_count} people detected - only you should be visible"
        
        # Priority 3: No person
        elif person_count == 0:
            violation_type = "no_person"
            warning_message = "👤 No person detected - please stay in frame"
        
        logger.info(
            f"📊 Result: {person_count} person(s), {len(prohibited_objects)} prohibited, "
            f"violation={violation_type}"
        )
        
        return {
            "person_count": person_count,
            "persons": persons,
            "has_multiple_persons": has_multiple,
            "has_unauthorized_presence": unauthorized,
            "prohibited_objects": prohibited_objects,
            "has_prohibited_objects": has_prohibited,
            "warning_message": warning_message,
            "violation_type": violation_type,
            "all_detections_count": len(all_detections),
            "error": None
        }
    
    # Alias
    def detect_persons(self, image_data: bytes) -> Dict[str, Any]:
        return self.detect_persons_and_objects(image_data)

# Global person detector instance
person_detector: Optional[PersonDetector] = None


def get_person_detector() -> PersonDetector:
    """Get or create the global person detector instance"""
    global person_detector
    if person_detector is None:
        person_detector = PersonDetector()
    return person_detector

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
    
    def extract_face_embedding(self, image_data: bytes) -> Tuple[Optional[np.ndarray], str, str]:
        """
        Extract face embedding with FIXED attention detection.
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
            
            # ==================== ATTENTION DETECTION (FIXED) ====================
            pose = getattr(face, 'pose', None)
            
            if pose is not None and len(pose) >= 3:
                # InsightFace returns pose as [pitch, yaw, roll] in DEGREES
                # BUT the order and signs might vary by model version
                
                # Log raw values for debugging
                raw_pitch = float(pose[0])
                raw_yaw = float(pose[1])
                raw_roll = float(pose[2])
                
                logger.info(f"👀 Raw pose: pitch={raw_pitch:.1f}, yaw={raw_yaw:.1f}, roll={raw_roll:.1f}")
                
                # Normalize - some models return different ranges
                # Typically: yaw (left/right), pitch (up/down), roll (tilt)
                
                # Try to detect which value is which based on typical ranges
                # Yaw usually has the largest range when looking left/right
                
                # Use absolute values for thresholds
                abs_yaw = abs(raw_yaw)
                abs_pitch = abs(raw_pitch)
                abs_roll = abs(raw_roll)
                
                # ===== YAW: Looking LEFT/RIGHT =====
                # Threshold: 20 degrees = definitely looking away
                YAW_THRESHOLD = 20
                if abs_yaw > YAW_THRESHOLD:
                    direction = "left" if raw_yaw > 0 else "right"
                    logger.warning(f"👀 LOOKING {direction.upper()}: yaw={raw_yaw:.1f}°")
                    return None, f"👀 Looking {direction} - please face the camera", "not_looking_at_camera"
                
                # ===== PITCH: Looking UP/DOWN =====
                # Threshold: 15 degrees for down (reading), 20 for up
                PITCH_DOWN_THRESHOLD = 15
                PITCH_UP_THRESHOLD = 20
                
                # Positive pitch usually means looking down
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
                        # Use nose tip and face center to estimate pose
                        # Nose tip is usually around index 54-55
                        nose_tip = landmarks[54] if len(landmarks) > 54 else None
                        
                        # Face center from bbox
                        face_center_x = (bbox[0] + bbox[2]) / 2
                        face_center_y = (bbox[1] + bbox[3]) / 2
                        
                        if nose_tip is not None:
                            # Calculate horizontal offset (yaw estimate)
                            nose_offset_x = (nose_tip[0] - face_center_x) / face_width
                            # Calculate vertical offset (pitch estimate)
                            nose_offset_y = (nose_tip[1] - face_center_y) / face_height
                            
                            logger.info(f"👃 Nose offset: x={nose_offset_x:.2f}, y={nose_offset_y:.2f}")
                            
                            # If nose is significantly off-center, person is looking away
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
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name
            
            # Convert to WAV first — raw webm chunks may lack container headers
            wav_path = tmp_path.replace(f".{audio_format}", ".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True, timeout=10
                )
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                    load_path = wav_path
                else:
                    load_path = tmp_path  # fallback to original
            except Exception:
                load_path = tmp_path  # ffmpeg not available, try direct load
            
            try:
                waveform, sample_rate = torchaudio.load(load_path)       
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
                for p in [tmp_path, wav_path]:
                    try:
                        os.unlink(p)
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

    def verify_face_with_person_detection(self, student_code: str, image_data: bytes) -> Dict[str, Any]:
        """
        Comprehensive verification that checks ALL security violations:
        
        Priority Order (to avoid false positives):
        1. Face attention check FIRST (looking at camera, pose) - InsightFace
        2. Face identity verification - InsightFace  
        3. Multiple persons detection - YOLO
        4. Prohibited objects (phones, laptops, etc.) - YOLO (only if face is OK)
        
        This order prevents YOLO false positives when user bends down.
        """
        
        # ==================== STEP 1: FACE VERIFICATION FIRST ====================
        # Check face attention and identity BEFORE running YOLO
        # This catches "looking down", "no face", "looking away" etc.
        face_result = self.verify_face(student_code, image_data)
        
        # If face verification failed due to attention/pose issues, return immediately
        # Don't run YOLO - the user is not looking at camera, that's the real issue
        face_error_type = face_result.get("error_type", "")
        if face_error_type in ["no_face", "not_looking_at_camera", "looking_down", "head_tilted", 
                            "face_too_small", "face_too_close", "poor_quality", "eyes_not_visible"]:
            logger.warning(f"👀 ATTENTION VIOLATION: {face_result.get('error')} (type: {face_error_type})")
            
            face_result["violation_type"] = "attention_violation"
            face_result["person_count"] = 1  # Assume person is there but not looking
            face_result["prohibited_objects"] = []
            face_result["detection_method"] = "face_attention"
            return face_result
        
        # ==================== STEP 2: YOLO DETECTION (only if face is OK) ====================
        detector = get_person_detector()
        detection_result = detector.detect_persons_and_objects(image_data)
        
        if detection_result.get("error"):
            logger.warning(f"Detection error: {detection_result['error']} - continuing with face result")
        
        # Check for multiple persons
        elif detection_result["has_multiple_persons"]:
            logger.warning(f"👥 MULTIPLE PERSONS: {detection_result['warning_message']}")
            
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": self.FACE_SIMILARITY_THRESHOLD,
                "error": detection_result["warning_message"],
                "can_proceed": False,
                "error_type": "multiple_persons",
                "violation_type": "multiple_persons",
                "person_count": detection_result["person_count"],
                "detection_method": "yolo_person"
            }
        
        # Check for prohibited objects
        elif detection_result["has_prohibited_objects"]:
            objects = detection_result.get("prohibited_objects", [])
            
            logger.warning(f"🚨 PROHIBITED OBJECT: {detection_result['warning_message']}")
            
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": self.FACE_SIMILARITY_THRESHOLD,
                "error": detection_result["warning_message"],
                "can_proceed": False,
                "error_type": "prohibited_object",
                "violation_type": "prohibited_object",
                "prohibited_objects": [o["class_name"] for o in objects],
                "person_count": detection_result["person_count"],
                "detection_method": "yolo_object",
                "detection_details": objects[:3]
            }
        
        # ==================== STEP 3: RETURN FACE RESULT ====================
        # Add detection info
        face_result["person_count"] = detection_result.get("person_count", 1)
        face_result["prohibited_objects"] = []
        face_result["detection_method"] = "face_and_object"
        
        # Map specific error types for frontend
        if face_error_type == "face_mismatch":
            face_result["violation_type"] = "identity_mismatch"
        
        return face_result

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

# ================== LIGHTWEIGHT PROCTORING (CONTINUOUS) ==================

    def quick_proctoring_check(self, image_data: bytes, 
                                phone_confidence_threshold: float = 0.40,
                                head_yaw_threshold: float = 35.0) -> Dict[str, Any]:
        """
        Lightweight check for continuous interview proctoring.
        Only checks TWO things (fast):
        1. Phone detection via YOLOv8 (nano model for speed)
        2. Head turn via InsightFace pose estimation (yaw angle)
        
        Returns:
            {
                "phone_detected": bool,
                "phone_confidence": float,
                "head_turned": bool,
                "head_yaw": float,
                "face_detected": bool,
                "violations": ["phone", "head_turn"],  # list of violations found
                "error": str or None
            }
        """
        result = {
            "phone_detected": False,
            "phone_confidence": 0.0,
            "head_turned": False,
            "head_yaw": 0.0,
            "face_detected": False,
            "violations": [],
            "error": None
        }
        
        try:
            import cv2
            
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                result["error"] = "Failed to decode image"
                return result
            
            img_height, img_width = img.shape[:2]
            img_area = img_height * img_width
            
            # ===== CHECK 1: Phone Detection via YOLOv8 =====
            try:
                detector = get_person_detector()
                yolo_results = detector.model(img, verbose=False, conf=0.05, iou=0.3)
                
                for yolo_result in yolo_results:
                    for box in yolo_result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # COCO class 67 = cell phone, 77 = alternate phone class
                        if class_id in (67, 77) and confidence >= phone_confidence_threshold:
                            result["phone_detected"] = True
                            result["phone_confidence"] = round(confidence, 3)
                            result["violations"].append("phone")
                            logger.warning(f"📱 PHONE DETECTED: confidence={confidence:.1%}")
                            break  # one phone is enough
                    if result["phone_detected"]:
                        break
                        
            except Exception as yolo_err:
                logger.error(f"YOLOv8 proctoring error: {yolo_err}")
            
            # ===== CHECK 2: Head Turn via InsightFace =====
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = self.face_analyzer.get(img_rgb)
                
                if len(faces) >= 1:
                    result["face_detected"] = True
                    face = faces[0]
                    
                    pose = getattr(face, 'pose', None)
                    if pose is not None and len(pose) >= 3:
                        raw_yaw = float(pose[1])  # yaw = left/right
                        result["head_yaw"] = round(raw_yaw, 1)
                        
                        if abs(raw_yaw) > head_yaw_threshold:
                            result["head_turned"] = True
                            direction = "left" if raw_yaw > 0 else "right"
                            result["violations"].append("head_turn")
                            logger.warning(f"👀 HEAD TURNED {direction.upper()}: yaw={raw_yaw:.1f}°")
                else:
                    # No face detected — could be they left the frame
                    logger.info("👤 No face detected in proctoring frame")
                    
            except Exception as face_err:
                logger.error(f"InsightFace proctoring error: {face_err}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Proctoring check error: {e}")
            result["error"] = str(e)
            return result

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

class PhoneViolationTracker:
    """Tracks phone detection violations during interview sessions"""
    
    def __init__(self, max_warnings: int = 3):
        self.max_warnings = max_warnings
        self.sessions: Dict[str, Dict] = {}
    
    def start_session(self, session_id: str, student_code: str):
        self.sessions[session_id] = {
            "student_code": student_code,
            "warning_count": 0,
            "detection_history": [],
            "started_at": datetime.utcnow(),
            "terminated": False,
        }
        logger.info(f"📱 Phone violation tracking started for session {session_id}")
    
    def record_violation(self, session_id: str, confidence: float) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {"warning_count": 0, "should_terminate": False, "message": "Session not found"}
        
        session = self.sessions[session_id]
        if session["terminated"]:
            return {"warning_count": session["warning_count"], "should_terminate": True, "message": "Already terminated"}
        
        session["warning_count"] += 1
        session["detection_history"].append({
            "timestamp": datetime.utcnow(),
            "confidence": confidence
        })
        
        warning_count = session["warning_count"]
        
        if warning_count >= self.max_warnings:
            session["terminated"] = True
            logger.warning(f"🛑 Session {session_id} TERMINATED: {warning_count} phone detections")
            return {
                "warning_count": warning_count,
                "should_terminate": True,
                "message": f"Session terminated: Phone detected {warning_count} times"
            }
        
        remaining = self.max_warnings - warning_count
        return {
            "warning_count": warning_count,
            "should_terminate": False,
            "message": f"Warning {warning_count}/{self.max_warnings}: Phone detected. {remaining} warning(s) remaining."
        }
    
    def end_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]


class HeadTurnTracker:
    """Tracks head turn violations during interview sessions"""
    
    def __init__(self, max_warnings: int = 3):
        self.max_warnings = max_warnings
        self.sessions: Dict[str, Dict] = {}
    
    def start_session(self, session_id: str, student_code: str):
        self.sessions[session_id] = {
            "student_code": student_code,
            "warning_count": 0,
            "detection_history": [],
            "started_at": datetime.utcnow(),
            "terminated": False,
        }
        logger.info(f"👀 Head turn tracking started for session {session_id}")
    
    def record_violation(self, session_id: str, yaw_angle: float) -> Dict[str, Any]:
        if session_id not in self.sessions:
            return {"warning_count": 0, "should_terminate": False, "message": "Session not found"}
        
        session = self.sessions[session_id]
        if session["terminated"]:
            return {"warning_count": session["warning_count"], "should_terminate": True, "message": "Already terminated"}
        
        session["warning_count"] += 1
        session["detection_history"].append({
            "timestamp": datetime.utcnow(),
            "yaw_angle": yaw_angle
        })
        
        warning_count = session["warning_count"]
        
        if warning_count >= self.max_warnings:
            session["terminated"] = True
            logger.warning(f"🛑 Session {session_id} TERMINATED: {warning_count} head turn violations")
            return {
                "warning_count": warning_count,
                "should_terminate": True,
                "message": f"Session terminated: Looked away {warning_count} times"
            }
        
        remaining = self.max_warnings - warning_count
        direction = "left" if yaw_angle > 0 else "right"
        return {
            "warning_count": warning_count,
            "should_terminate": False,
            "message": f"Warning {warning_count}/{self.max_warnings}: Looking {direction} detected. {remaining} warning(s) remaining."
        }
    
    def end_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

# ================== GLOBAL INSTANCES ==================
biometric_service: Optional[BiometricAuthService] = None
voice_tracker: Optional[VoiceVerificationTracker] = None
face_tracker: Optional[FaceVerificationTracker] = None
phone_tracker: Optional[PhoneViolationTracker] = None
head_turn_tracker: Optional[HeadTurnTracker] = None


def init_biometric_services(
    mongo_host: str = "192.168.48.201",
    mongo_port: int = 27017,
    db_name: str = "connectlydb",
    username: str = "connectly",
    password: str = "LT@connect25",
    auth_source: str = "admin",
    max_voice_warnings: int = 3,
    max_face_warnings: int = 3,
    max_phone_warnings: int = 3,
    max_head_turn_warnings: int = 3
) -> Tuple[BiometricAuthService, VoiceVerificationTracker, FaceVerificationTracker]:
    """Initialize biometric services - call this at app startup"""
    global biometric_service, voice_tracker, face_tracker, phone_tracker, head_turn_tracker
    
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
    phone_tracker = PhoneViolationTracker(max_warnings=max_phone_warnings)
    head_turn_tracker = HeadTurnTracker(max_warnings=max_head_turn_warnings)
    
    logger.info("✅ Biometric services initialized (voice + face + phone + head_turn)")
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

def get_phone_tracker() -> Optional[PhoneViolationTracker]:
    """Get the global phone violation tracker instance"""
    return phone_tracker


def get_head_turn_tracker() -> Optional[HeadTurnTracker]:
    """Get the global head turn tracker instance"""
    return head_turn_tracker