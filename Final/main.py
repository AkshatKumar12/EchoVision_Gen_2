import cv2
import torch
import numpy as np
from transformers import pipeline as hf_pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
import time
from datetime import datetime
import google.generativeai as genai


class BaseModel(ABC):
    """Base class for all models"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.position = (20, 40)
    
    @abstractmethod
    def load(self): ...
    
    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, Any]: ...
    
    @abstractmethod
    def draw(self, frame: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray: ...


class FaceRecognitionModel(BaseModel):
    def __init__(self, dataset_dir: str = "echoDataset", accuracy_threshold: float = 60.0):
        super().__init__("Face Recognition")
        self.dataset_dir = dataset_dir
        self.accuracy_threshold = accuracy_threshold
        self.img_size = (160, 160)
        self.margin = 30
        self.position = (20, 40)
        
    def preprocess_face(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        resized = cv2.resize(equalized, self.img_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return rgb
    
    def load_dataset_embeddings(self):
        embeddings, labels, label_map = [], [], {}
        
        if not os.path.exists(self.dataset_dir):
            print(f"⚠️ Dataset directory '{self.dataset_dir}' not found")
            return np.array([]), np.array([]), {}
        
        for i, person in enumerate(os.listdir(self.dataset_dir)):
            person_dir = os.path.join(self.dataset_dir, person)
            if not os.path.isdir(person_dir):
                continue
            
            label_map[i] = person
            for img_file in os.listdir(person_dir):
                path = os.path.join(person_dir, img_file)
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                
                img = cv2.imread(path)
                if img is None:
                    continue
                
                processed = self.preprocess_face(img)
                tensor = torch.tensor(processed, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    emb = self.resnet(tensor.to(self.device)).cpu().numpy()
                embeddings.append(emb[0])
                labels.append(i)
        
        if not embeddings:
            print(f"⚠️ No face images found in {self.dataset_dir}")
            return np.array([]), np.array([]), {}
        
        return np.array(embeddings), np.array(labels), label_map
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def load(self):
        print(f"⏳ Loading {self.name} model...")
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mtcnn = MTCNN(keep_all=True, device=self.device)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            print(f"   Loading dataset embeddings from {self.dataset_dir}...")
            self.dataset_embs, self.dataset_labels, self.label_map = self.load_dataset_embeddings()
            if len(self.dataset_embs) > 0:
                print(f"✅ {self.name}: {len(self.dataset_embs)} embeddings, {len(self.label_map)} identities")
            else:
                print(f"✅ {self.name}: Detection only")
        except Exception as e:
            print(f"⚠️ Failed to load {self.name}: {e}")
            self.enabled = False
    
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        boxes, _ = self.mtcnn.detect(frame)
        results = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1 - self.margin), max(0, y1 - self.margin)
                x2, y2 = min(frame.shape[1], x2 + self.margin), min(frame.shape[0], y2 + self.margin)
                face_result = {'bbox': (x1, y1, x2, y2), 'name': 'Unknown', 'confidence': 0.0}
                if len(self.dataset_embs) > 0:
                    try:
                        face_roi = frame[y1:y2, x1:x2]
                        processed = self.preprocess_face(face_roi)
                        tensor = torch.tensor(processed, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                        with torch.no_grad():
                            emb = self.resnet(tensor.to(self.device)).cpu().numpy()[0]
                        sims = [self.cosine_similarity(emb, e) for e in self.dataset_embs]
                        best_idx = int(np.argmax(sims))
                        best_score = sims[best_idx] * 100
                        best_name = self.label_map[self.dataset_labels[best_idx]]
                        if best_score >= self.accuracy_threshold:
                            face_result['name'] = best_name
                            face_result['confidence'] = best_score
                        else:
                            face_result['confidence'] = best_score
                    except Exception as e:
                        print(f"Recognition error: {e}")
                results.append(face_result)
        return {'faces': results, 'count': len(results)}
    
    def draw(self, frame: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        for face in prediction['faces']:
            x1, y1, x2, y2 = face['bbox']
            color = (0, 255, 0) if face['name'] != 'Unknown' else (0, 0, 255)
            label = f"{face['name']} ({face['confidence']:.1f}%)" if face['confidence'] > 0 else face['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"{self.name}: {prediction['count']} face(s)", self.position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return frame


class HARModel(BaseModel):
    def __init__(self, model_name="Harsha901/vit-human-pose-classification-model"):
        super().__init__("Activity Recognition")
        self.model_name = model_name
        self.position = (20, 80)
        
    def load(self):
        print(f"⏳ Loading {self.name} model...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            print(f"✅ {self.name} loaded on {self.device.upper()}")
        except Exception as e:
            print(f"⚠️ Failed to load {self.name}: {e}")
            self.enabled = False
    
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_class = torch.max(probs, dim=1)
        return {"activity": self.model.config.id2label[top_class.item()],
                "confidence": top_prob.item()}
    
    def draw(self, frame: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        text = f"{self.name}: {prediction['activity']} ({prediction['confidence']*100:.1f}%)"
        cv2.putText(frame, text, self.position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2)
        return frame


class BackgroundRecognitionModel(BaseModel):
    """Scene/Background recognition using hansin91/scene_classification (ViT)"""
    
    def __init__(self, model_name="hansin91/scene_classification"):
        super().__init__("Background Recognition")
        self.model_name = model_name
        self.position = (20, 120)
    
    def load(self):
        print(f"⏳ Loading {self.name} model...")
        try:
            from transformers import pipeline
            self.device = 0 if torch.cuda.is_available() else -1
            self.scene_classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=self.device
            )
            print(f"✅ {self.name} loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load {self.name}: {e}")
            self.enabled = False
    
    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Predict indoor scene type from frame."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            results = self.scene_classifier(image)

            if len(results) > 0:
                top = results[0]
                return {"scene": top["label"], "confidence": top["score"]}
            else:
                return {"scene": "Unknown", "confidence": 0.0}
        except Exception as e:
            print(f"⚠️ Background prediction failed: {e}")
            return {"scene": "Error", "confidence": 0.0}
    
    def draw(self, frame: np.ndarray, prediction: Dict[str, Any]) -> np.ndarray:
        """Overlay scene prediction on frame."""
        text = f"{self.name}: {prediction['scene']} ({prediction['confidence']*100:.1f}%)"
        cv2.putText(frame, text, self.position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        return frame


class GeminiContextBuilder:
    """Handles Gemini API calls to build context from detection data"""
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini API client
        
        Args:
            api_key: Your Google Gemini API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def build_context(self, detection_data: Dict[str, Any]) -> str:
        """
        Send detection data to Gemini and get contextual analysis
        
        Args:
            detection_data: Dictionary containing face, activity, and scene data
            
        Returns:
            Contextual analysis string from Gemini
        """
        try:
            prompt = self._create_prompt(detection_data)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return f"Error: {str(e)}"
    
    def _create_prompt(self, data: Dict[str, Any]) -> str:
        """Create a descriptive prompt from detection data"""
        faces = data.get('faces', [])
        activity = data.get('activity', {})
        scene = data.get('scene', {})
        
       # prompt = "Provide a brief 2-3 sentence contextual analysis:\n\n"

        prompt = """You are an assistive AI for visually impaired users. Given the detected scene elements, objects, and actions, generate a 1–2 line natural audio description that quickly tells the user what’s happening around them.Keep it clear, calm, and real-time suitable (like something that could be spoken through earphones). Avoid unnecessary details.\n\n"""
        
        if faces:
            prompt += f"People detected: {len(faces)}\n"
            for i, face in enumerate(faces, 1):
                name = face.get('name', 'Unknown')
                conf = face.get('confidence', 0)
                prompt += f"  Person {i}: {name} ({conf:.1f}% confidence)\n"
        else:
            prompt += "No people detected.\n"
        
        if activity:
            act = activity.get('activity', 'Unknown')
            conf = activity.get('confidence', 0) * 100
            prompt += f"Activity: {act} ({conf:.1f}% confidence)\n"
        
        if scene:
            scene_type = scene.get('scene', 'Unknown')
            conf = scene.get('confidence', 0) * 100
            prompt += f"Scene: {scene_type} ({conf:.1f}% confidence)\n"
        
        prompt += "\nDescribe what's happening in 2-3 sentences."
        return prompt


class ModelPipeline:
    def __init__(self, camera_source=0, use_ip_cam=False, ip_cam_url="", 
                 gemini_api_key=None, snapshot_dir="snapshots"):
        self.models: List[BaseModel] = []
        self.use_ip_cam = use_ip_cam
        self.camera_source = ip_cam_url if use_ip_cam else camera_source
        self.cap = None
        self.frame_skip = 15  # Process every 15th frame (~2 times per second at 30fps)
        self.frame_count = 0
        self.cached_predictions = {}
        
        # Snapshot settings
        self.snapshot_dir = snapshot_dir
        self.snapshot_interval = 7.0  # seconds - more reasonable for Gemini analysis
        self.last_snapshot_time = time.time()
        
        # Gemini integration
        self.gemini_api_key = gemini_api_key
        self.gemini_builder = None
        self.current_context = "Waiting for first analysis..."
        
        if gemini_api_key:
            self.gemini_builder = GeminiContextBuilder(gemini_api_key)
            print("✅ Gemini API integration enabled")
        
        # Create snapshot directory
        os.makedirs(snapshot_dir, exist_ok=True)
        
    def add_model(self, model: BaseModel):
        self.models.append(model)
        self.cached_predictions[model.name] = None
        
    def load_all(self):
        for model in self.models:
            model.load()
    
    def initialize_camera(self, width=640, height=480):
        self.cap = cv2.VideoCapture(self.camera_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ Could not open camera: {self.camera_source}")
        if not self.use_ip_cam:
            self.cap.set(3, width)
            self.cap.set(4, height)
        print("✅ Camera initialized")
    
    def _prepare_detection_data(self) -> Dict[str, Any]:
        """Prepare current predictions for Gemini"""
        data = {}
        
        # Add face recognition data
        face_pred = self.cached_predictions.get("Face Recognition")
        if face_pred:
            data["faces"] = [
                {
                    "name": f.get("name", "Unknown"),
                    "confidence": float(f.get("confidence", 0))
                }
                for f in face_pred.get("faces", [])
            ]
        
        # Add activity recognition data
        activity_pred = self.cached_predictions.get("Activity Recognition")
        if activity_pred:
            data["activity"] = {
                "activity": activity_pred.get("activity", "Unknown"),
                "confidence": float(activity_pred.get("confidence", 0))
            }
        
        # Add background/scene data
        scene_pred = self.cached_predictions.get("Background Recognition")
        if scene_pred:
            data["scene"] = {
                "scene": scene_pred.get("scene", "Unknown"),
                "confidence": float(scene_pred.get("confidence", 0))
            }
        
        return data
    
    def _save_snapshot(self, frame: np.ndarray):
        """Save current frame as snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.snapshot_dir, f"snapshot_{timestamp}.jpg")
        
        try:
            cv2.imwrite(filename, frame)
            print(f"📸 Snapshot saved: {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save snapshot: {e}")
    
    def _get_gemini_context(self, frame: np.ndarray):
        """Get Gemini context and save snapshot"""
        if not self.gemini_builder:
            return
        
        try:
            # Save snapshot
            self._save_snapshot(frame)
            
            # Get detection data
            detection_data = self._prepare_detection_data()
            
            # Get context from Gemini (show loading message)
            self.current_context = "⏳ Analyzing scene with Gemini..."
            print("🤖 Requesting Gemini analysis...")
            
            context = self.gemini_builder.build_context(detection_data)
            self.current_context = context
            print(f"🤖 Context: {context}")
            
        except Exception as e:
            print(f"⚠️ Failed to get Gemini context: {e}")
            self.current_context = f"Error: {str(e)}"
    
    def _draw_context_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw Gemini context on frame"""
        if not self.current_context:
            return frame
        
        # Create semi-transparent overlay at bottom
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Calculate text area
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 10
        line_height = 20
        
        # Split context into lines
        max_width = width - 2 * padding
        words = self.current_context.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (tw, th), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if tw <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Draw background rectangle
        panel_height = len(lines) * line_height + 2 * padding + 20
        cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw title
        cv2.putText(frame, "🤖 Gemini Analysis:", (padding, height - panel_height + 20),
                    font, 0.6, (0, 255, 255), 2)
        
        # Draw context lines
        y_offset = height - panel_height + 45
        for line in lines:
            cv2.putText(frame, line, (padding, y_offset),
                        font, font_scale, (255, 255, 255), thickness)
            y_offset += line_height
        
        return frame
    
    def run(self):
        if self.cap is None:
            raise RuntimeError("Camera not initialized.")
        
        enabled = [m for m in self.models if m.enabled]
        print("\n🎥 Active models:", ", ".join([m.name for m in enabled]))
        print(f"📸 Snapshot & Gemini analysis interval: {self.snapshot_interval}s")
        print(f"🔄 Model prediction frequency: Every {self.frame_skip} frames (~{30/self.frame_skip:.1f} times/sec at 30fps)")
        print(f"📁 Snapshot directory: {self.snapshot_dir}")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Frame not captured, exiting...")
                break
            
            self.frame_count += 1
            
            # Run predictions on frame skip
            if self.frame_count % self.frame_skip == 0:
                for m in self.models:
                    if not m.enabled:
                        continue
                    try:
                        pred = m.predict(frame)
                        self.cached_predictions[m.name] = pred
                    except Exception as e:
                        print(f"⚠️ Error in {m.name}: {e}")
            
            # Check if it's time for snapshot and Gemini analysis
            current_time = time.time()
            if current_time - self.last_snapshot_time >= self.snapshot_interval:
                self._get_gemini_context(frame)
                self.last_snapshot_time = current_time
            
            # Draw predictions on frame
            for m in self.models:
                if not m.enabled or self.cached_predictions[m.name] is None:
                    continue
                try:
                    frame = m.draw(frame, self.cached_predictions[m.name])
                except Exception as e:
                    print(f"⚠️ Draw error in {m.name}: {e}")
            
            # Draw Gemini context panel
            frame = self._draw_context_panel(frame)
            
            # Display frame
            cv2.imshow("Multi-Model Vision with Gemini", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Pipeline closed.")


if __name__ == "__main__":
    print("\n🚀 Multi-Model Vision Pipeline with Gemini Display\n")
    
    # IMPORTANT: Replace with your actual Gemini API key
    GEMINI_API_KEY = "AIzaSyCKlUQpeuLi5sfS149KPh1cbgbAqTmLsJI"  # Get from https://makersuite.google.com/app/apikey
    
    # Initialize pipeline with Gemini integration
    pipeline = ModelPipeline(
        gemini_api_key=GEMINI_API_KEY,
        snapshot_dir="snapshots"
    )
    
    # Add models
    pipeline.add_model(FaceRecognitionModel(dataset_dir="echoDataset", accuracy_threshold=60.0))
    pipeline.add_model(HARModel(model_name="Harsha901/vit-human-pose-classification-model"))
    pipeline.add_model(BackgroundRecognitionModel())
    
    # Load and run
    pipeline.load_all()
    pipeline.initialize_camera(640, 480)
    pipeline.run()