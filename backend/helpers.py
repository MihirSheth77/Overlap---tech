import logging
import ffmpeg
import os
import uuid
import cv2
import numpy as np
import mediapipe as mp
import time

# Haar Cascade models for person/face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize MediaPipe selfie segmentation with faster model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)  # Faster model
        
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available filter effects
AVAILABLE_FILTERS = {
    'grayscale': 'Classic Black & White',
    'sepia': 'Vintage Sepia Tone',
    'blur': 'Artistic Blur',
    'edge_detection': 'Edge Detection',
    'vintage': 'Vintage Film Look',
    'warm': 'Warm Color Tone',
    'cool': 'Cool Color Tone',
    'high_contrast': 'High Contrast',
    'soft_light': 'Soft Light',
    'neon': 'Neon Glow Effect'
}

def get_temp_path():
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    random_filename = f"temp_{str(uuid.uuid4())[:8]}"
    return os.path.join(temp_dir, random_filename)

def detect_person_with_haar(frame):
    """
    Optimized Haar Cascade detection - only use face and upper body.
    Returns detailed detection information for transparency.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detection_details = {
            "haar_method": "cv2.CascadeClassifier",
            "models_used": [],
            "detections_found": [],
            "processing_time": 0,
            "frame_size": frame.shape
        }
        
        start_time = time.time()
        
        # Detect faces with optimized parameters for speed
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,  # Faster with larger steps
            minNeighbors=3,   # Reduced for speed
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detection_details["models_used"].append("haarcascade_frontalface_default.xml")
        
        # Only check upper body if no face found
        if len(faces) == 0:
            upper_bodies = upper_body_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3,  # Faster
                minNeighbors=2,   # Reduced for speed
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detection_details["models_used"].append("haarcascade_upperbody.xml")
            
            if len(upper_bodies) > 0:
                x, y, w, h = upper_bodies[0]  # Take first detection
                detection_details["detections_found"].append({
                    "type": "upper_body",
                    "bbox": [x, y, w, h],
                    "confidence": 0.7,
                    "model": "haarcascade_upperbody.xml"
                })
                
                result = {
                    'type': 'upper_body',
                    'region': (x, y, w, h),
                    'confidence': 0.7,
                    'center': (x + w//2, y + h//2),
                    'details': detection_details
                }
            else:
                result = None
        else:
            # Process face detection
            x, y, w, h = faces[0]  # Take first face
            
            detection_details["detections_found"].append({
                "type": "face",
                "bbox": [x, y, w, h],
                "confidence": 0.9,
                "model": "haarcascade_frontalface_default.xml"
            })
            
            # Expand face region
            expanded_x = max(0, x - int(w * 0.3))
            expanded_y = max(0, y - int(h * 0.2))
            expanded_w = min(frame.shape[1] - expanded_x, int(w * 1.6))
            expanded_h = min(frame.shape[0] - expanded_y, int(h * 2.0))
            
            result = {
                'type': 'face',
                'region': (expanded_x, expanded_y, expanded_w, expanded_h),
                'original_face': (x, y, w, h),
                'confidence': 0.9,
                'center': (x + w//2, y + h//2),
                'details': detection_details
            }
        
        detection_details["processing_time"] = (time.time() - start_time) * 1000  # milliseconds
        
        if result:
            result['details'] = detection_details
            logger.info(f"🎯 HAAR CASCADE DETECTION: {result['type']} found using {', '.join(detection_details['models_used'])}")
        else:
            logger.info(f"🎯 HAAR CASCADE: No detection found (checked {len(detection_details['models_used'])} models)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in Haar detection: {e}")
        return None

def apply_filter_effect(frame, filter_type):
    """
    Apply various filter effects to the frame.
    """
    try:
        if filter_type == 'grayscale':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif filter_type == 'sepia':
            # Sepia transformation matrix
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                   [0.349, 0.686, 0.168],
                                   [0.393, 0.769, 0.189]])
            sepia_img = cv2.transform(frame, sepia_filter)
            return np.clip(sepia_img, 0, 255).astype(np.uint8)
        
        elif filter_type == 'blur':
            return cv2.GaussianBlur(frame, (21, 21), 0)
        
        elif filter_type == 'edge_detection':
            # Edge-enhanced effect applied to ENTIRE background
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Create edge-enhanced version by blending original with edge overlay
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Blend original frame with enhanced edges for full background effect
            # This creates an artistic edge-enhanced look on the entire background
            edge_enhanced = cv2.addWeighted(frame, 0.6, edges_colored, 0.4, 0)
            
            return edge_enhanced
        
        elif filter_type == 'vintage':
            # Vintage effect: slight sepia + vignette + noise
            sepia_filter = np.array([[0.393, 0.769, 0.189],
                                   [0.349, 0.686, 0.168],
                                   [0.272, 0.534, 0.131]])
            vintage = cv2.transform(frame, sepia_filter)
            vintage = np.clip(vintage, 0, 255).astype(np.uint8)
            
            # Add vignette effect
            h, w = vintage.shape[:2]
            center_x, center_y = w // 2, h // 2
            Y, X = np.ogrid[:h, :w]
            mask = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            mask = mask / mask.max()
            mask = 1 - mask
            mask = np.clip(mask, 0.3, 1)
            
            for i in range(3):
                vintage[:, :, i] = vintage[:, :, i] * mask
            
            return vintage.astype(np.uint8)
        
        elif filter_type == 'warm':
            # Warm color tone
            warm = frame.astype(np.float32)
            warm[:, :, 0] = warm[:, :, 0] * 0.8  # Reduce blue
            warm[:, :, 2] = warm[:, :, 2] * 1.2  # Increase red
            return np.clip(warm, 0, 255).astype(np.uint8)
        
        elif filter_type == 'cool':
            # Cool color tone
            cool = frame.astype(np.float32)
            cool[:, :, 0] = cool[:, :, 0] * 1.2  # Increase blue
            cool[:, :, 2] = cool[:, :, 2] * 0.8  # Reduce red
            return np.clip(cool, 0, 255).astype(np.uint8)
        
        elif filter_type == 'high_contrast':
            # High contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        elif filter_type == 'soft_light':
            # Soft light effect
            soft = cv2.GaussianBlur(frame, (15, 15), 0)
            return cv2.addWeighted(frame, 0.7, soft, 0.3, 0)
        
        elif filter_type == 'neon':
            # Neon color effect applied to ENTIRE background (like grayscale but neon colors)
            neon_frame = frame.astype(np.float32)
            
            # Create bright neon color transformation
            # Boost green and blue channels, reduce red for cyberpunk neon look
            neon_frame[:, :, 0] = neon_frame[:, :, 0] * 1.8  # Boost blue channel significantly
            neon_frame[:, :, 1] = neon_frame[:, :, 1] * 2.2  # Boost green channel most
            neon_frame[:, :, 2] = neon_frame[:, :, 2] * 0.3  # Reduce red channel for cyan/green effect
            
            # Add brightness and saturation boost for neon glow
            neon_frame = neon_frame * 1.4  # Overall brightness boost
            
            # Apply slight blur for glow effect
            neon_frame = cv2.GaussianBlur(neon_frame, (3, 3), 0)
            
            # Ensure values stay in valid range
            return np.clip(neon_frame, 0, 255).astype(np.uint8)
        
        else:
            return frame
            
    except Exception as e:
        logger.error(f"Error applying filter {filter_type}: {e}")
        return frame

def process_frame_with_hybrid_segmentation(frame, filter_type='grayscale', start_time=None, end_time=None, current_time=0, debug_mode=False):
    """
    Enhanced hybrid approach with minimal logging for performance.
    Since video trimming is handled at the video level, we always apply filters to processed frames.
    """
    try:
        # Always apply filter since we're only processing frames within the desired timeframe
        apply_filter = True
        
        # Debug timeframe logic occasionally (just for logging)
        if start_time is not None and end_time is not None and int(current_time * 10) % 50 == 0:
            logger.info(f"⏱️ TIMEFRAME INFO: Processing frame at {current_time:.2f}s with {filter_type} filter")
        elif int(current_time * 10) % 50 == 0:
            logger.info(f"⏱️ FULL VIDEO: Processing frame at {current_time:.2f}s with {filter_type} filter")
        
        # Only log every 30 frames (1 second at 30fps) to avoid spam
        should_log = int(current_time * 30) % 30 == 0
        
        # STEP 1: Use Haar Cascade for initial detection (ASSIGNMENT REQUIREMENT)
        if should_log:
            logger.info(f"🔍 STEP 1: Running Haar Cascade Detection (cv2.CascadeClassifier)")
        
        haar_start = time.time()
        haar_detection = detect_person_with_haar(frame)
        haar_time = (time.time() - haar_start) * 1000
        
        # STEP 2: Use MediaPipe for segmentation (ENHANCEMENT)
        if should_log:
            logger.info(f"🤖 STEP 2: Running MediaPipe Selfie Segmentation")
        
        mediapipe_start = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_frame)
        mask = results.segmentation_mask
        mediapipe_time = (time.time() - mediapipe_start) * 1000
        
        # STEP 3: Hybrid fusion (COMBINING BOTH APPROACHES)
        if should_log:
            logger.info(f"🔗 STEP 3: Fusing Haar Detection + MediaPipe Segmentation")
        
        fusion_start = time.time()
        
        if haar_detection:
            threshold = 0.5  # More permissive when Haar validates
            fusion_method = f"Haar-validated threshold (detected {haar_detection['type']})"
            if should_log:
                logger.info(f"✅ HYBRID MODE: Haar detected {haar_detection['type']}, using threshold {threshold}")
        else:
            threshold = 0.7  # Higher threshold for pure MediaPipe
            fusion_method = "Pure MediaPipe threshold (no Haar detection)"
            if should_log:
                logger.info(f"🤖 MEDIAPIPE MODE: No Haar detection, using threshold {threshold}")
        
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        # Post-processing
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        fusion_time = (time.time() - fusion_start) * 1000
        
        # STEP 4: Apply selected filter effect to background
        filter_start = time.time()
        mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        filtered_background = apply_filter_effect(frame, filter_type)
        
        # Apply selective filtering: keep person in color, background with selected filter
        result = np.where(mask_3channel > 128, frame, filtered_background)
        filter_time = (time.time() - filter_start) * 1000
        
        # Debug filter application occasionally
        if should_log:
            person_pixels = np.sum(binary_mask > 128)
            total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
            background_pixels = total_pixels - person_pixels
            logger.info(f"🎨 {filter_type.upper()} FILTER: Applied to {background_pixels}/{total_pixels} pixels ({(background_pixels/total_pixels)*100:.1f}% background)")
        
        # Extra debug for neon filter
        if should_log and filter_type == 'neon':
            logger.info(f"💚 NEON GLOW: Enhanced dramatic effect applied with bright green/blue channels!")
        
        # Add debug overlay if requested (but only for demo)
        if debug_mode and haar_detection:
            # Draw Haar detection box for visualization
            if 'original_face' in haar_detection:
                x, y, w, h = haar_detection['original_face']
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result, f"Haar: {haar_detection['type']}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Store minimal processing log only every 2 seconds for API access
        if should_log and int(current_time) % 2 == 0:
            processing_log = {
                "timestamp": current_time,
                "step1_haar": {
                    "method": "OpenCV Haar Cascade (cv2.CascadeClassifier)",
                    "models": ["haarcascade_frontalface_default.xml", "haarcascade_upperbody.xml"],
                    "processing_time_ms": haar_time,
                    "detection_found": haar_detection is not None,
                    "detection_type": haar_detection['type'] if haar_detection else None,
                    "confidence": haar_detection['confidence'] if haar_detection else 0
                },
                "step2_mediapipe": {
                    "method": "Google MediaPipe Selfie Segmentation",
                    "model": "selfie_segmentation (model_selection=0)",
                    "processing_time_ms": mediapipe_time,
                    "mask_shape": mask.shape,
                    "confidence_range": [float(mask.min()), float(mask.max())]
                },
                "step3_fusion": {
                    "method": fusion_method,
                    "threshold_used": threshold,
                    "haar_influenced": haar_detection is not None,
                    "processing_time_ms": fusion_time,
                    "mask_pixels_detected": int(np.sum(binary_mask > 0))
                },
                "final_result": {
                    "filter_applied": filter_type,
                    "filter_time_ms": filter_time,
                    "total_processing_time_ms": haar_time + mediapipe_time + fusion_time + filter_time,
                    "hybrid_approach": "✅ Haar Cascade + MediaPipe + OpenCV Filters"
                }
            }
            
            # Store processing log efficiently (only keep last 5 logs)
            try:
                if not hasattr(process_frame_with_hybrid_segmentation, 'processing_logs'):
                    process_frame_with_hybrid_segmentation.processing_logs = []
                process_frame_with_hybrid_segmentation.processing_logs.append(processing_log)
                # Keep only last 5 logs to prevent memory buildup
                if len(process_frame_with_hybrid_segmentation.processing_logs) > 5:
                    process_frame_with_hybrid_segmentation.processing_logs.pop(0)
            except:
                pass
        
        # Log comprehensive processing info only occasionally
        if should_log:
            total_time = haar_time + mediapipe_time + fusion_time + filter_time
            logger.info(f"📊 FRAME {current_time:.1f}s: Total {total_time:.1f}ms (Haar {haar_time:.1f}ms + MediaPipe {mediapipe_time:.1f}ms + Filter {filter_time:.1f}ms)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in hybrid processing: {e}")
        return frame

def process_video_with_segmentation(input_path, output_path, filter_type='grayscale', start_time=None, end_time=None, playback_speed=1.0):
    """
    Process video with hybrid Haar+MediaPipe segmentation and apply filter to background.
    If start_time and end_time are provided, trim the video to that duration.
    If playback_speed is not 1.0, adjust video speed using FFmpeg.
    """
    try:
        logger.info(f"Starting video processing: {filter_type} at {playback_speed}x speed")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, duration: {video_duration:.2f}s")
        
        # Handle timeframe logic
        start_frame = 0
        end_frame = total_frames
        
        if start_time is not None and end_time is not None:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Validate timeframe
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)
            
            logger.info(f"🎬 TRIMMING VIDEO: {start_time:.1f}s-{end_time:.1f}s (frames {start_frame}-{end_frame})")
            logger.info(f"Output duration will be: {(end_frame - start_frame) / fps:.2f}s")
        else:
            logger.info(f"Processing entire video: {video_duration:.2f}s")
        
        # Set up video writer with browser-compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More browser-compatible than H264
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        output_frame_count = 0
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps")
        logger.info(f"Filter: {filter_type}, Start frame: {start_frame}, End frame: {end_frame}")
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames before start_time
            if frame_count < start_frame:
                frame_count += 1
                continue
                
            # Stop processing after end_time
            if frame_count >= end_frame:
                break
            
            # Calculate current time in seconds (relative to original video)
            current_time = frame_count / fps
                
            # Process frame with selected filter - always apply since we're in the timeframe
            processed_frame = process_frame_with_hybrid_segmentation(
                frame, filter_type, start_time, end_time, current_time, debug_mode=False
            )
            
            # Write processed frame
            out.write(processed_frame)
            
            frame_count += 1
            output_frame_count += 1
            
            # Update progress more frequently
            if output_frame_count % 10 == 0:  # Every 10 output frames
                total_output_frames = end_frame - start_frame
                progress = (output_frame_count / total_output_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({output_frame_count}/{total_output_frames} frames)")
                
                # Update global status if available
                try:
                    from main import processing_status
                    job_ids = list(processing_status.keys())
                    if job_ids:
                        latest_job = job_ids[-1]
                        if processing_status[latest_job]["status"] == "processing":
                            processing_status[latest_job]["progress"] = int(progress)
                except:
                    pass  # Ignore if can't update
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Verify output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            output_duration = output_frame_count / fps
            logger.info(f"✅ Video processing completed successfully!")
            logger.info(f"📁 Output file: {output_path}")
            logger.info(f"📊 Output duration: {output_duration:.2f}s ({output_frame_count} frames)")
            logger.info(f"💾 File size: {os.path.getsize(output_path)} bytes")
            
            # Apply speed adjustment if needed
            if playback_speed != 1.0:
                logger.info(f"🚀 Applying speed adjustment: {playback_speed}x")
                if apply_speed_adjustment(output_path, playback_speed):
                    final_duration = output_duration / playback_speed
                    logger.info(f"🎬 Final video duration after {playback_speed}x speed: {final_duration:.2f}s")
                else:
                    logger.warning("⚠️ Speed adjustment failed, using original video")
            
            return True
        else:
            logger.error(f"❌ Output file is missing or too small: {output_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return False

def apply_speed_adjustment(video_path, speed):
    """
    Apply speed adjustment to video using FFmpeg.
    speed: 0.25 = 4x slower, 0.5 = 2x slower, 2.0 = 2x faster, 4.0 = 4x faster
    """
    try:
        import subprocess
        
        # Create temporary output path
        temp_output = video_path.replace('.mp4', '_speed_adjusted.mp4')
        
        # FFmpeg command for speed adjustment
        # For speed < 1.0 (slow motion): use setpts filter to slow down
        # For speed > 1.0 (fast forward): use setpts filter to speed up
        pts_multiplier = 1.0 / speed  # Inverse of speed for setpts
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-filter:v', f'setpts={pts_multiplier}*PTS',
            '-filter:a', f'atempo={min(speed, 2.0)}',  # Audio tempo (max 2.0 for FFmpeg)
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-y',  # Overwrite output
            temp_output
        ]
        
        # If speed is > 2.0, chain multiple atempo filters
        if speed > 2.0:
            # Chain multiple atempo filters for speeds > 2.0
            atempo_chain = []
            remaining_speed = speed
            while remaining_speed > 2.0:
                atempo_chain.append('atempo=2.0')
                remaining_speed /= 2.0
            if remaining_speed > 1.0:
                atempo_chain.append(f'atempo={remaining_speed}')
            
            audio_filter = ','.join(atempo_chain)
            cmd[cmd.index(f'atempo={min(speed, 2.0)}')] = audio_filter
        
        logger.info(f"🎬 Running FFmpeg speed adjustment: {speed}x")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # Replace original with speed-adjusted version
            import shutil
            shutil.move(temp_output, video_path)
            logger.info(f"✅ Speed adjustment successful: {speed}x")
            return True
        else:
            logger.error(f"❌ FFmpeg error: {result.stderr}")
            # Clean up temp file if it exists
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ FFmpeg speed adjustment timed out")
        return False
    except Exception as e:
        logger.error(f"💥 Error in speed adjustment: {e}")
        return False

def download_video(url, output_path):
    """
    Download video from URL with progress tracking.
    """
    try:
        import requests
        
        logger.info(f"Downloading video from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log download progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Every MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Video downloaded successfully: {output_path}")
        logger.info(f"Downloaded file size: {os.path.getsize(output_path)} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        return False

def save_uploaded_video(file_data, filename):
    """
    Save uploaded video file to temp directory.
    """
    try:
        temp_path = get_temp_path() + "_uploaded.mp4"
        
        with open(temp_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Uploaded video saved: {temp_path}")
        logger.info(f"File size: {os.path.getsize(temp_path)} bytes")
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded video: {e}")
        return None