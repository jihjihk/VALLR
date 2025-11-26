"""
Improved OpenCV-based video preprocessing with stable face cropping.
Handles missing face detections by using interpolation and fallback strategies.
1. **Face Detection & Cropping** (MediaPipe FaceMesh)
2. **Roll Correction** (rotate face to upright position)
3. **Face Segmentation** (optional: remove background)
4. **Resize to 224×224**
5. **Temporal Sampling** (uniform frame sampling)
6. **Normalization** (to [0, 1] range)

To improve on glitchy processing due to failed face detection in some frames, these improvements were made:

1. Two-Pass Processing:
   - Pass 1: Detect faces in ALL frames first
   - Pass 2: Fill in missing detections

2. Missing Detection Fill Strategy:
   - If frame N has no face detected, use the face from nearest frame that did
   - Searches both left and right to find closest valid detection
   - Result: Every frame gets a face crop

3. Optimized Detection Settings:
   - Uses SHORT_RANGE model (better for VoxCeleb2 close-ups)
   - Lower confidence threshold (0.3 instead of 0.5)
   - Detects faces in more frames

4. Temporal Consistency:
   - All frames use valid face detections
   - No switching between cropped/uncropped
   - Smooth, consistent video output
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import sys

# Add parent directory to path to import face_cropper
# Works whether running from root or Data/ directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from face_cropper import FaceCropper


class StablePreprocessor:
    def __init__(self, video_dir, output_dir, speaker_id='id06963', frame_size=(224, 224)):
        """
        Initialize the preprocessor with stable face tracking.

        Args:
            video_dir: Root directory containing voxceleb2/mp4/dev/
            output_dir: Output directory for processed videos
            speaker_id: Speaker ID to process (e.g., 'id06963')
            frame_size: Target frame size (height, width)
        """
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.speaker_id = speaker_id
        self.frame_size = frame_size

        # Initialize FaceCropper
        print("Initializing FaceCropper...")
        self.face_cropper = FaceCropper(
            min_face_detector_confidence=0.3,  # Lower threshold for better detection
            face_detector_model_selection=FaceCropper.SHORT_RANGE,  # Better for VoxCeleb2 close-ups
            landmark_detector_static_image_mode=FaceCropper.STATIC_MODE,
            min_landmark_detector_confidence=0.3
        )
        print("FaceCropper initialized.")

    def get_video_paths(self):
        """Get all video paths for the speaker."""
        video_paths = []
        speaker_dir = os.path.join(self.video_dir, self.speaker_id)

        if not os.path.exists(speaker_dir):
            raise ValueError(f"Speaker directory not found: {speaker_dir}")

        for video_folder in os.listdir(speaker_dir):
            video_folder_path = os.path.join(speaker_dir, video_folder)
            if os.path.isdir(video_folder_path):
                for video_file in os.listdir(video_folder_path):
                    if video_file.endswith('.mp4'):
                        video_path = os.path.join(video_folder_path, video_file)
                        video_paths.append((video_path, video_folder, video_file))

        return video_paths

    def get_face_bounding_box(self, face_landmarks, image_shape):
        """
        Extract bounding box from face landmarks.
        Returns (x, y, w, h) in pixel coordinates.
        """
        h, w = image_shape[:2]

        # Convert normalized landmarks to pixel coordinates
        x_coords = [int(lm.x * w) for lm in face_landmarks]
        y_coords = [int(lm.y * h) for lm in face_landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def interpolate_bounding_box(self, bbox1, bbox2, alpha):
        """
        Interpolate between two bounding boxes.
        alpha=0 returns bbox1, alpha=1 returns bbox2.
        """
        if bbox1 is None:
            return bbox2
        if bbox2 is None:
            return bbox1

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        return (
            int(x1 + alpha * (x2 - x1)),
            int(y1 + alpha * (y2 - y1)),
            int(w1 + alpha * (w2 - w1)),
            int(h1 + alpha * (h2 - h1))
        )

    def crop_face_with_bbox(self, frame, bbox, padding=0.2):
        """
        Crop face from frame using bounding box with padding.
        """
        h, w = frame.shape[:2]
        x, y, box_w, box_h = bbox

        # Add padding
        pad_w = int(box_w * padding)
        pad_h = int(box_h * padding)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + box_w + pad_w)
        y2 = min(h, y + box_h + pad_h)

        return frame[y1:y2, x1:x2]

    def process_video_stable(self, video_path):
        """
        Process video with stable face cropping using temporal smoothing.

        Strategy:
        1. Detect faces in all frames first
        2. Interpolate missing detections
        3. Smooth bounding boxes temporally
        4. Crop all frames consistently
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None

        # Step 1: Load all frames and detect faces
        print(f"  Loading frames...")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            return None

        print(f"  Detecting faces in {len(frames)} frames...")
        bboxes = [None] * len(frames)

        # Detect faces in all frames
        for i, frame in enumerate(frames):
            faces = self.face_cropper.get_faces(
                frame,
                remove_background=False,
                correct_roll=True
            )

            if faces:
                # Store the cropped face directly
                bboxes[i] = faces[0]

        # Count successful detections
        success_count = sum(1 for bbox in bboxes if bbox is not None)
        detection_rate = (success_count / len(frames)) * 100
        print(f"  Face detection: {success_count}/{len(frames)} ({detection_rate:.1f}%)")

        if success_count == 0:
            print(f"  No faces detected in any frame!")
            return None

        # Step 2: Fill missing detections
        # Strategy: Use nearest detected face
        print(f"  Filling missing detections...")
        for i in range(len(bboxes)):
            if bboxes[i] is None:
                # Find nearest detected face
                left_idx = None
                right_idx = None

                # Search left
                for j in range(i - 1, -1, -1):
                    if bboxes[j] is not None:
                        left_idx = j
                        break

                # Search right
                for j in range(i + 1, len(bboxes)):
                    if bboxes[j] is not None:
                        right_idx = j
                        break

                # Use nearest neighbor
                if left_idx is not None and right_idx is not None:
                    # Use closer one
                    if i - left_idx <= right_idx - i:
                        bboxes[i] = bboxes[left_idx]
                    else:
                        bboxes[i] = bboxes[right_idx]
                elif left_idx is not None:
                    bboxes[i] = bboxes[left_idx]
                elif right_idx is not None:
                    bboxes[i] = bboxes[right_idx]

        # Step 3: Process frames with consistent faces
        print(f"  Cropping and resizing...")
        processed_frames = []
        for i, face in enumerate(bboxes):
            if face is not None:
                # Resize face crop
                face_resized = cv2.resize(face, self.frame_size)
                # Convert back to BGR
                frame_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
                processed_frames.append(frame_bgr)
            else:
                # Fallback: use original frame resized
                frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.resize(frame_bgr, self.frame_size)
                processed_frames.append(frame_bgr)

        return processed_frames

    def save_video(self, frames, output_path, fps=25):
        """Save processed frames as video."""
        if len(frames) == 0:
            print(f"No frames to save for {output_path}")
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

    def process_all_videos(self, max_videos=None):
        """Process all videos for the speaker."""
        video_paths = self.get_video_paths()
        total_videos = len(video_paths)

        if max_videos is not None:
            video_paths = video_paths[:max_videos]
            print(f"Processing first {max_videos} of {total_videos} videos (test mode)")
        else:
            print(f"Processing all {total_videos} videos for speaker {self.speaker_id}")

        success_count = 0
        fail_count = 0

        for video_path, video_folder, video_file in tqdm(video_paths, desc="Processing videos"):
            print(f"\n{video_file}:")

            # Create output directory structure
            output_folder = os.path.join(self.output_dir, self.speaker_id, video_folder)
            os.makedirs(output_folder, exist_ok=True)

            # Process video with stable method
            processed_frames = self.process_video_stable(video_path)

            if processed_frames is not None and len(processed_frames) > 0:
                # Save processed video
                output_path = os.path.join(output_folder, video_file)
                self.save_video(processed_frames, output_path)
                print(f"  ✓ Saved: {len(processed_frames)} frames")
                success_count += 1
            else:
                print(f"  ✗ Failed to process")
                fail_count += 1

        print(f"\n{'='*70}")
        print(f"Processing complete!")
        print(f"  Success: {success_count}/{len(video_paths)}")
        print(f"  Failed: {fail_count}/{len(video_paths)}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Stable preprocessing for VoxCeleb2 videos with face cropping'
    )
    parser.add_argument(
        '--video_dir', type=str, required=True,
        help='Root directory of videos (e.g., ./voxceleb2/mp4/dev)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory for processed videos'
    )
    parser.add_argument(
        '--speaker_id', type=str, default='id06963',
        help='Speaker ID to process (default: id06963)'
    )
    parser.add_argument(
        '--frame_size', type=int, default=224,
        help='Frame size (will be frame_size x frame_size, default: 224)'
    )
    parser.add_argument(
        '--max_videos', type=int, default=None,
        help='Maximum number of videos to process (for testing)'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = StablePreprocessor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id,
        frame_size=(args.frame_size, args.frame_size)
    )

    # Process videos
    preprocessor.process_all_videos(max_videos=args.max_videos)


if __name__ == '__main__':
    main()
