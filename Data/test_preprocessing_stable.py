"""
Test the stable preprocessing that fixes glitchy face detection.
"""

import os
import sys
from Data.preprocess_opencv_stable import StablePreprocessor


def test_stable_preprocessing():
    """
    Test stable preprocessing on a few videos.
    """
    # Configuration
    video_dir = './voxceleb2/mp4/dev'
    speaker_id = 'id06963'

    print("=" * 70)
    print("VALLR STABLE Preprocessing Test")
    print("=" * 70)
    print("\nFixes:")
    print("  ✓ No glitchy frames (fills missing detections)")
    print("  ✓ Consistent face crops across all frames")
    print("  ✓ Uses SHORT_RANGE detector (better for VoxCeleb2)")
    print("  ✓ Lower confidence threshold (detects more faces)")
    print("=" * 70)

    # Initialize preprocessor
    print("\n1. Initializing stable preprocessor...")
    preprocessor = StablePreprocessor(
        video_dir=video_dir,
        output_dir='./test_output_stable',
        speaker_id=speaker_id,
        frame_size=(224, 224)
    )

    # Get video paths
    print("\n2. Finding videos...")
    video_paths = preprocessor.get_video_paths()
    print(f"   Found {len(video_paths)} videos")

    # Test on first 3 videos
    print(f"\n3. Testing on 3 videos with stable preprocessing...\n")
    print("=" * 70)

    preprocessor.process_all_videos(max_videos=3)

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    print("\nCheck the output:")
    print("  1. Open videos in: ./test_output_stable/id06963/")
    print("  2. Videos should have consistent face crops (no glitching)")
    print("  3. All frames should look smooth")
    print("\nIf it looks good, run full preprocessing:")
    print("\n  python Data/preprocess_opencv_stable.py \\")
    print("      --video_dir ./voxceleb2/mp4/dev \\")
    print("      --output_dir ./processed_voxceleb2 \\")
    print("      --speaker_id id06963")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_stable_preprocessing()
