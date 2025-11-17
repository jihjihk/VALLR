import whisper
import pronouncing
import re
import os
import json
import sys
from tqdm import tqdm
import argparse

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import get_vocab

class PhonemeGenerator:
    def __init__(self, whisper_model="base", output_format="json"):
        """
        Initialize the phoneme generator.

        Args:
            whisper_model (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            output_format (str): Format to save phoneme labels ('json' or 'txt')
        """
        print(f"Loading Whisper model: {whisper_model}")
        self.model = whisper.load_model(whisper_model)
        self.output_format = output_format
        self.phoneme_vocab = get_vocab()
        print(f"Loaded phoneme vocabulary with {len(self.phoneme_vocab)} phonemes")

    def transcribe_audio(self, video_path):
        """
        Transcribe audio from video using Whisper.

        Args:
            video_path (str): Path to video file

        Returns:
            dict: Transcription result with text and segments
        """
        try:
            result = self.model.transcribe(video_path, word_timestamps=True, language='en')
            return result
        except Exception as e:
            print(f"Error transcribing {video_path}: {e}")
            return None

    def word_to_phonemes(self, word):
        """
        Convert a word to phonemes using pronouncing library.
        Removes stress markers (numbers) from phonemes.

        Args:
            word (str): Input word

        Returns:
            list: List of phonemes (without stress markers)
        """
        word = word.lower().strip()
        # Remove punctuation
        word = re.sub(r'[^\w\s]', '', word)

        if not word:
            return []

        phonemes_list = pronouncing.phones_for_word(word)
        if phonemes_list:
            # Get first pronunciation and remove stress markers (digits)
            phonemes = [re.sub(r'\d+', '', p) for p in phonemes_list[0].split()]
            # Filter to only include phonemes in our vocabulary
            valid_phonemes = [p for p in phonemes if p in self.phoneme_vocab]
            return valid_phonemes
        return []

    def text_to_phonemes(self, text):
        """
        Convert a full text transcript to a sequence of phonemes.

        Args:
            text (str): Input text

        Returns:
            list: List of phonemes
        """
        words = text.lower().split()
        all_phonemes = []

        for word in words:
            phonemes = self.word_to_phonemes(word)
            all_phonemes.extend(phonemes)

        return all_phonemes

    def align_phonemes_to_timestamps(self, segments):
        """
        Align phonemes with word timestamps from Whisper output.

        Args:
            segments (list): Whisper segments with word timestamps

        Returns:
            list: List of phoneme alignments with timestamps
        """
        aligned_phonemes = []

        for segment in segments:
            if 'words' not in segment:
                continue

            for word_data in segment['words']:
                word = word_data['word'].strip()
                start_time = word_data.get('start', 0)
                end_time = word_data.get('end', 0)
                phonemes = self.word_to_phonemes(word)

                if phonemes and end_time > start_time:
                    word_duration = end_time - start_time
                    num_phonemes = len(phonemes)
                    phoneme_duration = word_duration / num_phonemes

                    # Assign timestamps to each phoneme
                    for i, phoneme in enumerate(phonemes):
                        phoneme_start = start_time + i * phoneme_duration
                        phoneme_end = phoneme_start + phoneme_duration
                        aligned_phonemes.append({
                            'phoneme': phoneme,
                            'phoneme_id': self.phoneme_vocab.get(phoneme, 0),
                            'start': phoneme_start,
                            'end': phoneme_end,
                            'word': word
                        })

        return aligned_phonemes

    def process_video(self, video_path, output_dir=None):
        """
        Process a single video: transcribe and generate phonemes.

        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save output (if None, saves alongside video)

        Returns:
            dict: Phoneme data
        """
        print(f"Processing: {video_path}")

        # Transcribe
        result = self.transcribe_audio(video_path)
        if result is None:
            return None

        # Get full text and convert to phonemes
        full_text = result['text']
        phoneme_sequence = self.text_to_phonemes(full_text)
        phoneme_ids = [self.phoneme_vocab.get(p, 0) for p in phoneme_sequence]

        # Get aligned phonemes with timestamps
        aligned_phonemes = self.align_phonemes_to_timestamps(result['segments'])

        # Prepare output data
        output_data = {
            'video_path': video_path,
            'transcript': full_text,
            'phoneme_sequence': phoneme_sequence,
            'phoneme_ids': phoneme_ids,
            'aligned_phonemes': aligned_phonemes
        }

        # Save output
        if output_dir is None:
            output_dir = os.path.dirname(video_path)

        os.makedirs(output_dir, exist_ok=True)

        video_basename = os.path.splitext(os.path.basename(video_path))[0]

        if self.output_format == 'json':
            output_path = os.path.join(output_dir, f"{video_basename}_phonemes.json")
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:  # txt format
            output_path = os.path.join(output_dir, f"{video_basename}_phonemes.txt")
            with open(output_path, 'w') as f:
                f.write(f"Transcript: {full_text}\n")
                f.write(f"Phonemes: {' '.join(phoneme_sequence)}\n")
                f.write(f"Phoneme IDs: {' '.join(map(str, phoneme_ids))}\n")

        print(f"Saved phoneme data to: {output_path}")
        return output_data

    def process_directory(self, video_dir, output_dir=None, recursive=True, video_extensions=('.mp4', '.avi', '.mov', '.mkv')):
        """
        Process all videos in a directory.

        Args:
            video_dir (str): Directory containing videos
            output_dir (str): Directory to save outputs (if None, saves alongside videos)
            recursive (bool): Whether to search recursively
            video_extensions (tuple): Video file extensions to process

        Returns:
            list: List of processed video data
        """
        video_files = []

        if recursive:
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if file.endswith(video_extensions):
                        video_files.append(os.path.join(root, file))
        else:
            video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                          if f.endswith(video_extensions)]

        print(f"Found {len(video_files)} videos to process")

        results = []
        for video_path in tqdm(video_files, desc="Processing videos"):
            # Maintain directory structure in output
            if output_dir:
                rel_path = os.path.relpath(os.path.dirname(video_path), video_dir)
                video_output_dir = os.path.join(output_dir, rel_path)
            else:
                video_output_dir = None

            result = self.process_video(video_path, video_output_dir)
            if result:
                results.append(result)

        print(f"Successfully processed {len(results)}/{len(video_files)} videos")
        return results


def main():
    parser = argparse.ArgumentParser(description="Generate phoneme labels from videos using Whisper")
    parser.add_argument('--video_path', type=str, help="Path to a single video file")
    parser.add_argument('--video_dir', type=str, help="Path to directory containing videos")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for phoneme labels")
    parser.add_argument('--whisper_model', type=str, default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help="Whisper model size")
    parser.add_argument('--output_format', type=str, default='json',
                       choices=['json', 'txt'],
                       help="Output format for phoneme labels")
    parser.add_argument('--recursive', action='store_true', default=True,
                       help="Process directories recursively")

    args = parser.parse_args()

    # Create phoneme generator
    generator = PhonemeGenerator(
        whisper_model=args.whisper_model,
        output_format=args.output_format
    )

    # Process video(s)
    if args.video_path:
        generator.process_video(args.video_path, args.output_dir)
    elif args.video_dir:
        generator.process_directory(args.video_dir, args.output_dir, args.recursive)
    else:
        print("Error: Please provide either --video_path or --video_dir")
        parser.print_help()


if __name__ == '__main__':
    main()
