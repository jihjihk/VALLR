"""
Verification script to check phoneme generation accuracy.
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import get_vocab


def load_phoneme_json(json_path):
    """Load phoneme JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def verify_phoneme_coverage(data, vocab):
    """Check if all phonemes are in vocabulary."""
    phoneme_sequence = data['phoneme_sequence']
    phoneme_ids = data['phoneme_ids']

    issues = []

    # Check if all phonemes are valid
    unknown_phonemes = [p for p in phoneme_sequence if p not in vocab]
    if unknown_phonemes:
        issues.append(f"Unknown phonemes found: {set(unknown_phonemes)}")

    # Check if phoneme IDs match
    expected_ids = [vocab.get(p, 0) for p in phoneme_sequence]
    if expected_ids != phoneme_ids:
        issues.append("Phoneme IDs don't match phoneme sequence")

    # Check aligned phonemes
    aligned = data['aligned_phonemes']
    if len(aligned) != len(phoneme_sequence):
        issues.append(f"Aligned phonemes count ({len(aligned)}) != phoneme sequence count ({len(phoneme_sequence)})")

    return issues


def analyze_phoneme_distribution(data):
    """Analyze phoneme distribution."""
    phoneme_sequence = data['phoneme_sequence']
    counter = Counter(phoneme_sequence)

    total = len(phoneme_sequence)
    print(f"\n📊 Phoneme Distribution (Total: {total} phonemes)")
    print("=" * 60)

    # Top 10 most common
    print("\nTop 10 most common phonemes:")
    for phoneme, count in counter.most_common(10):
        percentage = (count / total) * 100
        print(f"  {phoneme:4s}: {count:3d} ({percentage:5.2f}%)")

    # Unique phonemes
    unique = len(counter)
    print(f"\nUnique phonemes used: {unique}")

    return counter


def check_timestamp_alignment(data):
    """Check if timestamps are properly aligned."""
    aligned = data['aligned_phonemes']
    issues = []

    for i, phoneme_data in enumerate(aligned):
        start = phoneme_data['start']
        end = phoneme_data['end']

        # Check if end > start
        if end <= start:
            issues.append(f"Phoneme {i}: end time ({end}) <= start time ({start})")

        # Check if timestamps are sequential (with gaps allowed for word boundaries)
        if i > 0:
            prev_end = aligned[i-1]['end']
            # Allow small gaps between words, but flag large gaps
            gap = start - prev_end
            if gap > 1.0:  # More than 1 second gap
                issues.append(f"Phoneme {i}: large gap ({gap:.2f}s) from previous phoneme")

    return issues


def verify_transcript_to_phoneme_mapping(data):
    """Verify transcript was properly converted to phonemes."""
    transcript = data['transcript']
    phoneme_sequence = data['phoneme_sequence']
    aligned_phonemes = data['aligned_phonemes']

    print(f"\n📝 Transcript Analysis")
    print("=" * 60)
    print(f"Original transcript: {transcript[:100]}...")
    print(f"Word count: {len(transcript.split())}")
    print(f"Phoneme count: {len(phoneme_sequence)}")
    print(f"Average phonemes per word: {len(phoneme_sequence) / len(transcript.split()):.2f}")

    # Check if aligned phonemes have word information
    words_in_aligned = set(p['word'] for p in aligned_phonemes)
    print(f"\nUnique words in aligned phonemes: {len(words_in_aligned)}")

    # Sample some word-to-phoneme mappings
    print("\n📌 Sample word-to-phoneme mappings:")
    seen_words = set()
    for p in aligned_phonemes[:50]:  # Check first 50
        word = p['word']
        if word not in seen_words and len(seen_words) < 5:
            word_phonemes = [x['phoneme'] for x in aligned_phonemes if x['word'] == word]
            print(f"  '{word}' → {' '.join(word_phonemes)}")
            seen_words.add(word)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify phoneme JSON output")
    parser.add_argument('json_path', type=str, help="Path to phoneme JSON file")
    parser.add_argument('--detailed', action='store_true', help="Show detailed analysis")

    args = parser.parse_args()

    # Load vocab and data
    vocab = get_vocab()
    data = load_phoneme_json(args.json_path)

    print("\n" + "=" * 60)
    print("🔍 PHONEME VERIFICATION REPORT")
    print("=" * 60)
    print(f"\nFile: {args.json_path}")
    print(f"Video: {data['video_path']}")

    # 1. Check phoneme coverage
    print("\n✅ Checking Phoneme Coverage...")
    coverage_issues = verify_phoneme_coverage(data, vocab)
    if coverage_issues:
        print("❌ Issues found:")
        for issue in coverage_issues:
            print(f"  - {issue}")
    else:
        print("✅ All phonemes are valid and properly mapped!")

    # 2. Check timestamp alignment
    print("\n✅ Checking Timestamp Alignment...")
    timestamp_issues = check_timestamp_alignment(data)
    if timestamp_issues:
        print(f"⚠️  {len(timestamp_issues)} timestamp issues found")
        if args.detailed:
            for issue in timestamp_issues[:10]:  # Show first 10
                print(f"  - {issue}")
    else:
        print("✅ All timestamps are properly aligned!")

    # 3. Analyze phoneme distribution
    if args.detailed:
        analyze_phoneme_distribution(data)

    # 4. Verify transcript mapping
    verify_transcript_to_phoneme_mapping(data)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    total_issues = len(coverage_issues) + len(timestamp_issues)
    if total_issues == 0:
        print("✅ PASS - No issues found!")
        print("   The phoneme generation looks accurate.")
    else:
        print(f"⚠️  WARNINGS - {total_issues} issue(s) found")
        print("   Review the issues above.")

    print("\n💡 Tips for validation:")
    print("  1. Listen to the audio and compare with transcript")
    print("  2. Check if phoneme sequence sounds correct when spoken")
    print("  3. Verify timestamp alignment with audio playback")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()