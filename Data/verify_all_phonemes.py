"""
Batch verification script to check all phoneme JSON files in a directory.
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import get_vocab


def load_phoneme_json(json_path):
    """Load phoneme JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def verify_phoneme_file(json_path, vocab):
    """Verify a single phoneme JSON file."""
    data = load_phoneme_json(json_path)
    if data is None:
        return {'file': json_path, 'status': 'error', 'error': 'Failed to load JSON', 'issues': []}

    issues = []

    # 1. Check phoneme coverage
    phoneme_sequence = data.get('phoneme_sequence', [])
    phoneme_ids = data.get('phoneme_ids', [])

    unknown_phonemes = [p for p in phoneme_sequence if p not in vocab]
    if unknown_phonemes:
        issues.append(f"Unknown phonemes: {set(unknown_phonemes)}")

    # 2. Check phoneme IDs match
    expected_ids = [vocab.get(p, 0) for p in phoneme_sequence]
    if expected_ids != phoneme_ids:
        issues.append("Phoneme IDs mismatch")

    # 3. Check aligned phonemes
    aligned = data.get('aligned_phonemes', [])
    if len(aligned) != len(phoneme_sequence):
        issues.append(f"Aligned count mismatch: {len(aligned)} vs {len(phoneme_sequence)}")

    # 4. Check timestamps
    timestamp_issues = 0
    for i, phoneme_data in enumerate(aligned):
        start = phoneme_data.get('start', 0)
        end = phoneme_data.get('end', 0)
        if end <= start:
            timestamp_issues += 1

    if timestamp_issues > 0:
        issues.append(f"{timestamp_issues} timestamp issues")

    # 5. Check if data is empty
    if len(phoneme_sequence) == 0:
        issues.append("Empty phoneme sequence")

    status = 'pass' if len(issues) == 0 else 'warning'

    return {
        'file': json_path,
        'status': status,
        'phoneme_count': len(phoneme_sequence),
        'word_count': len(data.get('transcript', '').split()),
        'issues': issues
    }


def find_all_json_files(directory):
    """Recursively find all JSON files."""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_phonemes.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def generate_summary_report(results):
    """Generate a summary report of all verification results."""
    total = len(results)
    passed = sum(1 for r in results if r['status'] == 'pass')
    warnings = sum(1 for r in results if r['status'] == 'warning')
    errors = sum(1 for r in results if r['status'] == 'error')

    total_phonemes = sum(r.get('phoneme_count', 0) for r in results)
    total_words = sum(r.get('word_count', 0) for r in results)

    # Collect all issues
    issue_counter = Counter()
    for r in results:
        for issue in r.get('issues', []):
            # Extract issue type (first part before colon)
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counter[issue_type] += 1

    print("\n" + "=" * 80)
    print("📊 BATCH VERIFICATION SUMMARY REPORT")
    print("=" * 80)

    print(f"\n📁 Files Processed: {total}")
    print(f"   ✅ Passed:   {passed:4d} ({passed/total*100:5.1f}%)")
    print(f"   ⚠️  Warnings: {warnings:4d} ({warnings/total*100:5.1f}%)")
    print(f"   ❌ Errors:   {errors:4d} ({errors/total*100:5.1f}%)")

    print(f"\n📈 Statistics:")
    print(f"   Total phonemes: {total_phonemes:,}")
    print(f"   Total words:    {total_words:,}")
    if total_words > 0:
        print(f"   Avg phonemes/word: {total_phonemes/total_words:.2f}")

    if issue_counter:
        print(f"\n⚠️  Common Issues:")
        for issue_type, count in issue_counter.most_common(10):
            print(f"   - {issue_type}: {count} files")

    return passed, warnings, errors


def save_detailed_report(results, output_file):
    """Save detailed results to a JSON file."""
    report = {
        'total_files': len(results),
        'summary': {
            'passed': sum(1 for r in results if r['status'] == 'pass'),
            'warnings': sum(1 for r in results if r['status'] == 'warning'),
            'errors': sum(1 for r in results if r['status'] == 'error')
        },
        'files': results
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Detailed report saved to: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch verify phoneme JSON files")
    parser.add_argument('directory', type=str, help="Directory containing phoneme JSON files")
    parser.add_argument('--output', type=str, default=None, help="Output file for detailed report (JSON)")
    parser.add_argument('--show-failures', action='store_true', help="Show files that failed")
    parser.add_argument('--show-all', action='store_true', help="Show all file results")

    args = parser.parse_args()

    # Load vocab
    vocab = get_vocab()

    # Find all JSON files
    print(f"\n🔍 Searching for phoneme JSON files in: {args.directory}")
    json_files = find_all_json_files(args.directory)

    if len(json_files) == 0:
        print("❌ No phoneme JSON files found!")
        return

    print(f"✅ Found {len(json_files)} JSON files")

    # Verify all files
    print("\n🔄 Verifying files...")
    results = []
    for json_file in tqdm(json_files, desc="Processing"):
        result = verify_phoneme_file(json_file, vocab)
        results.append(result)

    # Generate summary report
    passed, warnings, errors = generate_summary_report(results)

    # Show failures if requested
    if args.show_failures or args.show_all:
        print("\n" + "=" * 80)
        print("📋 DETAILED RESULTS")
        print("=" * 80)

        for result in results:
            if args.show_all or result['status'] != 'pass':
                status_symbol = {
                    'pass': '✅',
                    'warning': '⚠️ ',
                    'error': '❌'
                }.get(result['status'], '?')

                print(f"\n{status_symbol} {os.path.basename(result['file'])}")
                if result['status'] != 'pass':
                    print(f"   Status: {result['status']}")
                if result.get('issues'):
                    print(f"   Issues:")
                    for issue in result['issues']:
                        print(f"     - {issue}")

    # Save detailed report if requested
    if args.output:
        save_detailed_report(results, args.output)

    # Final summary
    print("\n" + "=" * 80)
    if errors == 0 and warnings == 0:
        print("🎉 SUCCESS - All files passed verification!")
    elif errors == 0:
        print(f"⚠️  COMPLETED WITH WARNINGS - {warnings} file(s) need attention")
    else:
        print(f"❌ COMPLETED WITH ERRORS - {errors} file(s) failed, {warnings} warning(s)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()