"""
Utility functions for dataset preparation and validation
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
import pretty_midi


def validate_midi_file(midi_path: str) -> bool:
    """
    Validate that a MIDI file can be loaded and has content
    
    Args:
        midi_path: Path to MIDI file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        # Check if there are any notes
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        return total_notes > 0
    except Exception as e:
        print(f"Invalid MIDI file {midi_path}: {e}")
        return False


def scan_midi_directory(directory: str, recursive: bool = True) -> List[str]:
    """
    Scan directory for MIDI files
    
    Args:
        directory: Directory to scan
        recursive: Whether to scan subdirectories
    
    Returns:
        List of MIDI file paths
    """
    path = Path(directory)
    if recursive:
        midi_files = list(path.rglob("*.mid")) + list(path.rglob("*.midi"))
    else:
        midi_files = list(path.glob("*.mid")) + list(path.glob("*.midi"))
    
    return [str(f) for f in midi_files]


def create_metadata_template(
    midi_directory: str,
    output_path: str,
    default_emotion: str = "calm",
    format: str = "json"
):
    """
    Create a metadata template file for a directory of MIDI files
    
    Args:
        midi_directory: Directory containing MIDI files
        output_path: Where to save the metadata file
        default_emotion: Default emotion label
        format: Output format (json or csv)
    """
    midi_files = scan_midi_directory(midi_directory)
    
    if format == "json":
        metadata = []
        for midi_file in midi_files:
            rel_path = Path(midi_file).relative_to(midi_directory)
            metadata.append({
                "path": str(rel_path),
                "emotion": default_emotion
            })
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    elif format == "csv":
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "emotion"])
            for midi_file in midi_files:
                rel_path = Path(midi_file).relative_to(midi_directory)
                writer.writerow([str(rel_path), default_emotion])
    
    print(f"Created metadata template at {output_path} with {len(midi_files)} files")


def validate_dataset_metadata(
    dataset_path: str,
    metadata_file: str = "metadata.json"
) -> Dict:
    """
    Validate dataset metadata and check for issues
    
    Args:
        dataset_path: Path to dataset directory
        metadata_file: Name of metadata file
    
    Returns:
        Dictionary with validation results
    """
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path / metadata_file
    
    results = {
        "valid": True,
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": [],
        "missing_files": [],
        "emotion_distribution": {},
        "errors": []
    }
    
    if not metadata_path.exists():
        results["valid"] = False
        results["errors"].append(f"Metadata file not found: {metadata_path}")
        return results
    
    # Load metadata
    try:
        if metadata_file.endswith(".json"):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        elif metadata_file.endswith(".csv"):
            metadata = []
            with open(metadata_path, 'r') as f:
                reader = csv.DictReader(f)
                metadata = list(reader)
        else:
            results["valid"] = False
            results["errors"].append(f"Unsupported metadata format: {metadata_file}")
            return results
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Error loading metadata: {e}")
        return results
    
    # Validate each entry
    emotions = []
    for item in metadata:
        results["total_files"] += 1
        
        midi_path = dataset_path / item.get("path", "")
        emotion = item.get("emotion", "")
        
        # Check if file exists
        if not midi_path.exists():
            results["missing_files"].append(str(item.get("path", "")))
            continue
        
        # Validate MIDI file
        if validate_midi_file(str(midi_path)):
            results["valid_files"] += 1
            emotions.append(emotion)
        else:
            results["invalid_files"].append(str(midi_path))
    
    # Emotion distribution
    results["emotion_distribution"] = dict(Counter(emotions))
    
    # Check for issues
    if results["missing_files"]:
        results["errors"].append(f"{len(results['missing_files'])} files not found")
    if results["invalid_files"]:
        results["errors"].append(f"{len(results['invalid_files'])} invalid MIDI files")
    
    if results["errors"]:
        results["valid"] = False
    
    return results


def print_validation_report(results: Dict):
    """Print a formatted validation report"""
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    print(f"\nStatus: {'✓ VALID' if results['valid'] else '✗ INVALID'}")
    print(f"Total files: {results['total_files']}")
    print(f"Valid files: {results['valid_files']}")
    
    if results['missing_files']:
        print(f"\nMissing files ({len(results['missing_files'])}):")
        for f in results['missing_files'][:5]:
            print(f"  - {f}")
        if len(results['missing_files']) > 5:
            print(f"  ... and {len(results['missing_files']) - 5} more")
    
    if results['invalid_files']:
        print(f"\nInvalid MIDI files ({len(results['invalid_files'])}):")
        for f in results['invalid_files'][:5]:
            print(f"  - {f}")
        if len(results['invalid_files']) > 5:
            print(f"  ... and {len(results['invalid_files']) - 5} more")
    
    if results['emotion_distribution']:
        print("\nEmotion distribution:")
        for emotion, count in sorted(results['emotion_distribution'].items()):
            print(f"  {emotion}: {count}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*60)


def merge_datasets(
    dataset_paths: List[str],
    output_path: str,
    output_format: str = "json"
):
    """
    Merge multiple datasets into a single metadata file
    
    Args:
        dataset_paths: List of dataset directories
        output_path: Where to save merged metadata
        output_format: Output format (json or csv)
    """
    all_samples = []
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        metadata_path = dataset_path / "metadata.json"
        
        if not metadata_path.exists():
            print(f"Warning: No metadata found for {dataset_path}")
            continue
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update paths to be absolute or relative to output
        for item in metadata:
            item["path"] = str(dataset_path / item["path"])
            item["source"] = dataset_path.name
        
        all_samples.extend(metadata)
    
    # Save merged metadata
    if output_format == "json":
        with open(output_path, 'w') as f:
            json.dump(all_samples, f, indent=2)
    elif output_format == "csv":
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["path", "emotion", "source"])
            writer.writeheader()
            writer.writerows(all_samples)
    
    print(f"Merged {len(all_samples)} samples from {len(dataset_paths)} datasets")
    print(f"Saved to {output_path}")
