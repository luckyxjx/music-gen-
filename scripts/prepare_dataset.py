#!/usr/bin/env python3
"""
CLI tool for preparing and validating emotion-labeled MIDI datasets
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dataset_utils import (
    create_metadata_template,
    validate_dataset_metadata,
    print_validation_report,
    merge_datasets,
    scan_midi_directory
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and validate emotion-labeled MIDI datasets"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create template command
    create_parser = subparsers.add_parser(
        "create-template",
        help="Create metadata template for a directory of MIDI files"
    )
    create_parser.add_argument(
        "midi_dir",
        help="Directory containing MIDI files"
    )
    create_parser.add_argument(
        "-o", "--output",
        default="metadata.json",
        help="Output metadata file path (default: metadata.json)"
    )
    create_parser.add_argument(
        "-e", "--emotion",
        default="calm",
        help="Default emotion label (default: calm)"
    )
    create_parser.add_argument(
        "-f", "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate dataset metadata"
    )
    validate_parser.add_argument(
        "dataset_path",
        help="Path to dataset directory"
    )
    validate_parser.add_argument(
        "-m", "--metadata",
        default="metadata.json",
        help="Metadata file name (default: metadata.json)"
    )
    
    # Scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan directory for MIDI files"
    )
    scan_parser.add_argument(
        "directory",
        help="Directory to scan"
    )
    scan_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't scan subdirectories"
    )
    
    # Merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple datasets"
    )
    merge_parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset directories to merge"
    )
    merge_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output metadata file path"
    )
    merge_parser.add_argument(
        "-f", "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    if args.command == "create-template":
        print(f"Creating metadata template for {args.midi_dir}...")
        create_metadata_template(
            args.midi_dir,
            args.output,
            args.emotion,
            args.format
        )
        print(f"\nTemplate created at {args.output}")
        print("Please edit the file to add correct emotion labels.")
    
    elif args.command == "validate":
        print(f"Validating dataset at {args.dataset_path}...")
        results = validate_dataset_metadata(args.dataset_path, args.metadata)
        print_validation_report(results)
        
        if not results["valid"]:
            sys.exit(1)
    
    elif args.command == "scan":
        print(f"Scanning {args.directory} for MIDI files...")
        midi_files = scan_midi_directory(
            args.directory,
            recursive=not args.no_recursive
        )
        print(f"\nFound {len(midi_files)} MIDI files:")
        for f in midi_files[:10]:
            print(f"  {f}")
        if len(midi_files) > 10:
            print(f"  ... and {len(midi_files) - 10} more")
    
    elif args.command == "merge":
        print(f"Merging {len(args.datasets)} datasets...")
        merge_datasets(args.datasets, args.output, args.format)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
