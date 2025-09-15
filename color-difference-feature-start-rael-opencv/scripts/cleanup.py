#!/usr/bin/env python3
"""
Cleanup script for Color Difference Analysis System.
Removes all cache and temporary output files.
"""

import argparse
import shutil
import sys
from pathlib import Path


def get_cleanup_patterns() -> list[str]:
    """Get list of patterns/directories to clean up."""
    return [
        # Python cache directories
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        # Project-specific temp/output directories
        "temp_output",
        "alignment_results",
        "best_results",
        # Common temp files
        "*.tmp",
        "*.temp",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        # Build artifacts
        "build/",
        "dist/",
        "*.egg-info/",
        ".coverage",
        "htmlcov/",
        # IDE files
        ".vscode/settings.json",
        ".idea/",
        "*.swp",
        "*.swo",
    ]


def find_files_to_clean(root_path: Path, patterns: list[str]) -> list[Path]:
    """Find all files and directories matching cleanup patterns."""
    files_to_clean = []

    # Exclude paths that should not be cleaned
    exclude_paths = {".venv", ".git", "node_modules", ".tox", "venv", "env"}

    for pattern in patterns:
        if pattern.endswith("/"):
            # Directory pattern
            pattern = pattern.rstrip("/")
            for path in root_path.rglob(pattern):
                if path.is_dir() and not any(
                    exclude in str(path) for exclude in exclude_paths
                ):
                    files_to_clean.append(path)
            # File pattern
            for path in root_path.rglob(pattern):
                if not any(exclude in str(path) for exclude in exclude_paths):
                    files_to_clean.append(path)

    return files_to_clean


def clean_files(
    files_to_clean: list[Path], dry_run: bool = False, verbose: bool = False
) -> int:
    """Clean the specified files and directories."""
    total_size = 0
    count = 0

    for path in files_to_clean:
        try:
            if path.exists():
                # Calculate size before deletion
                if path.is_file():
                    size = path.stat().st_size
                elif path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                else:
                    size = 0

                total_size += size
                count += 1

                if verbose or dry_run:
                    action = "Would delete" if dry_run else "Deleting"
                    size_mb = size / (1024 * 1024)
                    print(f"{action}: {path} ({size_mb:.2f} MB)")

                if not dry_run:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path)

        except Exception as e:
            print(f"Error cleaning {path}: {e}", file=sys.stderr)

    return total_size, count


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean cache and temporary files from the Color Difference Analysis System"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to clean (default: current directory)",
    )

    args = parser.parse_args()

    # Ensure we're in a project directory
    if not (args.root / "pyproject.toml").exists():
        print(
            "Warning: pyproject.toml not found. Are you in the project root?",
            file=sys.stderr,
        )

    print("🧹 Color Difference Analysis System - Cleanup Tool")
    print("=" * 50)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be deleted")
        print()

    patterns = get_cleanup_patterns()
    files_to_clean = find_files_to_clean(args.root, patterns)

    if not files_to_clean:
        print("✅ No cache or temporary files found to clean!")
        return 0

    total_size, count = clean_files(files_to_clean, args.dry_run, args.verbose)

    print()
    if args.dry_run:
        print(
            f"📊 Would clean {count} items, freeing {total_size / (1024 * 1024):.2f} MB"
        )
    else:
        print(f"✅ Cleaned {count} items, freed {total_size / (1024 * 1024):.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
