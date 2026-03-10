#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple update script for Umaplay.

Runs:
1. git pull (updates code from repository)
2. Updates all game data (skills, characters, support cards) with images
3. Builds the catalog and web UI

Usage:
    python update.py
    python update.py --no-pull    # Skip git pull
    python update.py --no-images  # Skip image downloads
    python update.py --debug      # Verbose output
"""

import argparse
import subprocess
import sys
import os

# Ensure we're in the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=check)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"[ERROR] Command not found: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Update Umaplay: pull latest code, update game data, and rebuild",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip git pull (only update game data and rebuild)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip downloading card images",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  UMAPLAY UPDATE SCRIPT")
    print("="*60)

    success = True

    # Step 1: Git pull
    if not args.no_pull:
        if not run_command(["git", "pull"], "Pulling latest code from repository"):
            print("[WARN] Git pull failed, continuing with game data update...")

    # Step 2: Update game data (skills, characters, support cards)
    update_cmd = [
        sys.executable,
        "scripts/update_game_data.py",
        "--all",
    ]
    if not args.no_images:
        update_cmd.append("--images")
    if args.debug:
        update_cmd.append("--debug")

    if not run_command(update_cmd, "Updating game data (skills, characters, support cards)"):
        print("[ERROR] Game data update failed!")
        success = False

    # Summary
    print("\n" + "="*60)
    if success:
        print("  UPDATE COMPLETE!")
        print("="*60)
        print("\nPlease restart the bot for changes to take effect.")
        print("Close all terminals/IDEs and do a fresh start with:")
        print("  python main.py")
    else:
        print("  UPDATE COMPLETED WITH ERRORS")
        print("="*60)
        print("\nSome steps failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
