#!/usr/bin/env python3
"""
Setup script for Color Difference Analysis System
This script helps set up the development environment using uv.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False


def check_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """Install uv package manager."""
    print("📦 Installing uv package manager...")
    if sys.platform.startswith("win"):
        # Windows installation
        command = 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    else:
        # Unix-like systems (Linux, macOS)
        command = "curl -LsSf https://astral.sh/uv/install.sh | sh"

    return run_command(command, "Installing uv")


def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up Color Difference Analysis System environment")

    # Check if uv is installed
    if not check_uv_installed():
        print("⚠️  uv is not installed. Installing now...")
        if not install_uv():
            print(
                "❌ Failed to install uv. Please install manually from https://docs.astral.sh/uv/"
            )
            return False

        # Add uv to PATH for current session
        if not sys.platform.startswith("win"):
            os.environ[
                "PATH"
            ] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"

    print("✅ uv is available")

    # Initialize uv project (if not already initialized)
    if not Path("uv.lock").exists():
        run_command("uv lock", "Creating lock file for reproducible builds")

    # Install dependencies
    commands = [
        ("uv sync", "Installing project dependencies"),
        ("uv sync --group dev", "Installing development dependencies"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Failed to execute: {command}")
            return False

    print("\n🎉 Environment setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Activate the virtual environment:")
    print("      source .venv/bin/activate  # Linux/macOS")
    print("      .venv\\Scripts\\activate     # Windows")
    print("   2. Run the application:")
    print("      uv run gradio app.py")
    print("   3. Or run directly:")
    print("      uv run python app.py")
    print("\n🛠️  Development commands:")
    print("   uv run pytest                 # Run tests")
    print("   uv run black .                # Format code")
    print("   uv run mypy .                 # Type checking")
    print("   uv add <package>              # Add new dependency")
    print("   uv remove <package>           # Remove dependency")
    print("   uv lock --upgrade             # Update dependencies")

    return True


def main():
    """Main setup function."""
    print("🎨 Color Difference Analysis System Setup")
    print("=" * 50)

    # Ensure we're in the project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
