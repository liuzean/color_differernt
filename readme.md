# 🎨 Color Difference Analysis System

A high-performance color difference analysis tool with advanced image processing capabilities, built with modern Python tools and powered by OpenCV.

## Version Requirements

- Python 3.10+
- Gradio 5.35.0+
- NumPy 2.2.0+
- OpenCV Contrib 4.8.0+

## Quick Setup

### Option 1: Automated Setup (Recommended)

Run the setup script to automatically install uv and set up the environment:

```bash
python setup.py
```

### Option 2: Manual Setup

1. **Install uv** (if not already installed):

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or using pip
   pip install uv
   ```

2. **Install project dependencies**:

   ```bash
   uv sync
   ```

3. **Install development dependencies** (optional):

   ```bash
   uv sync --group dev
   ```

## Running the Application

### Using uv (Recommended)

```bash
# Run the Gradio app with hot reloading
uv run gradio app.py

# Or run directly
uv run python app.py
```

### Using traditional virtual environment

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Run the application
python app.py
```

## Development Commands

### Package Management

```bash
# Install new dependencies
uv add package-name

# Install development dependencies
uv add --group dev package-name

# Remove dependencies
uv remove package-name

# Update all dependencies
uv lock --upgrade

# Sync environment with lock file
uv sync

# Export requirements for compatibility
uv export > requirements.txt
```

### Code Quality & Testing

```bash
# Format code with ruff
uv run ruff format .

# Check code formatting
uv run ruff format --check .

# Run linting
uv run ruff check .

# Fix linting issues automatically
uv run ruff check --fix .

# Type checking
uv run mypy . --ignore-missing-imports

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=core --cov=interface --cov=visualization --cov-report=html

# Run pre-commit hooks
uv run pre-commit run --all-files
```

### Development Scripts

Additional utilities are available in `scripts/dev.py`:

```bash
# Development environment setup
uv run python scripts/dev.py setup

# Comprehensive quality checks
uv run python scripts/dev.py check

# Create test data
uv run python scripts/dev.py test-data

# Performance benchmarks
uv run python scripts/dev.py benchmark

# Update pre-commit hooks
uv run python scripts/dev.py update-hooks
```

### Cleanup Utilities

Keep your workspace clean with our comprehensive cleanup tools:

#### Using Makefile (Recommended)

```bash
# Show available cleanup commands
make help

# Clean all cache and temporary files
make clean

# See what would be cleaned (dry run)
make clean-dry

# Clean only Python cache files
make clean-cache

# Clean only temporary output files
make clean-temp
```

#### Direct Script Usage

```bash
# Show what would be cleaned
python scripts/cleanup.py --dry-run --verbose

# Clean all cache and temp files
python scripts/cleanup.py

# Use the bash wrapper
./scripts/clean.sh --dry-run  # Preview
./scripts/clean.sh            # Clean
```

#### Pre-commit Integration

```bash
# Manual cleanup check (dry-run)
uv run pre-commit run cleanup-cache --hook-stage manual
```

#### What Gets Cleaned

The cleanup tools remove:

- **Python Cache**: `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- **Temporary Output**: `temp_output/`, `alignment_results/`, `best_results/`
- **Build Artifacts**: `build/`, `dist/`, `*.egg-info/`
- **IDE Files**: `.DS_Store`, `Thumbs.db`, `.vscode/settings.json`
- **Backup Files**: `*~`, `*.tmp`, `*.temp`

## Project Structure

```
color-difference/
├── app.py                 # Main application entry point
├── config.yaml           # Configuration file
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock               # Locked dependency versions (auto-generated)
├── setup.py              # Automated setup script
├── core/                 # Core functionality modules
│   ├── color/            # Color processing and ICC profiles
│   ├── image/            # Image processing and feature detection
│   └── utils/            # Utility functions
├── interface/            # Web interface (Gradio)
│   ├── components/       # UI components
│   ├── handlers/         # Event handlers
│   └── gui.py           # Main GUI interface
├── visualization/        # Visualization modules
│   ├── blockwise.py     # Block-based analysis
│   ├── charts.py        # Chart generation
│   ├── heatmap.py       # Heatmap visualization
│   └── covermap.py      # Coverage mapping
├── tests/               # Test files
└── scripts/            # Development utilities
```

## Features

### Core Functionality

- **Advanced Color Analysis**: Delta E calculations with multiple color spaces
- **Image Alignment**: Feature-based image registration using OpenCV
- **Heatmap Generation**: Visual representation of color differences
- **ICC Profile Support**: Professional color management
- **Batch Processing**: Process multiple images efficiently

### Web Interface

- **Modern UI**: Built with Gradio for responsive web interface
- **Real-time Processing**: Live updates during analysis
- **Interactive Visualizations**: Explore results with interactive charts
- **Export Options**: Save results in various formats

### Development Features

- **Fast Dependencies**: 10-100x faster than pip with uv
- **Reproducible Builds**: Lock files ensure consistent environments
- **Modern Tooling**: Pre-commit hooks, automated testing, CI/CD
- **Type Safety**: Full MyPy type checking

## Benefits of Using uv

- **Speed**: 10-100x faster than pip for dependency resolution
- **Reliability**: Lock files ensure reproducible environments
- **Modern**: Built-in support for Python project standards
- **Simplicity**: Single tool for all package management needs

## Troubleshooting

### Common Issues

1. **uv command not found**
   - Restart your terminal after installation
   - Or manually add to PATH: `export PATH="$HOME/.cargo/bin:$PATH"`

2. **Permission errors on Windows**
   - Run PowerShell as Administrator
   - Or install with `--user` flag

3. **Lock file conflicts**
   - Delete `uv.lock` and run `uv lock` to regenerate

4. **Import errors**
   - Ensure virtual environment is activated or use `uv run`
   - Check that all dependencies are installed: `uv sync`

### Getting Help

- [uv Documentation](https://docs.astral.sh/uv/)
- [Project Issues](https://github.com/gdut/color-difference/issues)

## Pre-commit Hooks & Code Quality

This project uses pre-commit hooks to ensure code quality and consistency:

### Setup Pre-commit

```bash
# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

### What's Included

The pre-commit configuration includes:

- **Code Formatting**: Ruff for formatting and linting
- **Type Checking**: MyPy with strict settings
- **Documentation**: Markdown and YAML linting
- **Git**: Commit message formatting (Conventional Commits)

### Manual Execution

```bash
# Run on all files
uv run pre-commit run --all-files

# Run on staged files only
uv run pre-commit run

# Update hook versions
uv run python scripts/dev.py update-hooks
```

## CI/CD & GitHub Actions

The project includes comprehensive GitHub Actions workflows:

### Continuous Integration

- **Quality Checks**: Ruff formatting and linting, MyPy type checking
- **Testing**: Cross-platform testing with pytest
- **Coverage**: Code coverage reporting
- **Build**: Package building and validation

### Workflow Features

- **Fast Execution**: Optimized with uv for rapid CI/CD
- **Parallel Jobs**: Multiple jobs running simultaneously
- **Modern Python**: Latest Python versions and best practices
- **Caching**: Efficient dependency caching

## Developer Experience

### 🚀 **Quick Start**

```bash
git clone <repo>
cd color-difference
python setup.py     # Complete setup in one command
uv run gradio app.py # Start developing immediately
```

### 🔧 **Development Workflow**

```bash
# Daily development
uv add new-package           # Add dependencies
uv run ruff format .         # Format code
uv run pytest              # Run tests
uv run pre-commit run       # Quality checks
```

### 📊 **Analysis Features**

- Professional color difference analysis
- Advanced image alignment algorithms
- Real-time visualization
- Export capabilities

## Legacy Support

If you prefer to use pip, you can still install dependencies traditionally:

```bash
pip install -e .
pip install -e ".[dev]"
```

However, we strongly recommend using uv for better performance and reproducibility.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `uv sync --group dev`
4. Make your changes
5. Run quality checks: `uv run pre-commit run --all-files`
6. Run tests: `uv run pytest`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
