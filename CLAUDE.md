# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands
- Run Flask app: `python app.py` or `flask run`
- Run tests: `python -m pytest`
- Run specific test: `python -m pytest test_strategy.py`
- Run backtest: `python run_backtest.py --config config.json`

## Code Style Guidelines
- Imports: Group standard library imports first, followed by external packages, then local imports
- Formatting: 4-space indentation, 100-character line limit
- Types: Use built-in Python types (str, int, float) in docstrings
- Naming: snake_case for variables and functions, CamelCase for classes
- Error handling: Use try-except blocks with specific exception types, logging the error details
- Documentation: Include docstrings for all functions/methods using triple quotes
- JSON handling: Use the SafeJSONEncoder for complex objects with pandas/numpy data
- Paths: Use os.path.join() for file paths, prefer absolute paths
- Environment variables: Access with os.getenv() with default values