#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for the trading strategy backtester frontend.
"""
import os
import sys
import subprocess

def check_python_version():
    """Check if the Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Warning: Python 3.7 or higher is recommended for this application.")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        print("Flask is installed.")
        return True
    except ImportError:
        print("Flask is not installed. Please run 'pip install -r requirements.txt'")
        return False

def create_directories():
    """Create required directories."""
    dirs = [
        '/home/pyzron02/frontend/temp',
        '/home/pyzron02/frontend/static',
        '/home/pyzron02/trading-strategy-backtester/output'
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")

def check_backtester():
    """Check if the backtester codebase is accessible."""
    backtester_dir = '/home/pyzron02/trading-strategy-backtester'
    if not os.path.exists(backtester_dir):
        print(f"Error: Trading Strategy Backtester not found at {backtester_dir}")
        return False
    
    # Check for critical files
    critical_files = [
        os.path.join(backtester_dir, 'src', 'workflows', 'unified_workflow.py'),
        os.path.join(backtester_dir, 'src', 'strategies', 'registry.py')
    ]
    
    missing = [f for f in critical_files if not os.path.exists(f)]
    if missing:
        print("Error: Some critical files are missing:")
        for file in missing:
            print(f"  - {file}")
        return False
    
    print("Trading Strategy Backtester found and accessible.")
    return True

def main():
    """Run all setup checks."""
    print("Setting up Trading Strategy Backtester Frontend...")
    
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    create_directories()
    backtester_ok = check_backtester()
    
    if python_ok and deps_ok and backtester_ok:
        print("\nSetup complete! You can now run the frontend with 'python app.py'")
        return 0
    else:
        print("\nSetup encountered issues. Please resolve them before running the application.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 