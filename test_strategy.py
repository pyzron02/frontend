#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the MultiPosition strategy works correctly.
"""
import os
import sys
import json
from datetime import datetime

# Add the trading-strategy-backtester to the path
project_root = '/home/pyzron02/trading-strategy-backtester'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Test imports
from src.strategies.registry import get_registered_strategies, get_strategy_class
from src.workflows.unified_workflow import run_complete_workflow

def main():
    """Run a simple test of the MultiPosition strategy."""
    print("Testing MultiPosition Strategy")
    print("=============================")
    
    # Get all registered strategies
    strategies = get_registered_strategies()
    print(f"Available strategies: {strategies}")
    
    # Check if MultiPosition is available
    try:
        strategy_class = get_strategy_class('MultiPosition')
        print(f"MultiPosition class found: {strategy_class.__name__}")
    except Exception as e:
        print(f"Error getting MultiPosition strategy: {e}")
        print("Available strategies:")
        for strategy in strategies:
            print(f"  - {strategy['name']} (v{strategy['version']})")
        return 1
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, 'output', f'MultiPosition_test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create parameter file
    param_file = os.path.join(output_dir, 'parameters.json')
    with open(param_file, 'w') as f:
        # Create parameters with each value in a list for optimization
        json.dump({
            'parameters': {
                'sma_period': [20],
                'position_size': [50],
                'max_positions': [5]
            }
        }, f, indent=4)
    
    print(f"Created parameter file: {param_file}")
    print(f"Output directory: {output_dir}")
    
    # Run the backtest
    try:
        print("Running workflow...")
        result = run_complete_workflow(
            strategy_name='MultiPosition',
            tickers=['AAPL', 'MSFT', 'GOOG'],
            start_date='2020-01-01',
            end_date='2021-12-31',
            param_file=param_file,
            num_workers=1,
            output_dir=output_dir,
            in_sample_ratio=0.7,
            num_simulations=0,  # Skip Monte Carlo
            verbose=True,
            seed=42
        )
        
        # Save the results
        results_file = os.path.join(output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(result, f, default=lambda o: str(o), indent=4)
        
        print(f"Test completed successfully! Results saved to {results_file}")
        return 0
    except Exception as e:
        import traceback
        print(f"Error running test: {e}")
        print(traceback.format_exc())
        error_file = os.path.join(output_dir, 'error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        print(f"Error details saved to {error_file}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 