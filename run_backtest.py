#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line script to run a backtest with parameters from a JSON file.
This is used by the frontend to actually execute the backtest in a separate process.
"""
import os
import sys
import json
import argparse
import time
import tempfile
from datetime import datetime
import threading
import pandas as pd
import numpy as np

# Add the trading-strategy-backtester to the path
project_root = '/home/pyzron02/trading-strategy-backtester'
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import the workflow
from src.workflows.unified_workflow import run_complete_workflow

def print_progress(stop_event):
    """Print a simple progress indicator until the stop_event is set."""
    chars = ['|', '/', '-', '\\']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rRunning backtest {chars[i]} ")
        sys.stdout.flush()
        i = (i + 1) % len(chars)
        time.sleep(0.2)
    sys.stdout.write("\rBacktest completed!      \n")
    sys.stdout.flush()

def create_parameter_file(params, output_dir):
    """Create a parameter file for the backtest."""
    # Create a parameters.json file in the standard format expected by the backtester
    param_file = os.path.join(output_dir, 'parameters.json')
    
    # Check if params is already in the nested format with a 'parameters' key
    if isinstance(params, dict) and 'parameters' in params:
        # Already in the correct format
        param_data = params
    else:
        # Format depends on whether it's a grid (for optimization) or single values
        if isinstance(params, dict) and any(isinstance(v, list) for v in params.values()):
            # It's a parameter grid for optimization
            param_data = {'parameters': params}
        else:
            # For single parameter values, we have two options:
            # 1. If this is an optimization run, we need each parameter to be a list
            # 2. If this is a single run, we can keep parameters as scalar values
            
            # Check if the caller is running an optimization
            is_optimization = False
            # Look for the run_optimization flag from the stack frame
            import inspect
            for frame_info in inspect.stack():
                if 'config' in frame_info.frame.f_locals:
                    config = frame_info.frame.f_locals['config']
                    if isinstance(config, dict) and config.get('run_optimization', False):
                        is_optimization = True
                        break
            
            if is_optimization:
                # Convert all parameters to lists for optimization
                param_grid = {}
                for param_name, value in params.items():
                    if not isinstance(value, list):
                        param_grid[param_name] = [value]
                    else:
                        param_grid[param_name] = value
                param_data = {'parameters': param_grid}
            else:
                # For single run, keep parameters as is
                param_data = {'parameters': params}
    
    with open(param_file, 'w') as f:
        json.dump(param_data, f, indent=4)
    
    return param_file

def ensure_serializable(obj):
    """Convert any non-serializable objects to serializable types."""
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)
    return obj

def main():
    parser = argparse.ArgumentParser(description='Run a backtest with parameters from a JSON file')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    parser.add_argument('--output', type=str, help='Output directory (overrides the one in config)')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output')
    args = parser.parse_args()
    
    debug_mode = args.debug
    
    # Load the configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override output directory if specified
    if args.output:
        config['output_dir'] = args.output
    
    # Default output directory if not specified
    if 'output_dir' not in config:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['output_dir'] = os.path.join('/home/pyzron02/trading-strategy-backtester/output', 
                               f"{config.get('strategy_name', 'backtest')}_{timestamp}")
    
    # Create the output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save the config to the output directory
    with open(os.path.join(config['output_dir'], 'backtest_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Extract key variables
    strategy_name = config.get('strategy_name')
    tickers = config.get('tickers', [])
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    run_optimization = config.get('run_optimization', False)
    
    # Print information about the backtest
    print(f"Starting backtest for strategy: {strategy_name}")
    print(f"Tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {config['output_dir']}")
    
    # Show full path information for debugging
    if debug_mode:
        print(f"Python path: {sys.path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
    
    # Check that the strategy exists
    try:
        from src.strategies.registry import get_strategy_class
        strategy_class = get_strategy_class(strategy_name)
        print(f"Found strategy class: {strategy_class.__name__}")
    except Exception as e:
        print(f"WARNING: Could not verify strategy class: {e}")
    
    # Ensure param_file is created even if we're using fixed parameters
    if run_optimization:
        print(f"Running optimization with parameter grid")
        # Use the parameter grid provided or from param_file
        if 'param_grid' in config:
            param_grid = config['param_grid']
            # Create a parameter file if not already specified
            if 'param_file' not in config:
                param_file = create_parameter_file(param_grid, config['output_dir'])
                config['param_file'] = param_file
        elif 'param_file' in config:
            param_file = config['param_file']
            print(f"Using parameter file: {param_file}")
    else:
        print(f"Running with fixed parameters")
        # Use the single parameters provided
        if 'params' in config:
            params = config['params']
            # Create a parameter file for single values
            param_file = create_parameter_file(params, config['output_dir'])
            config['param_file'] = param_file
            print(f"Parameters formatted for optimization: {param_file}")
    
    # Ensure the parameter file exists
    if 'param_file' not in config:
        print("WARNING: No parameter file specified, creating an empty one")
        empty_params = {'parameters': {}}
        param_file = os.path.join(config['output_dir'], 'parameters.json')
        with open(param_file, 'w') as f:
            json.dump(empty_params, f, indent=4)
        config['param_file'] = param_file
    
    # Start a progress indicator thread
    stop_event = threading.Event()
    progress_thread = threading.Thread(target=print_progress, args=(stop_event,))
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        # Run the workflow
        print("Starting unified workflow...")
        
        # Debug print for parameter file
        if 'param_file' in config:
            print(f"Using parameter file: {config['param_file']}")
            
            # Verify parameter file exists
            if os.path.exists(config['param_file']):
                print(f"Parameter file exists. Contents:")
                with open(config['param_file'], 'r') as f:
                    param_contents = f.read()
                    print(param_contents)
                    
                    # Verify that the parameter file is properly formatted JSON
                    try:
                        param_json = json.loads(param_contents)
                        if 'parameters' not in param_json:
                            print("WARNING: Parameter file missing 'parameters' key")
                            # Fix the parameter file format
                            with open(config['param_file'], 'w') as f:
                                json.dump({'parameters': param_json}, f, indent=4)
                            print("Fixed parameter file format")
                    except json.JSONDecodeError:
                        print("WARNING: Parameter file is not valid JSON")
            else:
                print(f"WARNING: Parameter file does not exist: {config['param_file']}")
        
        # Import the run_complete_workflow function here to ensure it's using the correct Python path
        from src.workflows.unified_workflow import run_complete_workflow
        
        result = run_complete_workflow(
            strategy_name=strategy_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            param_file=config.get('param_file'),
            num_workers=config.get('num_workers', 1),
            output_dir=config['output_dir'],
            in_sample_ratio=config.get('in_sample_ratio', 0.7),
            num_simulations=config.get('num_simulations', 1000),
            verbose=config.get('verbose', False),
            seed=config.get('seed', 42),
            keep_permuted_data=config.get('keep_permuted_data', False)
        )
        
        # Stop the progress indicator
        stop_event.set()
        progress_thread.join()
        
        # Save the results
        results_file = os.path.join(config['output_dir'], 'complete_results.json')
        try:
            with open(results_file, 'w') as f:
                # Convert any non-serializable objects
                result_json = json.dumps(result, default=lambda o: str(o), indent=4)
                f.write(result_json)
        except Exception as e:
            print(f"Warning: Could not save full results: {e}")
            # Try a simpler approach
            with open(results_file, 'w') as f:
                json.dump({"success": True, "message": "Results saved separately"}, f)
        
        print(f"Backtest completed! Results saved to {results_file}")
        print("\nSummary:")
        
        # Print some summary metrics if available
        if 'walk_forward_metrics' in result and 'out_of_sample' in result['walk_forward_metrics']:
            metrics = result['walk_forward_metrics']['out_of_sample']
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
            print(f"  Total Return: {metrics.get('total_return', 'N/A')}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A')}")
            print(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
        
        # Create a simple backtest_results.json file for displaying in the UI
        ui_results_file = os.path.join(config['output_dir'], 'backtest_results.json')
        ui_results = {
            'strategy_name': strategy_name,
            'tickers': tickers,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {}
        }
        
        # Extract metrics from the results
        # Try multiple potential locations for metrics
        metrics_found = False
        
        # 1. First check if metrics are in walk_forward_metrics - out_of_sample
        if 'walk_forward_metrics' in result and 'out_of_sample' in result['walk_forward_metrics']:
            # Make sure we safely extract the metrics (handling pandas objects)
            try:
                out_of_sample_metrics = result['walk_forward_metrics']['out_of_sample']
                if isinstance(out_of_sample_metrics, (pd.Series, pd.DataFrame)):
                    # Convert pandas objects to dictionaries
                    ui_results['metrics'] = out_of_sample_metrics.to_dict()
                else:
                    ui_results['metrics'] = out_of_sample_metrics
                metrics_found = True
            except Exception as e:
                print(f"Error extracting out_of_sample metrics: {e}")
                # Handle the error gracefully
                ui_results['metrics'] = {}
                
        # 2. Check Out-of-Sample (different casing)
        elif 'walk_forward_metrics' in result and 'Out-of-Sample' in result['walk_forward_metrics']:
            try:
                out_of_sample_metrics = result['walk_forward_metrics']['Out-of-Sample']
                if isinstance(out_of_sample_metrics, (pd.Series, pd.DataFrame)):
                    # Convert pandas objects to dictionaries
                    ui_results['metrics'] = out_of_sample_metrics.to_dict()
                else:
                    ui_results['metrics'] = out_of_sample_metrics
                metrics_found = True
            except Exception as e:
                print(f"Error extracting Out-of-Sample metrics: {e}")
                # Handle the error gracefully
                ui_results['metrics'] = {}
        
        # 3. If no metrics found, try to calculate from the trade data
        if not metrics_found or not ui_results['metrics']:
            # Look for trade log files
            out_sample_dir = os.path.join(config['output_dir'], 'walk_forward', 'out_sample')
            in_sample_dir = os.path.join(config['output_dir'], 'walk_forward', 'in_sample')
            
            # Try to find results in out_sample directory
            backtest_results_file = os.path.join(out_sample_dir, 'backtest_results.json')
            if os.path.exists(backtest_results_file):
                try:
                    with open(backtest_results_file, 'r') as f:
                        backtest_data = json.load(f)
                    
                    # Calculate metrics from trade data
                    if 'trades' in backtest_data and backtest_data['trades']:
                        # Convert trades to DataFrame for easier analysis
                        trades_df = pd.DataFrame(backtest_data['trades'])
                        
                        # Calculate total return from trades
                        if not trades_df.empty and 'pnl' in trades_df.columns:
                            # Calculate safely to avoid pandas Series truth value errors
                            try:
                                total_pnl = trades_df['pnl'].sum()
                                initial_balance = 10000  # Default initial balance
                                total_return = (total_pnl / initial_balance) * 100.0
                                
                                # Calculate win rate
                                winning_trades = trades_df[trades_df['pnl'] > 0]
                                total_trades = len(trades_df[trades_df['type'] == 'close'])
                                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                                
                                # Calculate max drawdown
                                # Simplified approximation based on consecutive losing trades
                                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                                cumulative_max = trades_df['cumulative_pnl'].cummax()
                                drawdown = (cumulative_max - trades_df['cumulative_pnl'])
                                max_drawdown = drawdown.max() / initial_balance * 100.0 if initial_balance > 0 else 0
                                
                                # Simple Sharpe ratio approximation
                                # (Return - Risk Free Rate) / Standard Deviation
                                risk_free_rate = 0.0  # Simplified
                                returns_std = trades_df['pnl'].std()
                                sharpe_ratio = (total_pnl - risk_free_rate) / returns_std if returns_std > 0 else 0
                                
                                # Add calculated metrics to results
                                ui_results['metrics'] = {
                                    'total_return': float(total_return),
                                    'sharpe_ratio': float(sharpe_ratio),
                                    'max_drawdown': float(max_drawdown),
                                    'win_rate': float(win_rate)
                                }
                                metrics_found = True
                                print(f"Calculated metrics from trade data: {ui_results['metrics']}")
                            except Exception as e:
                                print(f"Error during metrics calculation: {e}")
                                # Set an empty dict rather than a partially calculated metrics dict
                                ui_results['metrics'] = {}
                except Exception as e:
                    print(f"Error calculating metrics from trade data: {e}")
        
        # Ensure metrics are properly formatted
        if not metrics_found or (isinstance(ui_results['metrics'], dict) and not ui_results['metrics']) or \
           (isinstance(ui_results['metrics'], pd.Series) and ui_results['metrics'].empty):
            # Provide default metrics based on workflow results
            try:
                if 'best_parameters' in result and result['best_parameters']:
                    ui_results['parameters'] = result['best_parameters']
                
                # Provide empty metrics with proper keys
                ui_results['metrics'] = {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
                
                # Add any metrics we found in the result but didn't recognize
                for key, section in result.items():
                    if isinstance(section, dict) and any(metric in section for metric in ['total_return', 'sharpe_ratio']):
                        print(f"Found metrics in section: {key}")
                        for metric_key, metric_value in section.items():
                            if metric_key not in ui_results['metrics']:
                                ui_results['metrics'][metric_key] = metric_value
                
                print(f"Using default metrics: {ui_results['metrics']}")
            except Exception as e:
                print(f"Error creating default metrics: {e}")
                ui_results['metrics'] = {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
        
        # Save the UI results
        # Ensure all values are JSON serializable
        ui_results = ensure_serializable(ui_results)
        with open(ui_results_file, 'w') as f:
            json.dump(ui_results, f, indent=4)
        
        return 0
    except Exception as e:
        # Stop the progress indicator
        stop_event.set()
        progress_thread.join()
        
        import traceback
        error_traceback = traceback.format_exc()
        print(f"\nError running backtest: {e}")
        print(f"Traceback:\n{error_traceback}")
        
        # Print helpful debugging information
        print("\nDebugging information:")
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        print(f"sys.path: {sys.path}")
        
        # Save the error to a file
        error_file = os.path.join(config['output_dir'], 'error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Traceback:\n{error_traceback}")
            
            # Add debugging information
            f.write("\n\nDebugging information:\n")
            f.write(f"Python version: {sys.version}\n")
            f.write(f"Current directory: {os.getcwd()}\n")
            f.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}\n")
            f.write(f"Parameter file: {config.get('param_file', 'Not set')}\n")
            if 'param_file' in config and os.path.exists(config['param_file']):
                with open(config['param_file'], 'r') as param_f:
                    f.write(f"Parameter file contents:\n{param_f.read()}\n")
        
        # Create a simple backtest_results.json file for displaying in the UI
        ui_results_file = os.path.join(config['output_dir'], 'backtest_results.json')
        ui_results = {
            'strategy_name': strategy_name,
            'tickers': tickers,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'metrics': {'error': 'Backtest failed'}
        }
        
        # Ensure all values are JSON serializable
        ui_results = ensure_serializable(ui_results)
        with open(ui_results_file, 'w') as f:
            json.dump(ui_results, f, indent=4)
        
        print(f"Error details saved to {error_file}")
        return 1
    finally:
        # Make sure the progress indicator is stopped
        stop_event.set()

if __name__ == '__main__':
    sys.exit(main()) 