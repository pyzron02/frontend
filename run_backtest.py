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
import inspect
from datetime import datetime
import threading
import pandas as pd
import numpy as np

# Import local strategy adapters
try:
    from strategy_adapters import create_adapted_param_file
except ImportError:
    # If import fails, define a dummy function
    def create_adapted_param_file(strategy_name, params, output_dir):
        print(f"WARNING: strategy_adapters module not found, using default parameter handling")
        return None

# Try to import dotenv, gracefully handle if not installed
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
except ImportError:
    # Define a dummy function if python-dotenv is not installed
    def load_dotenv():
        print("Warning: python-dotenv package not installed. Environment variables from .env file will not be loaded.")
        pass
    # Ensure the function is called for consistent behavior
    load_dotenv()

# Add the trading-strategy-backtester to the path
project_root = os.getenv('BACKTESTER_ROOT', '/home/pyzron02/trading-strategy-backtester')
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Try to import the workflow modules - we'll try again in main() if this fails
try:
    # Import unified_workflow for workflow selection
    from src.workflows.unified_workflow import run_unified_workflow
    # Import complete_workflow directly for running the complete workflow
    from src.workflows.complete_workflow import run_complete_workflow
except ImportError:
    print("Warning: Could not import workflow modules. Will try again during execution.")
    run_unified_workflow = None
    run_complete_workflow = None

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

def update_progress(progress_file, progress, status=None, current_step=None, current_step_progress=None, total_steps=None):
    """
    Update the progress file with current progress information.
    
    Args:
        progress_file (str): Path to the progress file
        progress (int): Overall progress percentage (0-100)
        status (str, optional): Current status message
        current_step (str, optional): Name of the current step
        current_step_progress (int, optional): Progress of the current step (0-100)
        total_steps (int, optional): Total number of steps
    """
    if not progress_file or not os.path.exists(progress_file):
        return False
    
    try:
        # Read existing progress data
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        # Update with new values
        if progress is not None:
            progress_data['progress'] = min(100, max(0, progress))
        if status:
            progress_data['status'] = status
        if current_step:
            progress_data['current_step'] = current_step
        if current_step_progress is not None:
            progress_data['current_step_progress'] = min(100, max(0, current_step_progress))
        if total_steps:
            progress_data['total_steps'] = total_steps
        
        # Update timestamp
        progress_data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write updated progress data
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
        
        return True
    except Exception as e:
        print(f"Error updating progress file: {e}")
        return False

def create_parameter_file(params, output_dir, strategy_name=None):
    """Create a parameter file for the backtest."""
    # Use strategy adapter if available
    if strategy_name == 'AuctionMarket':
        try:
            # Try to use the specialized adapter for AuctionMarket strategy
            if 'create_adapted_param_file' in globals():
                param_file = create_adapted_param_file(strategy_name, params, output_dir)
                if param_file:
                    print(f"Using strategy adapter for {strategy_name}")
                    return param_file
        except Exception as e:
            print(f"Error using strategy adapter: {e}, falling back to default")
    
    # If no adapter or adapter failed, use default parameter handling
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
    
    # Handle strategy-specific parameter adaptations for non-AuctionMarket strategies
    if strategy_name and strategy_name != 'AuctionMarket':
        # Specific handling for other strategies can be added here
        pass
    elif strategy_name == 'AuctionMarket':
        # For AuctionMarket, don't use nested parameters - use flat format 
        if isinstance(params, dict) and 'parameters' in params:
            param_data = params['parameters']
        else:
            param_data = params
            
        # Ensure param_preset is set
        if 'param_preset' not in param_data:
            param_data['param_preset'] = 'default'
            print("Added default param_preset for AuctionMarket strategy")
    
    # Create temp file first to verify it's properly formatted
    temp_param_file = os.path.join(output_dir, 'temp_parameters.json')
    try:
        with open(temp_param_file, 'w') as f:
            json.dump(param_data, f, indent=4)
        
        # Verify it's valid JSON by reading it back
        with open(temp_param_file, 'r') as f:
            test_data = json.load(f)
        
        # If we get here, the JSON is valid, so rename to final file
        os.rename(temp_param_file, param_file)
    except Exception as e:
        print(f"Error creating parameter file: {e}")
        # If there was an error, make sure we still create a valid parameters file
        with open(param_file, 'w') as f:
            fallback_params = {}
            if strategy_name == 'AuctionMarket':
                # Flat parameters for AuctionMarket
                fallback_params = {
                    'param_preset': 'default',
                    'value_area': 0.7,
                    'position_size': 100,
                    'use_vwap': True,
                    'use_volume_profile': True,
                    'risk_percent': 0.01,
                    'use_atr_sizing': True,
                    'atr_period': 14
                }
            else:
                # Nested parameters for other strategies
                fallback_params = {'parameters': {'position_size': 100}}
            json.dump(fallback_params, f, indent=4)
        print(f"Created fallback parameter file due to formatting error")
    
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

# Custom progress callback for the workflow
def progress_callback(step, progress, message, progress_file=None):
    """Callback function for workflow progress updates."""
    if not progress_file:
        return
    
    # Map workflow steps to progress percentages
    step_weights = {
        'initialization': 5,
        'data_loading': 10,
        'optimization': 30,
        'walk_forward': 30,
        'monte_carlo': 20,
        'finalization': 5
    }
    
    # Calculate overall progress based on step weights and current step progress
    steps = list(step_weights.keys())
    current_step_idx = steps.index(step) if step in steps else 0
    
    # Calculate progress from completed steps
    completed_steps_progress = sum(step_weights[steps[i]] for i in range(current_step_idx))
    
    # Add progress from current step
    current_step_progress = step_weights[step] * (progress / 100)
    overall_progress = int(completed_steps_progress + current_step_progress)
    
    # Update progress file
    update_progress(
        progress_file,
        progress=overall_progress,
        status='Running',
        current_step=f"{step.replace('_', ' ').title()}: {message}",
        current_step_progress=progress,
        total_steps=len(step_weights)
    )
    
    # Also print to console
    print(f"[PROGRESS] {step}: {progress}% - {message} (Overall: {overall_progress}%)")

def main():
    parser = argparse.ArgumentParser(description='Run a backtest with parameters from a JSON file')
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    parser.add_argument('--output', type=str, help='Output directory (overrides the one in config)')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug output')
    parser.add_argument('--progress-file', type=str, help='Path to file for tracking progress')
    args = parser.parse_args()
    
    debug_mode = args.debug
    progress_file = args.progress_file
    
    # Initialize progress if a progress file is provided
    if progress_file:
        update_progress(progress_file, 0, "Starting", "Initializing backtest", 0, 6)
    
    # Load the configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override output directory if specified
    if args.output:
        config['output_dir'] = args.output
    
    # Default output directory if not specified
    if 'output_dir' not in config:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['output_dir'] = os.path.join(os.getenv('BACKTESTER_OUTPUT_DIR', 
                                                     os.path.join(project_root, 'output')), 
                               f"{config.get('strategy_name', 'backtest')}_{timestamp}")
    
    # Ensure the correct data directory is set
    if 'data_dir' not in config:
        config['data_dir'] = os.path.join(project_root, 'input')
    
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
    
    # Update progress - Configuration loaded
    if progress_file:
        update_progress(progress_file, 5, "Preparing", "Loading configuration", 100, 6)
    
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
        # First try with the full path import
        try:
            # Try the absolute import path with project_root
            sys.path.insert(0, project_root)
            from src.strategies.registry import get_strategy_class
        except ImportError:
            # Fall back to the relative import
            from strategies.registry import get_strategy_class
            
        strategy_class = get_strategy_class(strategy_name)
        print(f"Found strategy class: {strategy_class.__name__}")
    except Exception as e:
        print(f"WARNING: Could not verify strategy class: {e}")
    
    # Update progress - Strategy loaded
    if progress_file:
        update_progress(progress_file, 10, "Preparing", "Strategy loaded", 100, 6)
    
    # Ensure param_file is created even if we're using fixed parameters
    if run_optimization:
        print(f"Running optimization with parameter grid")
        # Use the parameter grid provided or from param_file
        if 'param_grid' in config:
            param_grid = config['param_grid']
            # Create a parameter file if not already specified
            if 'param_file' not in config:
                param_file = create_parameter_file(param_grid, config['output_dir'], strategy_name)
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
            param_file = create_parameter_file(params, config['output_dir'], strategy_name)
            config['param_file'] = param_file
            print(f"Parameters formatted for optimization: {param_file}")
    
    # Ensure the parameter file exists
    if 'param_file' not in config:
        print("WARNING: No parameter file specified, creating an empty one")
        # Create empty parameters with strategy-specific defaults
        empty_params = {'parameters': {}}
        if strategy_name == 'AuctionMarket':
            empty_params['parameters'] = {
                'param_preset': 'default',
                'value_area': 0.7,
                'position_size': 100
            }
            
        param_file = create_parameter_file(empty_params, config['output_dir'], strategy_name)
        config['param_file'] = param_file
    
    # Update progress - Parameters loaded
    if progress_file:
        update_progress(progress_file, 15, "Preparing", "Parameters loaded", 100, 6)
    
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
        
        # Update progress - Starting workflow
        if progress_file:
            update_progress(progress_file, 20, "Running", "Starting workflow", 0, 6)
        
        # Import the run_complete_workflow function here to ensure it's using the correct Python path
        try:
            # First try with the full path import with project_root
            sys.path.insert(0, project_root)
            from src.workflows.unified_workflow import run_complete_workflow
        except ImportError:
            # Fall back to the relative import
            from workflows.unified_workflow import run_complete_workflow
        
        # Create a custom progress reporter wrapper function that includes the progress file
        def progress_reporter(step, progress, message):
            return progress_callback(step, progress, message, progress_file)
        
        # Use the specific workflow type if specified, otherwise use complete workflow
        workflow_type = config.get('workflow_type', 'complete')
        
        # Prepare common workflow arguments
        workflow_args = {
            "strategy_name": strategy_name,
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "param_file": config.get('param_file'),
            "output_dir": config['output_dir'],
            "data_dir": config.get('data_dir', os.path.join(project_root, 'input')),
            "verbose": config.get('verbose', False),
            "initial_capital": config.get('initial_capital', 100000.0),
            "commission": config.get('commission', 0.001),
            "keep_permuted_data": config.get('keep_permuted_data', False),
            "plot": config.get('generate_plots', False),  # Respect the generate_plots setting
        }
        
        # Add workflow-specific arguments
        if workflow_type == 'complete':
            workflow_args.update({
                "n_simulations": config.get('num_simulations', 1000),
                "n_trials": 50,  # Default number of optimization trials
                "enhanced_plots": config.get('enhanced_plots', False),
            })
            
            # Special handling for AuctionMarket parameters
            if strategy_name == 'AuctionMarket':
                # For AuctionMarket, param_file may not work properly due to nested structure
                # Remove param_file and directly pass the parameters
                if 'param_file' in workflow_args:
                    del workflow_args['param_file']
                    
                # Extract parameters from the parameter file
                if 'param_file' in config and os.path.exists(config['param_file']):
                    try:
                        with open(config['param_file'], 'r') as f:
                            direct_params = json.load(f)
                        
                        # Parameters should be directly at the root level for AuctionMarket
                        workflow_args['parameters'] = direct_params
                        print("Using direct parameters for AuctionMarket strategy")
                    except Exception as e:
                        print(f"Error loading parameters for AuctionMarket: {e}")
                        # Set default parameters
                        workflow_args['parameters'] = {
                            'param_preset': 'default',
                            'value_area': 0.7,
                            'position_size': 100
                        }
            
            # Try to create a progress callback for unified/complete workflow
            if 'progress_callback' in inspect.signature(run_complete_workflow).parameters:
                workflow_args["progress_callback"] = progress_reporter if progress_file else None
            
            # Run the complete workflow
            print(f"Running complete workflow with strategy: {strategy_name}")
            result = run_complete_workflow(**workflow_args)
        else:
            # If using unified_workflow, add in_sample_ratio and any other specific parameters
            # Check for specific workflow types
            if workflow_type in ['simple', 'optimization', 'monte_carlo', 'walkforward']:
                workflow_args.update({
                    "in_sample_ratio": config.get('in_sample_ratio', 0.7),
                    "n_simulations": config.get('num_simulations', 1000),  # Use n_simulations instead of num_simulations
                    "n_trials": 50,  # Set default number of optimization trials
                })
            else:
                # For any other workflow (especially unified workflow)
                workflow_args.update({
                    "in_sample_ratio": config.get('in_sample_ratio', 0.7),
                    "num_simulations": config.get('num_simulations', 1000),
                    "num_workers": config.get('num_workers', 1),
                    "seed": config.get('seed', 42),
                })
            
            print(f"Running {workflow_type} workflow with strategy: {strategy_name}")
            result = run_unified_workflow(workflow_type, **workflow_args)
        
        # Stop the progress indicator
        stop_event.set()
        progress_thread.join()
        
        # Save the results
        results_file = os.path.join(config['output_dir'], 'complete_results.json')
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=4, cls=json._default_encoder.__class__)
        
        # Add additional code to create a merged result for the UI
        try:
            merged_result = {'strategy_name': strategy_name}
            
            # Check for backtest results
            backtest_results_file = os.path.join(config['output_dir'], '2_backtest', 'backtest_results.json')
            if os.path.exists(backtest_results_file):
                with open(backtest_results_file, 'r') as f:
                    backtest_data = json.load(f)
                    if 'metrics' in backtest_data:
                        if 'metrics' not in merged_result:
                            merged_result['metrics'] = {}
                        merged_result['metrics'].update(backtest_data['metrics'])
                    if 'parameters' in backtest_data:
                        merged_result['parameters'] = backtest_data['parameters']
            
            # Check for Monte Carlo results
            monte_carlo_results_file = os.path.join(config['output_dir'], '3_monte_carlo', f'{strategy_name}_monte_carlo_results.json')
            if os.path.exists(monte_carlo_results_file):
                with open(monte_carlo_results_file, 'r') as f:
                    mc_data = json.load(f)
                    if 'metrics' in mc_data:
                        if 'metrics' not in merged_result:
                            merged_result['metrics'] = {}
                        merged_result['metrics'].update(mc_data['metrics'])
                    if 'analysis' in mc_data:
                        merged_result['monte_carlo_results'] = {'analysis': mc_data['analysis']}
            
            # Additionally save to a UI friendly format
            if merged_result:
                ui_results_file = os.path.join(config['output_dir'], 'ui_results.json')
                with open(ui_results_file, 'w') as f:
                    json.dump(merged_result, f, indent=4, cls=json._default_encoder.__class__)
                print(f"Created UI-friendly results file at {ui_results_file}")
        except Exception as e:
            print(f"Warning: Failed to create merged results file: {e}")
        
        print(f"Results saved to {results_file}")
        
        # Update progress - Results saved
        if progress_file:
            update_progress(progress_file, 95, "Finishing", "Saving results", 100, 6)
            time.sleep(1)  # Small delay to ensure progress is visible
            update_progress(progress_file, 100, "Completed", "Backtest complete", 100, 6)
        
        return 0
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error running backtest: {e}")
        print(error_traceback)
        
        # Log the error to a file
        error_file = os.path.join(config['output_dir'], 'backtest_error.log')
        with open(error_file, 'w') as f:
            f.write(f"Error: {e}\n\n")
            f.write(error_traceback)
        
        print(f"Error details saved to {error_file}")
        
        # Update progress with error
        if progress_file:
            update_progress(progress_file, -1, "Failed", f"Error: {str(e)}", 0, 6)
        
        # Stop the progress indicator
        stop_event.set()
        if progress_thread.is_alive():
            progress_thread.join()
        
        return 1

if __name__ == '__main__':
    sys.exit(main()) 