#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontend application for the trading strategy backtester.
"""
import os
import sys
import json
import subprocess
import numpy as np
import math
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

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

# Import components from the backtester
# Import the registry directly from its file to avoid module resolution issues
from src.strategies.registry import get_registered_strategies

# Monkey patch json encoder to handle non-serializable objects
# This ensures any object can be serialized
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            # Special check for built-in functions (more specific than just callable)
            if isinstance(obj, type(min)) or isinstance(obj, type(sum)) or type(obj).__name__ == 'builtin_function_or_method':
                return f"[Built-in function: {getattr(obj, '__name__', 'unknown')}]"
            # Regular callables
            elif callable(obj):
                return "[Function]"
            elif hasattr(obj, 'item') and callable(obj.item):
                try:
                    return float(obj.item())
                except:
                    return 0.0
            elif hasattr(obj, 'tolist') and callable(obj.tolist):
                return obj.tolist()
            elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return obj.to_dict()
            elif str(type(obj)).startswith("<class 'numpy.") or str(type(obj)).startswith("<class 'pandas."):
                # Check for pandas DataFrame attributes that might contain built-in functions
                if str(type(obj)).startswith("<class 'pandas."):
                    # Convert to a basic dict representation and filter out methods
                    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                        try:
                            data_dict = obj.to_dict()
                            # Filter out any potentially problematic values
                            for k in list(data_dict.keys()):
                                if callable(data_dict[k]) or str(type(data_dict[k])).startswith("<class 'builtin_"):
                                    data_dict[k] = str(data_dict[k])
                            return data_dict
                        except:
                            return str(obj)
                return str(obj)
            elif hasattr(obj, '__dict__'):
                # Filter out any methods or built-in functions from __dict__
                safe_dict = {}
                for k, v in obj.__dict__.items():
                    if k.startswith('_'):
                        continue
                    if callable(v) or str(type(v)).startswith("<class 'builtin_"):
                        safe_dict[k] = str(v)
                    else:
                        safe_dict[k] = v
                return safe_dict
            
            # Final try to check for any built-in functions in obj attributes
            if hasattr(obj, '__dir__'):
                for attr_name in dir(obj):
                    if not attr_name.startswith('_'):
                        try:
                            attr = getattr(obj, attr_name)
                            if isinstance(attr, type(min)) or type(attr).__name__ == 'builtin_function_or_method':
                                # If we find any built-in method, convert the whole object to string
                                return str(obj)
                        except:
                            pass
            
            return json.JSONEncoder.default(self, obj)
        except Exception as e:
            return f"[Unserializable {type(obj).__name__}]"

# Replace the default encoder
json._default_encoder = SafeJSONEncoder()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'trading_strategy_backtester_secret_key')

# Default parameters from environment variables
DEFAULT_START_DATE = os.getenv('DEFAULT_START_DATE', '2015-01-01')
DEFAULT_END_DATE = os.getenv('DEFAULT_END_DATE', '2021-12-31')
DEFAULT_IN_SAMPLE_RATIO = float(os.getenv('DEFAULT_IN_SAMPLE_RATIO', '0.7'))
DEFAULT_NUM_SIMULATIONS = int(os.getenv('DEFAULT_NUM_SIMULATIONS', '1000'))

# Track running backtests
RUNNING_BACKTESTS = {}

def ensure_json_serializable(obj):
    """Recursively convert a nested structure to contain only JSON serializable types.
    This function is extremely aggressive about sanitizing data to avoid JSON serialization errors."""
    
    # Handle None, strings, booleans directly
    if obj is None:
        return None
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, str):
        return obj
    
    # Handle numeric types
    elif isinstance(obj, (int, float)):
        # Handle NaN and Infinity values
        try:
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return float(obj) if isinstance(obj, float) else int(obj)
        except:
            return 0
    
    # Handle dictionaries
    elif isinstance(obj, dict):
        try:
            # Create a new dictionary with sanitized keys and values
            result = {}
            for k, v in obj.items():
                # Skip built-in functions or methods in the keys
                if callable(k) or (hasattr(k, '__call__') and not isinstance(k, type)):
                    continue
                    
                # Convert key to string to ensure it's serializable
                str_key = str(k)
                
                # Check if value is a built-in function
                if isinstance(v, type(min)) or isinstance(v, type(sum)) or type(v).__name__ == 'builtin_function_or_method':
                    try:
                        result[str_key] = f"[Built-in function: {v.__name__}]"
                    except AttributeError:
                        result[str_key] = "[Built-in function]"
                # Skip other callable functions or methods in the values
                elif callable(v) or (hasattr(v, '__call__') and not isinstance(v, type)):
                    result[str_key] = '[Function/Method - not serializable]'
                else:
                    result[str_key] = ensure_json_serializable(v)
            return result
        except Exception as e:
            print(f"Error sanitizing dictionary: {e}")
            return {"error": "Dictionary sanitization error"}
    
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        try:
            # Create a new list with sanitized values
            result = []
            for item in obj:
                # Skip built-in functions or methods
                if callable(item) or (hasattr(item, '__call__') and not isinstance(item, type)):
                    result.append('[Function/Method - not serializable]')
                else:
                    result.append(ensure_json_serializable(item))
            return result
        except Exception as e:
            print(f"Error sanitizing list/tuple: {e}")
            return ["Error serializing list data"]
    
    # Handle numpy arrays and pandas objects
    elif str(type(obj)).startswith("<class 'numpy.") or str(type(obj)).startswith("<class 'pandas."):
        try:
            # For DataFrames, explicitly check and remove methods
            if str(type(obj)) == "<class 'pandas.core.frame.DataFrame'>":
                data_dict = {}
                if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                    try:
                        data_dict = obj.to_dict()
                    except:
                        data_dict = {'error': 'Failed to convert DataFrame to dict'}
                        
                # Ensure no methods are included
                for k in list(data_dict.keys()):
                    if isinstance(data_dict[k], dict):
                        for inner_k in list(data_dict[k].keys()):
                            if callable(data_dict[k][inner_k]) or (hasattr(data_dict[k][inner_k], '__call__') and not isinstance(data_dict[k][inner_k], type)):
                                data_dict[k][inner_k] = str(data_dict[k][inner_k])
                            
                return ensure_json_serializable(data_dict)
                
            # For Series, explicitly handle
            elif str(type(obj)) == "<class 'pandas.core.series.Series'>":
                return ensure_json_serializable(obj.to_dict() if hasattr(obj, 'to_dict') else str(obj))
            
            # Regular handling for other numpy/pandas objects
            elif hasattr(obj, 'tolist'):
                # Convert to list first
                return ensure_json_serializable(obj.tolist())
            elif hasattr(obj, 'to_dict'):
                # Convert to dict first
                return ensure_json_serializable(obj.to_dict())
            elif hasattr(obj, 'item') and callable(obj.item):
                # For numpy scalars
                try:
                    value = obj.item()
                    if isinstance(value, (int, float)):
                        if math.isnan(value) or math.isinf(value):
                            return str(value)
                        return float(value) if isinstance(value, float) else int(value)
                    return str(value)
                except:
                    return 0
            else:
                # Last resort for numpy/pandas: convert to string
                return str(obj)
        except Exception as e:
            print(f"Error sanitizing numpy/pandas object: {e}")
            return "Unsupported data format"
    
    # Handle built-in functions or methods
    elif callable(obj) or (hasattr(obj, '__call__') and not isinstance(obj, type)):
        # Special check for built-in functions
        if isinstance(obj, type(min)) or isinstance(obj, type(sum)) or type(obj).__name__ == 'builtin_function_or_method':
            try:
                return f"[Built-in function: {obj.__name__}]"
            except AttributeError:
                return "[Built-in function]"
        return '[Function/Method - not serializable]'
    
    # Handle any other objects
    else:
        try:
            # Try to convert to a basic type
            if hasattr(obj, 'to_dict') and callable(obj.to_dict):
                return ensure_json_serializable(obj.to_dict())
            elif hasattr(obj, 'to_list') and callable(obj.to_list):
                return ensure_json_serializable(obj.to_list())
            elif hasattr(obj, '__dict__'):
                # For objects with __dict__, convert to a dictionary of basic attributes
                sanitized_dict = {}
                for k, v in obj.__dict__.items():
                    if not k.startswith('_') and not callable(v):
                        sanitized_dict[k] = ensure_json_serializable(v)
                return sanitized_dict
            else:
                # Last resort: convert to string
                return str(obj)
        except Exception as e:
            print(f"Error sanitizing object {type(obj)}: {e}")
            return f"[Object of type {type(obj).__name__}]"

@app.route('/')
def index():
    """Render the main page of the application."""
    # Get all registered strategies
    strategies = get_registered_strategies()
    
    return render_template(
        'index.html',
        strategies=strategies,
        default_start_date=DEFAULT_START_DATE,
        default_end_date=DEFAULT_END_DATE,
        default_in_sample_ratio=DEFAULT_IN_SAMPLE_RATIO,
        default_num_simulations=DEFAULT_NUM_SIMULATIONS,
        running_backtests=RUNNING_BACKTESTS
    )

@app.route('/strategy/<strategy_name>')
def get_strategy_params(strategy_name):
    """Get default parameters for a strategy."""
    try:
        # This would get filled out with actual parameter defaults
        # You could store these in a config file or derive them from the strategy class
        strategy_defaults = {
            'SimpleStock': {'sma_period': 20, 'position_size': 100},
            'MACrossover': {'fast_period': 10, 'slow_period': 30, 'position_size': 100},
            'MultiPosition': {'sma_period': 20, 'position_size': 50, 'max_positions': 5},
            'AuctionMarket': {'lookback_period': 20, 'position_size': 100}
        }
        
        # Print available strategies for debugging
        print(f"Available strategies: {', '.join([s['name'] for s in get_registered_strategies()])}")
        print(f"Requested strategy: {strategy_name}")
        
        if strategy_name in strategy_defaults:
            return jsonify({'success': True, 'params': strategy_defaults[strategy_name]})
        else:
            return jsonify({'success': False, 'error': f'Strategy {strategy_name} not found'})
    except Exception as e:
        import traceback
        print(f"Error getting strategy parameters: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

def create_parameter_grid(form_data):
    """Create a parameter grid from form data for optimization."""
    param_grid = {}
    
    # Collect all grid parameter specifications
    for key in form_data:
        if key.startswith('grid_') and '_min' in key:
            param_name = key.replace('grid_', '').replace('_min', '')
            min_val = form_data.get(f'grid_{param_name}_min', '')
            max_val = form_data.get(f'grid_{param_name}_max', '')
            step_val = form_data.get(f'grid_{param_name}_step', '')
            
            # Only add if all three values are provided
            if min_val and max_val and step_val:
                try:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    step_val = float(step_val)
                    
                    # Generate values in the range
                    if step_val > 0:
                        values = np.arange(min_val, max_val + step_val/2, step_val).tolist()
                        
                        # Convert to integers if all values are whole numbers
                        if all(v.is_integer() for v in np.array(values)):
                            values = [int(v) for v in values]
                            
                        param_grid[param_name] = values
                except ValueError:
                    # Skip malformed values
                    continue
    
    return param_grid

@app.route('/submit', methods=['POST'])
def submit_backtest():
    """Handle form submission to run a backtest."""
    try:
        # Get form data
        strategy_name = request.form.get('strategy')
        tickers = [ticker.strip() for ticker in request.form.get('tickers', '').split(',') if ticker.strip()]
        start_date = request.form.get('start_date', DEFAULT_START_DATE)
        end_date = request.form.get('end_date', DEFAULT_END_DATE)
        in_sample_ratio = float(request.form.get('in_sample_ratio', DEFAULT_IN_SAMPLE_RATIO))
        num_simulations = int(request.form.get('num_simulations', DEFAULT_NUM_SIMULATIONS))
        num_workers = int(request.form.get('num_workers', 1))
        verbose = True if request.form.get('verbose') == 'on' else False
        keep_permuted_data = True if request.form.get('keep_permuted_data') == 'on' else False
        run_optimization = True if request.form.get('run_optimization') == 'on' else False
        
        # Debug log
        print(f"Submit backtest request: strategy={strategy_name}, tickers={tickers}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Optimization: {run_optimization}, Workers: {num_workers}, Simulations: {num_simulations}")
        
        # Create a timestamp for the output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.getenv('BACKTESTER_OUTPUT_DIR', '/home/pyzron02/trading-strategy-backtester/output'), 
            f"{strategy_name}_{timestamp}")
        
        # Create temp directories if they don't exist
        temp_dir = os.getenv('TEMP_DIR', '/home/pyzron02/frontend/temp')
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # Generate configuration
        config = {
            'strategy_name': strategy_name,
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'in_sample_ratio': in_sample_ratio,
            'num_simulations': num_simulations,
            'num_workers': num_workers,
            'verbose': verbose,
            'keep_permuted_data': keep_permuted_data,
            'run_optimization': run_optimization,
            'output_dir': output_dir,
            'timestamp': timestamp
        }
        
        # Handle parameter grid for optimization or single parameter values
        if run_optimization:
            param_grid = create_parameter_grid(request.form)
            config['param_grid'] = param_grid
            print(f"Parameter grid created: {param_grid}")
            
            # Create a parameter file
            param_file = os.path.join(temp_dir, f"params_{timestamp}.json")
            with open(param_file, 'w') as f:
                json.dump({'parameters': param_grid}, f, indent=4)
                
            config['param_file'] = param_file
            print(f"Parameter file created at: {param_file}")
        else:
            # Extract single parameter values
            param_values = {}
            for key, value in request.form.items():
                if key.startswith('param_'):
                    param_name = key.replace('param_', '')
                    # Convert to appropriate type
                    try:
                        if '.' in value:
                            param_values[param_name] = float(value)
                        else:
                            param_values[param_name] = int(value)
                    except ValueError:
                        param_values[param_name] = value
            
            config['params'] = param_values
            print(f"Single parameter values: {param_values}")
        
        # Create a config file for the run_backtest.py script
        config_file = os.path.join(temp_dir, f"config_{timestamp}.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Config file created at: {config_file}")
        
        # Run the backtest script in a background process with absolute paths to ensure it works correctly
        python_path = sys.executable  # Get the current Python executable
        run_backtest_script = os.path.abspath('/home/pyzron02/frontend/run_backtest.py')
        
        log_file = os.path.join(temp_dir, f"backtest_{timestamp}.log")
        with open(log_file, 'w') as log:
            # Set up environment with correct paths
            env = dict(os.environ)
            pythonpath = f"{project_root}:{os.path.join(project_root, 'src')}"
            if 'PYTHONPATH' in env:
                pythonpath += f":{env['PYTHONPATH']}"
            env['PYTHONPATH'] = pythonpath
            
            print(f"Running backtest with PYTHONPATH={pythonpath}")
            
            process = subprocess.Popen(
                [python_path, run_backtest_script, '--config', config_file],
                stdout=log,
                stderr=log,
                text=True,
                env=env
            )
        
        # Add to running backtests
        RUNNING_BACKTESTS[timestamp] = {
            'process': process,
            'strategy': strategy_name,
            'tickers': tickers,
            'output_dir': output_dir,
            'log_file': log_file,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        flash(f'Backtest submitted for {strategy_name}. Check the results page for updates.')
        return redirect(url_for('index'))
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error submitting backtest: {error_details}")
        flash(f'Error: {str(e)}')
        return redirect(url_for('index'))

@app.route('/results')
def list_results():
    """List all available backtest results with enhanced metrics."""
    results_dir = os.getenv('BACKTESTER_OUTPUT_DIR', '/home/pyzron02/trading-strategy-backtester/output')
    
    # Use a simpler approach with only primitive data types
    # This ensures we avoid any serialization issues with complex objects
    simple_results = []
    
    # Debug flag to find problematic parts
    find_bad_objects = True
    
    # Add running jobs with minimal data
    for job_id, job in list(RUNNING_BACKTESTS.items()):
        if job['process'].poll() is not None:
            # Process has finished
            exit_code = job['process'].returncode
            if exit_code != 0:
                # Failed - add an error result
                simple_results.append({
                    'folder': str(os.path.basename(job['output_dir'])),
                    'strategy': str(job['strategy']),
                    'timestamp': str(job['start_time']),
                    'status': 'Failed',
                    'error': True,
                    'metrics': {'error': f'Process failed with exit code {exit_code}'}
                })
            
            # Remove from running processes
            del RUNNING_BACKTESTS[job_id]
        else:
            # Still running
            simple_results.append({
                'folder': str(os.path.basename(job['output_dir'])),
                'strategy': str(job['strategy']),
                'timestamp': str(job['start_time']),
                'status': 'Running',
                'running': True,
                'metrics': {}
            })
    
    # Add completed jobs with minimal data
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            # --- SIMPLIFIED APPROACH: JUST USE BASIC JSON FILES ---
            # Load only the essential data from JSON files and avoid complex operations
            
            # Don't skip any folders including the SimpleStock test folder
            print(f"Processing folder: {folder}")
            
            # Look for complete_results.json as primary source
            complete_results_file = os.path.join(folder_path, 'complete_results.json')
            parameters_file = os.path.join(folder_path, 'parameters.json')
            backtest_results_file = os.path.join(folder_path, 'backtest_results.json')
            
            # Initialize data structures
            result_data = {}
            param_data = {}
            metrics_data = {}
            timestamp = "Unknown"
            
            # First try to load complete_results.json (primary source)
            if os.path.exists(complete_results_file):
                try:
                    with open(complete_results_file, 'r') as f:
                        result_data = json.load(f)
                    
                    print(f"Successfully loaded: {complete_results_file}")
                    
                    # Collect metrics from complete_results.json
                    metrics_data = {}
                    
                    # 1. Extract metrics from optimization_results.best_metrics if available
                    if 'optimization_results' in result_data and 'best_metrics' in result_data['optimization_results']:
                        for key, value in result_data['optimization_results']['best_metrics'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    metrics_data[key] = float(value)
                                else:
                                    metrics_data[key] = str(value)
                            except:
                                metrics_data[key] = 0.0
                    
                    # 2. Extract metrics from monte_carlo_results if available
                    if 'monte_carlo_results' in result_data and 'metrics' in result_data['monte_carlo_results']:
                        mc_metrics = result_data['monte_carlo_results']['metrics']
                        for metric_name, metric_data in mc_metrics.items():
                            if isinstance(metric_data, dict) and 'original' in metric_data:
                                try:
                                    metrics_data[metric_name] = float(metric_data['original'])
                                except:
                                    metrics_data[metric_name] = 0.0
                    
                    # Extract parameters from complete_results.json
                    if 'best_parameters' in result_data:
                        param_data = result_data['best_parameters']
                except Exception as e:
                    print(f"Error loading complete_results.json for {folder}: {e}")
            
            # Try to load parameters.json (for parameters)
            if os.path.exists(parameters_file):
                try:
                    with open(parameters_file, 'r') as f:
                        params_json = json.load(f)
                    
                    print(f"Successfully loaded: {parameters_file}")
                    
                    # Override with parameters from parameters.json if available
                    if 'parameters' in params_json:
                        param_data = params_json['parameters']
                except Exception as e:
                    print(f"Error loading parameters.json for {folder}: {e}")
            
            # If complete_results.json is missing or has no metrics, try backtest_results.json
            if (not metrics_data or len(metrics_data) == 0) and os.path.exists(backtest_results_file):
                try:
                    with open(backtest_results_file, 'r') as f:
                        backtest_data = json.load(f)
                    
                    print(f"Successfully loaded: {backtest_results_file}")
                    
                    # Use strategy and timestamp from backtest_results.json if needed
                    if 'strategy_name' not in result_data and 'strategy_name' in backtest_data:
                        result_data['strategy_name'] = backtest_data['strategy_name']
                    
                    if 'timestamp' in backtest_data:
                        timestamp = backtest_data['timestamp']
                    
                    # Use metrics from backtest_results.json if available
                    if 'metrics' in backtest_data:
                        for key, value in backtest_data['metrics'].items():
                            try:
                                if isinstance(value, (int, float)):
                                    metrics_data[key] = float(value)
                                else:
                                    metrics_data[key] = str(value)
                            except:
                                metrics_data[key] = 0.0
                except Exception as e:
                    print(f"Error loading backtest_results.json for {folder}: {e}")
            
            # Skip if we don't have any meaningful data
            if not result_data and not metrics_data:
                print(f"No usable results found for {folder}")
                continue
            
            try:
                # Create a basic result with only simple types
                simple_result = {
                    'folder': str(folder),
                    'strategy': str(result_data.get('strategy_name', 'Unknown')),
                    'timestamp': str(timestamp),
                    'status': 'Completed',
                    'metrics': {},
                    'enhanced_data': {}
                }
                
                # Add basic metrics from the data
                for key, value in metrics_data.items():
                    if isinstance(value, (int, float)):
                        try:
                            simple_result['metrics'][str(key)] = float(value)
                        except:
                            simple_result['metrics'][str(key)] = 0.0
                    else:
                        simple_result['metrics'][str(key)] = str(value)
                
                print(f"Metrics for {folder}: {simple_result['metrics']}")
                
                # --- ENHANCED DATA: MINIMAL VERSION ONLY ---
                try:
                    # 1. Always include parameters
                    if param_data:
                        params = {}
                        for k, v in param_data.items():
                            params[str(k)] = str(v)
                        simple_result['enhanced_data']['parameters'] = params
                    
                    # 2. Include period information if available
                    if 'period' in result_data:
                        simple_result['enhanced_data']['period'] = {
                            'start': str(result_data['period'].get('start', '')),
                            'end': str(result_data['period'].get('end', ''))
                        }
                    
                    # 3. Include tickers if available
                    if 'tickers' in result_data:
                        simple_result['enhanced_data']['tickers'] = [str(t) for t in result_data['tickers']]
                    
                    # 4. Include walk forward metrics if available (as string)
                    if 'walk_forward_metrics' in result_data:
                        simple_result['enhanced_data']['walk_forward_metrics'] = str(result_data['walk_forward_metrics'])
                    
                    # 5. Include monte carlo summary if available
                    if 'monte_carlo_results' in result_data and 'analysis' in result_data['monte_carlo_results']:
                        monte_carlo_summary = {}
                        p_values = result_data['monte_carlo_results']['analysis'].get('p_values', {})
                        for k, v in p_values.items():
                            try:
                                monte_carlo_summary[str(k)] = float(v)
                            except:
                                monte_carlo_summary[str(k)] = str(v)
                        simple_result['enhanced_data']['monte_carlo_summary'] = monte_carlo_summary
                        
                    # 6. Add empty placeholders for standard sections
                    # This allows the UI to show tabs without breaking
                    simple_result['enhanced_data']['equity_curve'] = {
                        'dates': [],
                        'values': []
                    }
                    
                    simple_result['enhanced_data']['monthly_performance'] = {
                        'months': [],
                        'values': []
                    }
                    
                    simple_result['enhanced_data']['trade_stats'] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0.0,
                        'avg_win': 0.0,
                        'avg_loss': 0.0,
                        'win_loss_ratio': 0.0,
                        'ticker_performance': {}
                    }
                    
                except Exception as e:
                    print(f"Error building enhanced data: {e}")
                
                # --- SAFETY CHECK: ENSURE SERIALIZABLE ---
                is_safe = True
                try:
                    # Test if the entire result is serializable
                    json.dumps(simple_result)
                except Exception as e:
                    is_safe = False
                    print(f"Warning: Result for {folder} is not serializable, using minimal version: {e}")
                
                # Only add if it's safe
                if is_safe:
                    simple_results.append(simple_result)
                else:
                    # Add minimal version instead
                    simple_results.append({
                        'folder': str(folder),
                        'strategy': str(result_data.get('strategy_name', 'Unknown')),
                        'timestamp': str(timestamp),
                        'status': 'Completed (Basic Data)',
                        'metrics': metrics_data,
                        'enhanced_data': {'note': 'Only basic metrics available'}
                    })
                    
            except Exception as e:
                print(f"Error processing results for {folder}: {e}")
                # Skip problematic results entirely
                continue
    
    # Remove any problematic items that might have slipped through
    safe_results = []
    for result in simple_results:
        try:
            # Perform a final safety check by serializing each result individually
            json.dumps(result)
            safe_results.append(result)
        except Exception as e:
            print(f"Removing unsafe result for folder: {result.get('folder', 'unknown')}: {e}")
            # Skip this result entirely
    
    # Final verification of entire results list
    try:
        # Try to serialize
        print("Attempting to serialize final results...")
        json_str = json.dumps(safe_results)
        print("Serialization successful!")
        
        # If we get here, everything is serializable
        return render_template('results.html', results=safe_results)
    except TypeError as final_error:
        print(f"Final serialization error: {final_error}")
        
        # Try to identify which object is causing the issue
        for i, result in enumerate(safe_results):
            try:
                json.dumps(result)
            except TypeError as e:
                print(f"Error in result {i} (folder: {result.get('folder', 'unknown')}): {e}")
        
        # Return a minimal error version
        error_message = f"Error serializing results: {str(final_error)}"
        minimal_results = []
        
        # Create minimal results with just basics - strings only
        for result in simple_results:
            try:
                minimal_results.append({
                    'folder': str(result.get('folder', 'unknown')),
                    'strategy': str(result.get('strategy', 'unknown')),
                    'timestamp': str(result.get('timestamp', 'unknown')),
                    'status': 'Basic',
                    'metrics': {'note': 'Basic data only'},
                    'enhanced_data': {'note': 'Data not available'}
                })
            except:
                # Skip entirely if we can't even create a basic entry
                continue
        
        return render_template('results.html', 
                              results=minimal_results, 
                              error_message=error_message)

@app.route('/backtest_status')
def backtest_status():
    """Return the status of all running backtests."""
    status = []
    
    # Update running processes status
    for job_id, job in list(RUNNING_BACKTESTS.items()):
        if job['process'].poll() is not None:
            # Process has finished
            exit_code = job['process'].returncode
            status.append({
                'id': job_id,
                'strategy': job['strategy'],
                'status': 'Completed' if exit_code == 0 else 'Failed',
                'exit_code': exit_code
            })
            
            # Remove from running processes
            del RUNNING_BACKTESTS[job_id]
        else:
            # Still running
            status.append({
                'id': job_id,
                'strategy': job['strategy'],
                'status': 'Running',
            })
    
    return jsonify({'success': True, 'status': status})

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """API endpoint to run a backtest (for AJAX calls)."""
    try:
        data = request.get_json()
        
        strategy_name = data.get('strategy')
        tickers = data.get('tickers', [])
        start_date = data.get('start_date', DEFAULT_START_DATE)
        end_date = data.get('end_date', DEFAULT_END_DATE)
        in_sample_ratio = float(data.get('in_sample_ratio', DEFAULT_IN_SAMPLE_RATIO))
        num_simulations = int(data.get('num_simulations', DEFAULT_NUM_SIMULATIONS))
        num_workers = int(data.get('num_workers', 1))
        verbose = data.get('verbose', False)
        keep_permuted_data = data.get('keep_permuted_data', False)
        param_grid = data.get('params', {})
        
        # Create a timestamp for the output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.getenv('BACKTESTER_OUTPUT_DIR', '/home/pyzron02/trading-strategy-backtester/output'), 
            f"{strategy_name}_{timestamp}")
        
        # Create temp directories if they don't exist
        temp_dir = os.getenv('TEMP_DIR', '/home/pyzron02/frontend/temp')
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # Create a config file for the run_backtest.py script
        config_file = os.path.join(temp_dir, f"config_{timestamp}.json")
        config = {
            'strategy_name': strategy_name,
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'in_sample_ratio': in_sample_ratio,
            'num_simulations': num_simulations,
            'num_workers': num_workers,
            'verbose': verbose,
            'keep_permuted_data': keep_permuted_data,
            'params': param_grid,
            'output_dir': output_dir,
            'timestamp': timestamp
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Run the backtest script in a background process
        log_file = os.path.join(temp_dir, f"backtest_{timestamp}.log")
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                [sys.executable, '/home/pyzron02/frontend/run_backtest.py', '--config', config_file],
                stdout=log,
                stderr=log,
                text=True
            )
        
        # Add to running backtests
        RUNNING_BACKTESTS[timestamp] = {
            'process': process,
            'strategy': strategy_name,
            'tickers': tickers,
            'output_dir': output_dir,
            'log_file': log_file,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify({
            'success': True, 
            'message': 'Backtest started', 
            'job_id': timestamp,
            'output_dir': output_dir
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cancel_backtest/<job_id>', methods=['POST'])
def cancel_backtest(job_id):
    """Cancel a running backtest."""
    if job_id in RUNNING_BACKTESTS:
        job = RUNNING_BACKTESTS[job_id]
        if job['process'].poll() is None:  # Process is still running
            job['process'].terminate()
            flash(f'Backtest for {job["strategy"]} canceled.')
        else:
            flash(f'Backtest for {job["strategy"]} already completed.')
        
        del RUNNING_BACKTESTS[job_id]
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Job not found'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 