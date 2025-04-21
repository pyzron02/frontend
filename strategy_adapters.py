#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy adapters to handle special cases for different strategy parameter formats.
"""
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_adapted_param_file(strategy_name, params, output_dir):
    """
    Create a properly formatted parameter file for a specific strategy,
    handling special cases and format requirements.
    
    Args:
        strategy_name (str): Name of the strategy
        params (dict): Parameters dictionary
        output_dir (str): Directory to save parameter file
        
    Returns:
        str: Path to the created parameter file
    """
    # Strategy-specific adaptations
    if strategy_name == 'AuctionMarket':
        return create_auction_market_params(params, output_dir)
    
    # Default parameter file creation if no special handling needed
    param_file = os.path.join(output_dir, 'parameters.json')
    
    # Wrap parameters if they're not already wrapped
    if isinstance(params, dict) and 'parameters' not in params:
        param_data = {'parameters': params}
    else:
        param_data = params
    
    with open(param_file, 'w') as f:
        json.dump(param_data, f, indent=4)
    
    return param_file

def create_auction_market_params(params, output_dir):
    """
    Create parameters file for AuctionMarket strategy which has special requirements.
    Instead of using nested 'parameters' structure, it needs direct parameters.
    
    Args:
        params (dict): Parameters dictionary
        output_dir (str): Directory to save parameter file
        
    Returns:
        str: Path to the created parameter file
    """
    # Extract parameters from nested structure if needed
    if isinstance(params, dict) and 'parameters' in params:
        params = params['parameters']
    
    # Ensure essential parameters are set
    if 'param_preset' not in params:
        params['param_preset'] = 'default'
    
    # Essential params for AuctionMarket
    essential_params = [
        'value_area', 
        'use_vwap', 
        'use_volume_profile', 
        'position_size', 
        'risk_percent', 
        'use_atr_sizing', 
        'atr_period'
    ]
    
    # Add defaults for missing parameters
    defaults = {
        'value_area': 0.7,
        'use_vwap': True,
        'use_volume_profile': True,
        'position_size': 100,
        'risk_percent': 0.01,
        'use_atr_sizing': True,
        'atr_period': 14
    }
    
    for param in essential_params:
        if param not in params:
            params[param] = defaults.get(param)
    
    # Create parameter file
    param_file = os.path.join(output_dir, 'parameters.json')
    
    # AuctionMarket uses flat parameters
    with open(param_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    logger.info(f"Created AuctionMarket parameters file: {param_file}")
    return param_file