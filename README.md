# Trading Strategy Backtester Frontend

A web-based frontend for the Trading Strategy Backtester.

## Overview

This frontend provides a user-friendly interface to configure and run backtests using the Trading Strategy Backtester. It allows users to:

- Select a trading strategy
- Configure parameters for the strategy
- Select ticker symbols
- Set date ranges and other backtest settings
- View results of completed backtests

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the setup script to verify your environment and create necessary directories:

```bash
python setup.py
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open a web browser and navigate to:

```
http://localhost:5000
```

3. Configure your backtest:
   - Select a strategy from the dropdown
   - Enter comma-separated ticker symbols (e.g., AAPL,MSFT,GOOG)
   - Set the date range for backtesting
   - Adjust the in-sample ratio and other parameters
   - Configure strategy-specific parameters

4. Click the "Run Backtest" button to start the backtest
5. Check the "Results" page to view the status and results of your backtests

## Architecture

The frontend consists of:

- Flask web application (`app.py`)
- HTML templates for the user interface (`templates/`)
- Command-line script for running backtests (`run_backtest.py`)

Backtests are run in a separate process to avoid blocking the web interface. The Flask app tracks the status of running backtests and displays results when they're complete.

## Folder Structure

```
frontend/
├── app.py                  # Main Flask application
├── run_backtest.py         # Script for running backtests
├── setup.py                # Environment setup script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── static/                 # Static assets (CSS, JS, images)
├── templates/              # HTML templates
│   ├── index.html          # Main page
│   └── results.html        # Results page
└── temp/                   # Temporary files for backtest configuration
```

## How It Works

1. User configures a backtest through the web interface
2. Configuration is saved to a temporary JSON file
3. A background process runs the backtest using the unified workflow
4. Results are saved to the output directory
5. The web interface displays the status and results

This architecture allows multiple backtests to run simultaneously without blocking the user interface.

## Troubleshooting

### Import Errors

If you encounter import errors related to the `strategies` module, this is likely due to Python's module resolution. The setup script should address this, but if problems persist, ensure:

1. The trading-strategy-backtester directory is at `/home/pyzron02/trading-strategy-backtester`
2. The correct paths are added to Python's sys.path

### Process Errors

If backtests fail to start or complete:

1. Check the log files in the `/home/pyzron02/frontend/temp` directory
2. Ensure the backtester code is functioning by running a simple test directly
3. Look for error messages in the terminal where you started the Flask application

### Directory Access

If you encounter permission errors:

1. Ensure you have write access to both the frontend and backtester directories
2. Run `python setup.py` to create any missing directories

## Notes

- Backtests can take a significant amount of time to complete, especially with Monte Carlo simulations
- You can monitor the status of running backtests on the "Results" page
- Results are stored in the `/home/pyzron02/trading-strategy-backtester/output` directory 