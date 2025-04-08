# Trading Strategy Backtester - Frontend

A Flask web application that provides a user-friendly interface for the Trading Strategy Backtester framework.

## Features

- Configure and run backtests through a web interface
- Visualize backtest results with interactive charts
- Compare performance of different trading strategies
- Optimize strategy parameters
- View detailed metrics and trade statistics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Access to the Trading Strategy Backtester project

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/trading-strategy-backtester-frontend.git
cd trading-strategy-backtester-frontend
```

2. **Create and activate a virtual environment**

```bash
# On Linux/macOS
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy the template environment file and modify it with your specific settings:

```bash
cp .env.template .env
```

Edit the `.env` file to set the correct paths and configuration values:

```
# Set the path to your trading-strategy-backtester installation
BACKTESTER_ROOT=/path/to/trading-strategy-backtester
BACKTESTER_OUTPUT_DIR=/path/to/trading-strategy-backtester/output
```

5. **Configure application settings**

Copy the template configuration file:

```bash
cp config.template.json config.json
```

Edit `config.json` to customize the application settings.

## Running the Application

Start the Flask development server:

```bash
flask run
```

Or run it directly:

```bash
python app.py
```

Access the application in your web browser at http://localhost:5000

## Development

### Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, and other static assets
- `run_backtest.py` - Script to run backtests in the background
- `config.json` - Application configuration

### Adding a New Feature

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes
3. Test thoroughly
4. Submit a pull request

## Testing

Run the test suite:

```bash
python -m pytest
```

For a specific test file:

```bash
python -m pytest test_strategy.py
```

## Deployment

For production deployment:

1. Set `FLASK_ENV=production` in your `.env` file
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 app:app
   ```

3. Consider using a reverse proxy like Nginx to handle static files and SSL

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the Trading Strategy Backtester framework
- Utilizes Flask, Bootstrap, and Chart.js 