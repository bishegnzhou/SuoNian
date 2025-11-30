import logging
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme

from rich.markup import escape

import sys

# Custom theme for the console
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "success": "bold green",
    "thought": "italic cyan",
    "code": "bold yellow",
    "result": "white"
})

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

console = Console(theme=custom_theme)

def setup_logging():
    """Sets up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("agent.log", encoding="utf-8"),
            # We don't add RichHandler here because we want manual control over console output
            # to keep it "elegant" and not just a stream of logs.
        ]
    )
    # Create a separate logger for the file that doesn't propagate to root
    file_logger = logging.getLogger("agent_file")
    file_logger.setLevel(logging.DEBUG)
    return file_logger

# Global file logger instance
logger = setup_logging()

def log_step(step_name, status="INFO"):
    """Logs a step to the file."""
    logger.info(f"[{step_name}] {status}")

def print_panel(content, title, style="info"):
    """Prints a rich panel to the console."""
    console.print(Panel(content, title=title, border_style=style, expand=False))

def print_status(message, style="info"):
    """Prints a status message."""
    console.print(f"[{style}]{escape(message)}[/{style}]")
