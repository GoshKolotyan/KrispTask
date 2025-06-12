import logging
import sys
import functools
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Callable, Any

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        formatted = super().format(record)
        record.levelname = levelname
        return formatted
    
class LoggerDecorator:
    """Main decorator class for logging functionality"""
    
    def __init__(self, name: str = "armenian_asr",level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None, console_output: bool = True,file_output: bool = False,colored_output: bool = True
    ):
        self.logger = self._setup_logger(
            name, level, log_file, console_output, file_output, colored_output
        )
    
    def _setup_logger(
        self, 
        name: str,
        level: Union[str, int],
        log_file: Optional[Union[str, Path]],
        console_output: bool,
        file_output: bool,
        colored_output: bool
    ) -> logging.Logger:
        """Setup the logger with specified configuration"""
        
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()
        
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if colored_output:
                console_formatter = ColoredFormatter(fmt=format_string, datefmt=date_format)
            else:
                console_formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
            
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if file_output or log_file:
            if log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"armenian_asr_{timestamp}.log"
            
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_execution(self, log_args: bool = True,log_result: bool = False, log_time: bool = True,level: str = "INFO"):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Log function start
                log_func = getattr(self.logger, level.lower())
                log_func(f"Starting {func.__name__}")
                
                # Log arguments if requested
                if log_args:
                    if args:
                        log_func(f"  Args: {args}")
                    if kwargs:
                        log_func(f"  Kwargs: {kwargs}")
                
                try:
                    result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    if log_time:
                        log_func(f"Completed {func.__name__} in {execution_time:.2f}s")
                    else:
                        log_func(f"Completed {func.__name__}")
                    
                    if log_result:
                        log_func(f"  Result: {result}")
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error(f"Error in {func.__name__} after {execution_time:.2f}s: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def log_method(
        self, 
        log_args: bool = False, 
        log_result: bool = False, 
        log_time: bool = True,
        level: str = "INFO"
    ):
        """
        Decorator specifically for class methods
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(self_obj, *args, **kwargs):
                start_time = time.time()
                
                log_func = getattr(self.logger, level.lower())
                class_name = self_obj.__class__.__name__
                log_func(f"{class_name}.{func.__name__} started")
                
                if log_args:
                    if args:
                        log_func(f"  Args: {args}")
                    if kwargs:
                        log_func(f"  Kwargs: {kwargs}")
                
                try:
                    result = func(self_obj, *args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    if log_time:
                        log_func(f"{class_name}.{func.__name__} completed in {execution_time:.2f}s")
                    else:
                        log_func(f"{class_name}.{func.__name__} completed")
                    
                    if log_result:
                        log_func(f"  Result: {result}")
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error(f"{class_name}.{func.__name__} failed after {execution_time:.2f}s: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def log_phase(self, phase_name: str, level: str = "INFO"):
        """
        Decorator for logging training phases 
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                log_func = getattr(self.logger, level.lower())
                
                log_func("="*60)
                log_func(f"PHASE: {phase_name.upper()}")
                log_func("="*60)
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    execution_time = time.time() - start_time
                    log_func("="*60)
                    log_func(f"PHASE COMPLETED: {phase_name.upper()}")
                    log_func("="*60)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error("="*60)
                    self.logger.error(f"PHASE FAILED: {phase_name.upper()}")
                    self.logger.error(f"Error: {e}")
                    self.logger.error("="*60)
                    raise
            
            return wrapper
        return decorator
    
    def log_config(self, config, title: str = "Configuration"):
        """Log configuration object"""
        self.logger.info(f"   {title}:")
        
        if hasattr(config, '__dict__'):
            for key, value in config.__dict__.items():
                if isinstance(value, dict):
                    self.logger.info(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        self.logger.info(f"      {sub_key}: {sub_value}")
                else:
                    self.logger.info(f"    {key}: {value}")
        else:
            self.logger.info(f"  {config}")
    
    
    
    def info(self, msg: str): self.logger.info(f"Info {msg}")
    def debug(self, msg: str): self.logger.debug(f"Debug {msg}")
    def warning(self, msg: str): self.logger.warning(f"Warrrning {msg}")
    def error(self, msg: str): self.logger.error(f"Error {msg}")
    def success(self, msg: str): self.logger.info(f" {msg}")
    def progress(self, msg: str): self.logger.info(f"Processing {msg}")

log = LoggerDecorator(
    name="armenian_asr",
    level=logging.INFO,
    console_output=True,
    file_output=False,
    colored_output=True
)

# # Example usage
# if __name__ == "__main__":
#     # Test the logger
#     log.info("Logger initialized")
#     log.success("This is a success message")
#     log.warning("This is a warning")
#     log.error("This is an error")
    
#     # Test decorators
#     @log.log_execution(log_args=True, log_time=True)
#     def sample_function(x, y, name="test"):
#         time.sleep(1)  # Simulate work
#         return x + y
    
#     @log.log_phase("Data Loading")
#     def load_data():
#         time.sleep(2)  # Simulate data loading
#         return "data loaded"
    
#     # Test the decorators
#     result = sample_function(5, 10, name="example")
#     data = load_data()
    
#     # Test system info
#     log.log_system_info()