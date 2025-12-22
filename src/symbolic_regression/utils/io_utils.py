import os
import sys
import time
import socket
import smtplib
import datetime
import traceback
import dill

from functools import partial
from typing import Any, Callable, Dict, List, Optional
from email.message import EmailMessage

def load_pickle_files(
    directory: str,
    suffix: str = '.pkl'
) -> List[Dict]:
    """
    Recursively finds and loads data from pickle files in a directory.

    This function walks through a directory and its subdirectories,
    finds all files ending with the specified suffix, and loads
    the data from them using pickle.

    Args:
        directory (str): The path to the root directory to search.
        suffix (str, optional): The file extension to look for.

    Returns:
        List[Dict]: A list containing the loaded data from each pickle file. 
    """

    loaded_data = [] # List to store loaded data.
    
    # Walk through the directory and its subdirectories.
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the filename ends with the desired suffix.
            if filename.endswith(suffix):
                # Construct the full path to the file.
                file_path = os.path.join(root, filename)

                # Open the file and load its content using pickle.
                with open(file_path, 'rb') as f:
                    loaded_data.append(dill.load(f))
    
    return loaded_data

def load_pickle_file(
    file_path: str
) -> Dict:
    """
    Load data from a single pickle file.
    
    Args:
        file_path (str): Path to the pickle file.
        
    Returns:
        Dict: The loaded data from the pickle file.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """

    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return dill.load(f)

def persist(
    func: Optional[Callable] = None,
    *args,
    pickle_dir: str = '.',
    filename: str = 'data.pkl',
    execute: bool = True,
    save_result: bool = False,
    exclude_keys: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Persist function/data and metadata to disk, with optional execution.
    
    This function serves two purposes: function execution tracking (save function,
    args, execute it, and optionally save result) and data storage (just save data
    without execution when func=None or execute=False).

    Parameters
    ----------
    func : Callable, optional
        Callable to execute with the provided arguments. If None, only data in 
        kwargs will be saved.
    pickle_dir : str, default='.'
        Directory path where the pickle file will be saved.
    filename : str, default='data.pkl'
        Name of the pickle file to create.
    execute : bool, default=True
        Whether to execute the function if provided. If False, only saves 
        function and args without execution.
    save_result : bool, default=False
        Whether to save the function's return value to the pickle file. Only 
        applicable when execute=True. Useful for reproducibility but may 
        increase file size significantly.
    exclude_keys : list of str, optional
        List of kwargs keys to exclude from pickling. Useful for excluding
        non-picklable objects like multiprocessing managers. The function will
        still execute with these arguments, they just won't be saved.
    *args
        Positional arguments to persist and pass to func.
    **kwargs
        Keyword arguments to persist and pass to func, or data to save if 
        execute=False.

    Returns
    -------
    Any
        Result of func(*args, **kwargs) if func is provided and execute=True, 
        otherwise the pickle file path (str).
    """

    # Create pickle directory if it doesn't exist
    os.makedirs(pickle_dir, exist_ok=True)
    
    # Capture start timestamp
    timestamp = datetime.datetime.now()

    # Filter kwargs to exclude specified keys
    if exclude_keys:
        exclude_keys = [k for k in exclude_keys if k in kwargs]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in exclude_keys}

    else:
        exclude_keys = []
        filtered_kwargs = kwargs
    
    # Collect metadata
    metadata = {
        'timestamp': timestamp.isoformat(" "),
        'python_version': sys.version,
        'hostname': socket.gethostname(),
        'excluded_keys': exclude_keys,
    }
    
    # Add function metadata if function is provided
    if func:
        # Handle partial functions
        if isinstance(func, partial):
            actual_func = func.func
            function_name = actual_func.__name__
            function_module = actual_func.__module__

        else:
            function_name = func.__name__
            function_module = func.__module__
        
        metadata.update({
            'function_name': function_name,
            'function_module': function_module,
        })
    
    result = None # Function execution result
    error = None # Exception if occurred during execution
    
    # Determine storage mode
    if not execute or func is None:
        # Data storage mode: save filtered kwargs as flat data structure
        data = {**filtered_kwargs, 'metadata': metadata, 'status': 'success'}

    else:
        # Function execution mode: save function, args, filtered kwargs
        data = {
            'function': func,
            'args': args,
            'kwargs': filtered_kwargs,  # Save filtered kwargs
            'metadata': metadata,
            'status': 'success'
        }
        
        # Execute the function and capture result/error
        execution_start = time.time()
        
        try:
            result = func(*args, **kwargs)  # Execute with all kwargs
            execution_duration = time.time() - execution_start
            
            # Save result if requested
            if save_result: data['result'] = result
            
        except Exception as e:
            execution_duration = time.time() - execution_start
            error = e

            data['status'] = 'failure'
            data['error'] = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
    
        # Record execution duration
        data['metadata']['execution_duration_seconds'] = execution_duration
    
    # Save everything to pickle file
    pickle_path = os.path.join(pickle_dir, filename)
    with open(pickle_path, 'wb') as f:
        dill.dump(data, f)
    
    # Re-raise the exception if one occurred in execution mode
    if error: raise error
    
    # Return result or pickle path
    return result if (func and execute) else pickle_path

def send_email(
    subject: str, 
    body_message: str, 
    sender_email: str, 
    receiver_email: str,
    app_password: str, 
    smtp_server: str, 
    smtp_port: int, 
):
    """
    Send an email.

    Parameters
    ----------
    subject : str
        Subject of the email.
    body_message : str
        Body content of the email.
    sender_email : str
        Email address of the sender.
    receiver_email : str
        Email address of the receiver.
    app_password : str
        App-specific password for the sender's email account.
    smtp_server : str
        SMTP server address.
    smtp_port : int
        SMTP server port.

    Returns
    -------
    (success, error)
        success: True if sent, False otherwise.
        error: error message if failed, else None.
    """

    # Compose the email message
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(body_message)

    try:
        # Establish a secure SSL connection to the SMTP server
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            # Connect to the SMTP server and send the email
            server.login(sender_email, app_password)
            server.send_message(msg)
            return True, None
            
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
