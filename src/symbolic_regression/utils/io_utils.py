import os
import dill
import smtplib
import datetime

import pandas as pd

from typing import List, Dict
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

def save_results(
    df: pd.DataFrame, 
    equations: Dict, 
    features: Dict, 
    prefix: str = "data"
) -> str:
    """Save df, equations, and features to a pickle file with a timestamped filename."""

    data = {
        'df': df,
        'equations': equations,
        'features': features
    }

    timestamp = datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")
    filename = f"{prefix}{'_' if prefix else ''}{timestamp}.pkl"

    with open(filename, "wb") as f:
        dill.dump(data, f)

    return filename

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
