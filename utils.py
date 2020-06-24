import logging
import os


def setup_logger(name, log_file, level=logging.INFO):
    
    formatter = logging.Formatter('%(message)s')
    
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def check_create_path(path):
    
    dir_path = path
    if os.path.exists(path) == False:
        print("Creating directory for output and Log")
        os.mkdir(os.getcwd() + '/model/')
        os.mkdir(os.getcwd() + '/model/Logs')
        
        dir_path = os.getcwd() + '/model/'
    
    return dir_path
        
def check_file(path):
    
    assert os.path.isfile(path), path +'File Not Found' 
        