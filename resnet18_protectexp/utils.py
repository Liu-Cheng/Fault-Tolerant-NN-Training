import logging
import time

def current_time():
    tm = time.localtime()
    return time.strftime('%Y%m%d%H%M%S', tm)

class Logger: ...

logger = Logger()

def setlogger(name=''):
    global logger
    file_handler = logging.FileHandler(current_time()+'_'+name+'.log')
    console_handler = logging.StreamHandler()
    file_handler.setLevel('DEBUG')
    console_handler.setLevel('INFO')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.logger = logging.getLogger()
    logger.logger.setLevel('DEBUG')
    logger.logger.addHandler(file_handler)
    logger.logger.addHandler(console_handler)
