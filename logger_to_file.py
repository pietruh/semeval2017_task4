'''
Helper class for printing to the terminal and to log to a file
To use:
... # on top of your script add line:
from logger_to_file import Logger
...# on the beggining of your main add line:
sys.stdout = Logger("./logs/log_")
# if path to logging directory: "./logs/log_"
'''

import time
import sys

class Logger(object):
    def __init__(self, path_to_logs_dir):
        self.terminal = sys.stdout
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.log = open(path_to_logs_dir + timestr + ".dat", "a")

    def write(self, message):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.terminal.write(message)
        if(message == "\n"):
            self.log.write(message)
        else:
            self.log.write(timestr + ": " + message)

    def flush(self):
        self.log.write("\n")
        self.terminal.flush()
