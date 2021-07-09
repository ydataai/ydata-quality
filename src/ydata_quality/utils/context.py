import sys
import os

def noprint(func):
    "Disables all prints called within a method."
    def wrap(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w') # block print
        result = func(*args, **kwargs) # calculate results
        sys.stdout = sys.__stdout__ # unblock print
        return result
    return wrap

