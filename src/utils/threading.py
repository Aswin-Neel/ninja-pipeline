import os

def get_available_threads():
    return max(1, os.cpu_count() - 1)
