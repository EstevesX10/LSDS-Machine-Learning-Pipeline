# Defining which submodules to import when using from <package> import *
__all__ = ["loadConfig", "loadPathsConfig", "loadQueries"]

from .Config import (loadConfig, loadPathsConfig, loadQueries)