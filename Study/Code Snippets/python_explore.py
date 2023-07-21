vars()
dir()
locals(), globals()
help()
print(np.zeros.__doc__) # prints doc string of the function

from loguru import logger
logger.debug("loguru debug")
import logging
logging.basicConfig(level = logging.INFO)

logging.debug("debug") 
