import os
from dotenv import load_dotenv, dotenv_values

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

config = dotenv_values(".env")