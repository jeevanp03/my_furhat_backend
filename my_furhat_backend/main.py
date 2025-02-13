import argparse
import logging
import os
from models.llm import HuggingFaceLLM

def test():
    llm = HuggingFaceLLM(max_new_tokens=100, top_k=40, temperature=0.2)
    


