import argparse
import logging
import os
from my_furhat_backend.models.llm_factory import HuggingFaceLLM

def test():
    llm = HuggingFaceLLM(max_new_tokens=100, top_k=40, temperature=0.2)
    


