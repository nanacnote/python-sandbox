from . import (STYLE_UNDERLINE, STYLE_BOLD,
               STYLE_HEADER, STYLE_CYAN, STYLE_GREEN, STYLE_FAIL, STYLE_CLEAR)
from .cleaner import Cleaner
from .engine import Engine

import os
import sys


LOOP = True


if __name__ == "__main__":
    # clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # get script arguments
    script_arg, preprocess_arg, process_arg, model_arg = sys.argv

    # clean the corpus
    Cleaner().strategy("nohtml").run()

    # instantiate a search engine
    # use clean_build flag to build up new index on first run this takes a longer time
    engine = Engine(clean_build=True).preprocess(
        preprocess_arg).process(process_arg).model(model_arg)

    # init message
    print(f"""
    ___________________________________________________

        {STYLE_HEADER}{STYLE_BOLD}Base Search Engine{STYLE_CLEAR}
        Enter {STYLE_BOLD}"Q"{STYLE_CLEAR} to exit application.

        Version: 0.1.0
        Author: Owusu K (adjeibohyen@hotmail.com)

    ___________________________________________________
    """)

    # process user input
    while LOOP:
        q = input(
            f"Enter a search query:\n{STYLE_BOLD}{STYLE_UNDERLINE}")
        print(f"{STYLE_CLEAR}")
        if q.lower() == "q":
            LOOP = False
        else:
            res = engine.query(q.strip())
            res = res if len(res) else f"{STYLE_FAIL}no match{STYLE_CLEAR}"
            print(f"{STYLE_CLEAR}{STYLE_CYAN}{res}{STYLE_CLEAR}\n")

    # exit message
    print(f"\n{STYLE_CLEAR}{STYLE_BOLD}{STYLE_GREEN}Application exited successfully!{STYLE_CLEAR}\n")
