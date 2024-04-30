import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from berni.investigation import BaseGameInvestigator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Dig one game',
        description='Run analysis of one game setup'
    )
    parser.add_argument("setup")
    parser.add_argument("party")

    override = False
    
    args = parser.parse_args()

    investigator = BaseGameInvestigator(args.setup, args.party)
    
    investigator.process(override)
