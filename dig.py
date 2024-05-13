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

    max_round = 8

    override = False
    
    args = parser.parse_args()

    investigator = BaseGameInvestigator(args.setup, args.party, run_flips_on_initial_bias=False, track_round_prompt=False)
    
    investigator.process(override, clustering_method="hdbscan")
