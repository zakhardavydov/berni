import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Run simulation',
        description='Run LLM BERNI simulation'
    )
    parser.add_argument("path")
    return parser
