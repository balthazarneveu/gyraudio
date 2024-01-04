import argparse


def shared_parser(help="Train models for audio separation"):
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument("-e",  "--experiments", type=int, nargs="+", required=True,
                        help="Experiment ids to be trained sequentially")
    return parser
