import argparse


def shared_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e",  "--experiments", type=int, nargs="+", required=True,
                        help="Experiment ids to be trained sequentially")
    return parser
