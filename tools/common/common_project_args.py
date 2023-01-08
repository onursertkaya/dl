"""Common project CLI arguments."""
import argparse


def make_argument_parser(project_name: str):
    """Make an ArgumentParser with common arguments."""
    parser = argparse.ArgumentParser(f"Start a {project_name} experiment.")
    return _update(parser)


def update_argument_parser(parser: argparse.ArgumentParser):
    """Update an existing argument parser with common arguments."""
    return _update(parser)


def _update(parser: argparse.ArgumentParser):
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-t", "--tasks", type=str)
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-e", "--experiment-dir", type=str)
    parser.add_argument("-r", "--restore-from-chkpt", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    return parser
