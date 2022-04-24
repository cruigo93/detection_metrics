import os
import click
import utils

@click.command()
@click.option("--ground", "-g", help="Path to file with Ground Truth bounding boxes")
@click.option("--detection", "-d", help="Path to file with Prediction bounding boxes")
def main(ground, detection):
    utils.make_calculation(ground, detection)

if __name__ == "__main__":
    main()