import converters
import argparse
import os

parser = argparse.ArgumentParser(description="Simple example of a training script.")

# Training Settings
parser.add_argument("--checkpoint", required=True, help="The target path of the checkpoint.")
parser.add_argument("--diffusers", required=True, help="The destination path for the diffusers model")

args = parser.parse_args()

converters.Convert_SD_to_Diffusers(args.checkpoint, args.diffusers)