from argparse import ArgumentParser, Namespace
from pathlib import Path

from atlas_manager import build_pinpoint_atlases_path

parser = ArgumentParser(
    prog="Atlas Converter",
    description="Tool to convert BrainGlobe formatted atlases into Pinpoint V compatible atlases.",
)

# Only convert Allen CCF Mouse. Overrides build all.
parser.add_argument(
    "-m", "--mouse", action="store_true", help="Only convert Allen CCF Mouse atlases."
)

# Convert all BrainGlobe atlases.
parser.add_argument(
    "-a", "--all", action="store_true", help="Convert all BrainGlobe atlases."
)

# Convert atlases at path.
parser.add_argument(
    "-c",
    "--convert",
    default=Path.home() / ".brainglobe",
    type=Path,
    help="Convert BrainGlobe formatted atlases at the specified path (default: %(default)s).",
)

# Conversion output path.
parser.add_argument(
    "-o",
    "--output",
    default=build_pinpoint_atlases_path(),
    type=Path,
    help="Conversion output folder path (default: %(default)s).",
)

# Serve the build directory.
parser.add_argument(
    "-s",
    "--serve",
    default=build_pinpoint_atlases_path(),
    type=Path,
    help="Serve a folder (with converted atlases) on a local HTTP server with CORS configured (default: %(default)s).",
)

# Server port.
parser.add_argument(
    "-p",
    "--port",
    default=3000,
    type=int,
    help="Converted atlas server port (default: %(default)s).",
)


def get_parsed_arguments() -> Namespace:
    return parser.parse_args()
