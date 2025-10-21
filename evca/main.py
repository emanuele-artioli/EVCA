"""Command-line interface for the EVCA package."""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import List, Tuple

from . import PACKAGE_ROOT


def check_existence(input_args: argparse.Namespace) -> Tuple[List[str], bool]:
    files_list: List[str] = []
    if input_args.dir:
        files_list = [file for file in glob.glob(f"{input_args.dir}/*.yuv")]
    elif input_args.input:
        input_path = Path(input_args.input)
        if input_path.is_file():
            files_list = [str(input_path)]
    else:
        return files_list, False

    return files_list, True


def print_custom_help() -> None:
    print("EVCA:    Enhanced Video Complexity Analyzer v1.0")
    print("Usage:   python -m evca.main [options]")
    print("\nOptions:")
    print("-h /--help                Show this help text and exit.")
    print("-m /--method              Feature extraction method. Default is EVCA. [VCA, EVCA, SITI] ")
    print("-t /--transform           Discrete transform method. Default is DCT. [DCT, DWT, DCT_B] ")
    print("-fi/--filter              Edge detection filter. Default is sobel filter. [sobel, canny] ")
    print("-i /--input               Raw YUV input file name.")
    print("-d /--directory           Directory to multiple yuv files.")
    print("-r /--resolution          Set the resolution [w]x[h]. Default is 1920x1080.")
    print("-b /--block_size          Set the block size. Default is 32 and must be a multiple of 4.")
    print("-f /--frames              Maximum number of frames for features extraction. 0 for all frames.")
    print("-g /--gopsize             The number of frames that is processed simultaneously. Default is 32.")
    print("-p /--pix_fmt             yuv format. Default is yuv420. [yuv420, yuv444] ")
    print("-s /--sample_rate         Frame subsampling. Default is 1 ")
    print("-c /--csv                 Name of csv to write features. Default is ./csv/test.csv")
    print("-bi/--block_info          Write block level features into a csv. Default is disabled")
    print("-pi/--plot_info           Plot per frame features. Default is disabled")
    print("-dp/--dpi                 Image quality of the saved output. Default is 100.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', '--input', type=str, default='test.yuv')
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('-m', '--method', type=str, default='EVCA')
    parser.add_argument('-t', '--transform', type=str, default='DCT')
    parser.add_argument('-r', '--resolution', type=str, default='1920x1080')
    parser.add_argument('-b', '--block_size', type=int, default=32)
    parser.add_argument('-f', '--frames', type=int, default=0)
    parser.add_argument('-c', '--csv', type=str, default='./csv/test.csv')
    parser.add_argument('-g', '--gopsize', type=int, default=32)
    parser.add_argument('-p', '--pix_fmt', type=str, default='yuv420')
    parser.add_argument('-s', '--sample_rate', type=int, default=1)
    parser.add_argument('-bi', '--block_info', type=int, default=0)
    parser.add_argument('-pi', '--plot_info', type=int, default=0)
    parser.add_argument('-dp', '--dpi', type=int, default=100)
    parser.add_argument('-fi', '--filter', type=str, default='sobel')
    return parser


def get_parser_arguments(argv: List[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if path.is_absolute():
        return path
    return (PACKAGE_ROOT / path).resolve()


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if '-h' in argv or '--help' in argv:
        print_custom_help()
        return

    args = get_parser_arguments(argv)
    args.csv = str(resolve_csv_path(args.csv))

    import torch

    from .libs.EVCA import EVCA
    from .libs.SITI import SITI

    print("EVCA: Enhanced Video Complexity Analyzer v1.0.")

    input_list, success = check_existence(args)
    if not success:
        print('Input file or directory is not specified.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t1 = time.time()
    if args.method == 'EVCA':
        EVCA(args, input_list, device)
    elif args.method == 'VCA':
        EVCA(args, input_list, device)
    elif args.method == 'SITI':
        SITI(args, input_list, device)
    else:
        print('Unsupported method.')
        return
    t2 = time.time()
    print(f'Feature extraction completed in {t2 - t1:.2f} seconds.')


if __name__ == "__main__":
    main()
