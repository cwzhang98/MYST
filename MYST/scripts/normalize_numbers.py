import argparse
from tqdm import tqdm
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-d", required=True, type=str)
    args = parser.parse_args()
    num_out_of_range_numbers = 0
    with open(args.root, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            line = line.strip()
            numbers = re.findall(r"\d+", line)
            for number in numbers:
                if len(number) >= 33:
                    num_out_of_range_numbers += 1
                    print(number, len(number), i)
                    print(line)


if __name__ == '__main__':
    main()
