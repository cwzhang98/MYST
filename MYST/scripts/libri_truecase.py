import truecase
from tqdm import tqdm
import argparse

def main(args):
    with open(args.txt_path, 'r') as f:
        lines = f.readlines()
        with open(args.txt_path + '.true', 'w') as f2:
            for i, line in tqdm(enumerate(lines)):
                f2.write(truecase.get_true_case(line.strip()) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt-path", help="raw text file path", type=str, required=True)
    args = parser.parse_args()
    main(args)