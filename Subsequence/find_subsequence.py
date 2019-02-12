import argparse
from SubSequence.FileReader import FileReader
from SubSequence.SubSequenceMerantix import SubSequenceMerantix
from SubSequence.SubSequenceKadane import SubSequenceKadane


def main():
    # Fetch input parameters.
    args = return_args()

    # Read input file.
    input_seq = FileReader(args.input_file).read_file()

    # Call for subsequence based n the algorithm type.
    if str(args.algo).lower() == 'kadane':
        print("Running Kadane")
        max_sum, seq = SubSequenceKadane(input_seq, args.restricted_length, args.operation).max_sum_subsequence()
    else:
        print("Running Merantix")
        max_sum, seq = SubSequenceMerantix(input_seq, args.restricted_length, args.operation).max_sum_subsequence()

    print("Sum : ", max_sum, '\nSeq : ', seq)


def return_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('restricted_length')
    parser.add_argument('operation')
    parser.add_argument('algo', nargs='?')

    return parser.parse_args()

## Main execution.
main()
