import sys
from SubSequence.SubSequence import SubSequence as SubSeq

##Author : Sourabh Raj
class SubSequenceMerantix(SubSeq):

    def __init__(self, input_seq, max_len: int, operation):
        SubSeq.__init__(self, input_seq, max_len, operation)

    def max_sum_subsequence(self):

        if self.input_seq:
            max_sum = -sys.maxsize - 1
            start = 0
            end = 0

            for i in range(0, self.length):
                sum_j = 0
                for j in range(i, self.length):

                    # Breaks if max length crossed.
                    if j >= i + self.max_len:
                        break

                    sum_j += self.input_seq[j]
                    if sum_j > max_sum:
                        start = i
                        end = j
                        max_sum = sum_j

            max_sequence = [self.input_seq[i] for i in range(start, end + 1)]
            return max_sum, max_sequence
