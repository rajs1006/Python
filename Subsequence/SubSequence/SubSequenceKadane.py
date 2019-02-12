import sys
from SubSequence.SubSequence import SubSequence as SubSeq


##Author : Sourabh Raj : Does not work for all positive numbers
class SubSequenceKadane(SubSeq):

    def __init__(self, input_seq, max_len: int, operation='Values'):
        SubSeq.__init__(self, input_seq, max_len, operation)

    def max_sum_subsequence(self):

        if self.input_seq:

            max_sum = -sys.maxsize - 1
            current_list = []
            max_current = 0
            start, end, s = 0, 0, 0

            for i in range(0, self.length):

                current_list.append(self.input_seq[i])
                max_current += self.input_seq[i]

                if max_sum < max_current:
                    max_sum = max_current
                    start = s
                    end = i

                if max_current < 0  or len(current_list) >= self.max_len:
                    current_list.clear()
                    max_current = 0
                    s = i + 1

            max_sequence = [self.input_seq[i] for i in range(start, end + 1)]
            return max_sum, max_sequence
