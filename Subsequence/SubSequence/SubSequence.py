class SubSequence(object):

    def __init__(self, input_seq, max_len: int, operation='Values'):
	# Data manipulation
        if operation.lower() == 'differences':
            self.input_seq = SubSequence.difference_of_neighbour(self, input_seq)
            # Length based on incoming restricted length.
            self.max_len = int(max_len) - 1
        else:
            self.input_seq = input_seq
            # Length based on incoming restricted length.
            self.max_len = int(max_len)

        self.length = len(input_seq) - 1
        self.operation = operation

    def difference_of_neighbour(self, input_sub_seq):
        seq = self.input_seq if not input_sub_seq else input_sub_seq
        if seq:
            length_n = len(seq)
            output_seq = []
            for i in range(0, length_n - 1):
                output_seq.append(abs(seq[i] - seq[i + 1]))
            return output_seq

    def max_sum_subsequence(self, algorithm: str = 'Merantix'):
        pass
