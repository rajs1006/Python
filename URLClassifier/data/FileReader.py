class FileReader:

    def __init__(self, file):
        self.file = file

    def read_file(self):
        if self:
            try:
                with open(self.file, "r") as f:
                    input_seq = [int(n) for n in f.read().split()]

                return input_seq
            except IOError:
                pass
        else:
            raise ValueError("File name no specified. ")
