import numpy as np


class FileReader:

    def __init__(self, file):
        self.file = file

    def read_file(self):
        if self:
            try:
                with open(self.file, "r", encoding="ISO-8859-1") as f:
                    movie_list = [' '.join(np.array(l.split())[1:]) for l in f.readlines()]
                return movie_list
            except IOError:
                pass
        else:
            raise ValueError("File name no specified. ")
