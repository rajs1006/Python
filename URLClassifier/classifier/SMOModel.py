class SMOModel:

    def __init__(self, X, y, C, kernel, alphas, b, w):
        self.X = X
        self.y = y
        self.C = C
        self.kernel = kernel
        self.alphas = alphas
        self.b = b
        self.w = w
