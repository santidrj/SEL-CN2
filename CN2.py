class CN2:
    def __init__(self, max_star_size, min_significance):
        self.max_star_size = max_star_size
        self.min_significance = min_significance
        self.E = None


    def fit(self, X, y):
        self.E = X.copy()
        self.E['class'] = y
        
        self._init_selectors()

    def _init_selectors(self):
        self.selectors = []
        for attr in self.E.columns[:-1]:
            for value in self.E[attr].unique():
                self.selectors.append({attr: value})
