import numpy as np
import pandas as pd
from scipy.stats import entropy


class CN2:
    def __init__(self, max_star_size, min_significance):
        self.max_star_size = max_star_size
        self.min_significance = min_significance
        self.E = None
        self.rule_list = []

    def fit(self, X, y):
        self.E = X.copy()
        self.E['class'] = y

        self._init_selectors()

        best_complex = self._find_best_complex()
        while best_complex is not None and not self.E.empty:
            if best_complex is not None:
                E_prime = self.E.loc[self.E[best_complex.keys()].isin(best_complex.values()).all(axis=1), :]
                C = E_prime['class'].value_counts().idxmax()
                self.E.drop(E_prime.index, inplace=True)
                self.rule_list.append(best_complex)

            best_complex = self._find_best_complex()

    def _init_selectors(self):
        self.selectors = []
        for attr in self.E.columns[:-1]:
            for value in self.E[attr].unique():
                self.selectors.append({attr: value})

    def _find_best_complex(self):
        best_cpx = None
        best_entropy = np.inf
        best_significance = np.NINF
        star = []
        while True:
            new_star = self._specialize_star(star)
            entropy_list = []
            significance_list = []

            for cpx in new_star:
                E_prime = self.E.loc[self.E[cpx.keys()].isin(cpx.values()).all(axis=1), :]

                if not E_prime.empty:
                    pdf = E_prime['class'].value_counts(sort=False, normalize=True)

                    cpx_entropy = entropy(np.array(pdf))
                    class_pdf = np.array(
                        self.E['class'].loc[self.E['class'].isin(pdf.keys())].value_counts(sort=False,
                                                                                           normalize=True))
                    cpx_significance = 1 - 2 * np.sum(pdf * np.log(pdf / class_pdf))
                    entropy_list.append(cpx_entropy)
                    significance_list.append(cpx_significance)

                    if cpx_entropy < best_entropy and cpx_significance >= best_significance:
                        best_cpx = cpx
                        best_entropy = cpx_entropy
                        best_significance = cpx_significance
                else:
                    entropy_list.append(np.inf)
                    significance_list.append(np.NINF)

            aux_df = pd.DataFrame({'cpx': new_star, 'entropy': entropy_list, 'significance': significance_list})
            aux_df = aux_df.sort_values(by=['significance', 'entropy'], ascending=[False, True]).iloc[
                     :self.max_star_size]

            star = aux_df.cpx.to_list()

            # If star is empty exit the loop
            if not star:
                break

        return best_cpx

    def _specialize_star(self, star):
        if star:
            new_star = []
            for cpx in star:
                for selector in self.selectors:
                    selector_attr = list(selector)[0]
                    if selector_attr not in cpx.keys():
                        new_cpx = cpx.copy()
                        new_cpx[selector_attr] = selector[selector_attr]
                        if new_cpx not in new_star:
                            new_star.append(new_cpx)
                return new_star
        return self.selectors.copy()
