import warnings

import numpy as np
import pandas as pd
from scipy.stats import entropy


class CN2:
    def __init__(self, max_star_size=5, min_significance=0.5):
        self.max_star_size = max_star_size
        self.min_significance = min_significance
        self.E = None
        self.rule_list = []

    def fit(self, X, y, n_bins=4, fixed_bin_size=False):
        self.E = X.copy()
        self.E['class'] = y.copy()

        self._discretize(n_bins, fixed_bin_size)

        # Replace missing values with the most common value of the attribute.
        for c in self.E.columns:
            self.E[c] = self.E[c].replace('?', self.E[c].value_counts().idxmax())

        self._init_selectors()

        default_class = self.E['class'].value_counts().idxmax()

        n = len(self.E.index)
        best_complex, best_cpx_covered_examples, best_cpx_most_common_class, best_cpx_precision = self._find_best_complex()
        while best_complex is not None and not self.E.empty:
            if best_complex is not None:
                best_cpx_coverage = len(best_cpx_covered_examples) / n
                self.E.drop(best_cpx_covered_examples, inplace=True)
                self.rule_list.append((best_complex, best_cpx_most_common_class, best_cpx_precision, best_cpx_coverage))

            best_complex, best_cpx_covered_examples, best_cpx_most_common_class, best_cpx_precision = self._find_best_complex()

        self.rule_list.append((None, default_class, 0, 0))

    def _init_selectors(self):
        self.selectors = []
        for attr in self.E.columns[:-1]:
            for value in self.E[attr].unique():
                self.selectors.append({attr: value})

    def _find_best_complex(self):
        best_cpx = None
        best_entropy = np.inf
        best_significance = np.NINF
        best_cpx_covered_examples = None
        best_cpx_most_common_class = None
        best_cpx_precision = 0
        star = []
        while True:
            new_star = self._specialize_star(star)
            entropy_list = []
            significance_list = []

            for cpx in new_star:
                E_prime = self.E.loc[self.E[cpx.keys()].isin(cpx.values()).all(axis=1), :]
                covered_examples = E_prime.index

                if not E_prime.empty:
                    covered_prob_distribution = E_prime['class'].value_counts(sort=False, normalize=True)

                    cpx_entropy = entropy(np.array(covered_prob_distribution))
                    # Todo: check how to compute class probability distribution
                    class_prob_distribution = self.E['class'].loc[
                        self.E['class'].isin(covered_prob_distribution.keys())].value_counts(sort=False, normalize=True)
                    # class_prob_distribution = self.E['class'].value_counts(sort=False, normalize=True).loc[
                    #     covered_prob_distribution.keys()]

                    cpx_significance = 2 * np.sum(
                        covered_prob_distribution * np.log(covered_prob_distribution / class_prob_distribution))
                    entropy_list.append(cpx_entropy)
                    significance_list.append(cpx_significance)

                    if cpx_significance >= self.min_significance:
                        if cpx_entropy < best_entropy and cpx_significance >= best_significance:
                            best_cpx = cpx
                            best_entropy = cpx_entropy
                            best_significance = cpx_significance
                            best_cpx_covered_examples = covered_examples
                            best_cpx_most_common_class = E_prime['class'].value_counts().idxmax()
                            best_cpx_precision = covered_prob_distribution.sort_values(ascending=False)[0]
                else:
                    entropy_list.append(np.inf)
                    significance_list.append(np.NINF)

            aux_df = pd.DataFrame({'cpx': new_star, 'entropy': entropy_list, 'significance': significance_list})
            aux_df = aux_df.sort_values(by=['significance', 'entropy'], ascending=[False, True]).iloc[
                     :self.max_star_size]

            star = aux_df.cpx.to_list()

            # If star is empty exit the loop
            if not star or (best_entropy == 0 and best_significance >= 1):
                break

        return best_cpx, best_cpx_covered_examples, best_cpx_most_common_class, best_cpx_precision

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

    def _discretize(self, n_bins=4, fixed_bin_size=False):
        int_cols = self.E.select_dtypes(include=np.int).columns
        float_cols = self.E.select_dtypes(include=np.float).columns
        self.discretize_columns(int_cols, n_bins, fixed_bin_size)
        self.discretize_columns(float_cols, n_bins, fixed_bin_size, 2)

    def discretize_columns(self, columns, n_bins, fixed_bin_size, precision=0):
        for c in columns:
            if len(self.E[c].value_counts()) < n_bins:
                warnings.warn(
                    f"Column {c} only has {len(self.E[c].value_counts())} unique values and can not be discretized" +
                    f" using {n_bins} bins. The number of bins has been reduced to fit the column.")

                n_bins = len(self.E[c].value_counts())

            if not fixed_bin_size:
                self.E[c] = pd.cut(self.E[c], n_bins, precision=precision, duplicates='drop')
            else:
                self.E[c] = pd.qcut(self.E[c], n_bins, precision=precision, duplicates='drop')
