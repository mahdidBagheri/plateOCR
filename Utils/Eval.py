from textdistance import levenshtein as lev
import numpy as np
import torch
import re
import math
import os

from Utils.Utils import shrink_text, extend_text


def similarity(word1, word2):
    return lev.normalized_distance(word1, word2)

class Eval:
    def _blanks(self, max_vals,  max_indices):
        def get_ind(indices):
            result = []
            for i in range(len(indices)):
                if indices[i] != 0:
                    result.append(i)
            return result
        non_blank = list(map(get_ind, max_indices))
        scores = []

        for i, sub_list in enumerate(non_blank):
            sub_val = []
            if sub_list:
                for item in sub_list:
                    sub_val.append(max_vals[i][item])
            score = np.exp(np.sum(sub_val))
            if math.isnan(score):
                score = 0.0
            scores.append(score)
        return scores

    def char_accuracy(self, predictions, labels):
        words, truths = shrink_text(predictions), shrink_text(labels)
        sum_edit_dists = lev.distance(words, truths)
        sum_gt_lengths = sum(map(len, predictions))
        fraction = 0
        if sum_gt_lengths != 0:
            fraction = sum_edit_dists / sum_gt_lengths

        percent = fraction * 100
        if 100.0 - percent < 0:
            return 0.0
        else:
            return 100.0 - percent

    def format_target(self, target, target_sizes):
        target_ = []
        start = 0
        for size_ in target_sizes:
            target_.append(target[start:start + size_])
            start += size_
        return target_