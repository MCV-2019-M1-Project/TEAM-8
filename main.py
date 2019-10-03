import numpy as np
from tqdm.auto import tqdm


def calc_similarities(measure, db, qs, show_progress=False):
    def compute_one(hist):
        return [measure(hist, db_hist) for db_hist in db]
    generator = tqdm(qs) if show_progress else qs
    return np.array([compute_one(hist) for hist in generator])


def get_tops(similarities, k):
    tops = similarities.argsort(axis=1)[:, :k]
    return tops
