from scipy.spatial import distance

# Each measure should take two lists of histograms
# and return a final score (as a single number)


def euclidean(ls, rs):
    return sum(distance.euclidean(l, r) for l, r in zip(ls, rs))
