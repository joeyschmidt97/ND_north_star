
def detect_dimensionality(lst):
    if isinstance(lst, list):
        return 1 + detect_dimensionality(lst[0])
    else:
        return 0
