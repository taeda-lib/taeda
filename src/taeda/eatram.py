
import random

def rameat(repeat=10, max_num=10, min_num=1, use_verbose=False):

    random_list = []

    for _ in range(repeat):
        random_list.append(random.randint(min_num, max_num))

    if use_verbose:
        print(random_list)

    return random_list