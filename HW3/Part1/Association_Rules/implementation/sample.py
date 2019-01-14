import numpy as np
from itertools import combinations, groupby
from collections import Counter

# Sample data
orders = np.array([[1, 'apple'], [1, 'egg'], [1, 'milk'], [2, 'egg'], [2, 'milk']], dtype=object)


# Generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    # For each order, generate a list of items in that order
    for order_id, order_object in groupby(orders, lambda x: x[0]):
        item_list = [item[1] for item in order_object]

        # For each item list, generate item pairs, one at a time
        for item_pair in combinations(item_list, 2):
            yield item_pair

        # Counter iterates through the item pairs returned by our generator and keeps a tally of their occurrence


Counter(get_item_pairs(orders))