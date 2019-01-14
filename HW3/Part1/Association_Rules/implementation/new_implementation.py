import argparse
from itertools import chain, combinations


# Helper Functions
def joinset(itemset, k):
    return set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == k])


def subsets(itemset):
    return chain(*[combinations(itemset, i + 1) for i, a in enumerate(itemset)])


def make_itemset(data):
    itemset = set()
    transaction_list = list()
    for row in data:
        transaction_list.append(set(row))
        for item in row:
            if item:
                itemset.add(set([item]))
    return itemset, transaction_list


# Association rules functions

def support(transaction_list, itemset, min_support=0):
    transaction_list_length = len(transaction_list)

    new_length = [
        (item, float(sum(1 for row in transaction_list if item.issubset(row))) / transaction_list_length)
        for item in itemset
    ]
    return dict([(item, support) for item, support in new_length if support >= min_support])


def frequent_itemset(transaction_list, c_itemset, min_support):
    t = 1

    freq_itemset = dict()

    while True:

        if t > 1:
            c_itemset = joinset(l_itemset, t)
        l_itemset = support(transaction_list, c_itemset, min_support)
        if not l_itemset:
            break
        freq_itemset.update(l_itemset)
        t += 1

    return freq_itemset


def Apriori(data, min_support, min_confidence):
    #  first itemset and transactions
    itemset, transaction_list = make_itemset(data)

    # Get the frequent itemset
    f_itemset = frequent_itemset(transaction_list, itemset, min_support)

    # Association rules
    rules = list()
    for item, support in f_itemset.items():
        if len(item) > 1:
            for x in subsets(item):
                y = item.difference(x)
                if y:
                    x = set(x)
                    xy = x | y

                    confidence = float(f_itemset[xy]) / f_itemset[x]
                    if confidence >= min_confidence:
                        rules.append((x, y, confidence))
    return rules, f_itemset


def result(rules, f_itemset):
    print('lets get started:')
    print('#Frequent Itemset :')
    for item, support in f_itemset.items():
        print('support{}: {}'.format(tuple(item), round(support, 3)))

    print('#rules :')
    for A, B, confidence in rules:
        print(' confidence{} => {} : {}'.format(tuple(A), tuple(B), round(confidence, 3)))


def data_A():
    Set_A = list()
    Set_A.append('ABDG')
    Set_A.append('BDE')
    Set_A.append('ABCEF')
    Set_A.append('BDEG')
    Set_A.append('ABCEF')
    Set_A.append('BEG')
    Set_A.append('ACDE')
    Set_A.append('BE')
    Set_A.append('ABEF')
    Set_A.append('ACDE')
    return Set_A


def data_B():
    Set_B = list()
    Set_B.append('ACD')
    Set_B.append('BCE')
    Set_B.append('ABCE')
    Set_B.append('BE')
    return Set_B


def main():
    min_support_1 = 0.4

    min_confidence_1 = 0.5

    data1 = data_A()
    print("set1:")
    rules_1, itemset_1 = Apriori(data1, min_support_1, min_confidence_1)
    print(result(rules_1, itemset_1))

    min_support_2 = 0.2
    min_confidence_2 = 0

    print("set2:")
    data_2 = data_B()
    rules_2, itemset_2 = Apriori(data_2, min_support_2, min_confidence_2)
    print(result(rules_2, itemset_2))


main()
