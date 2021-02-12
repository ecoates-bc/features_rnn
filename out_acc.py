import re


def get_file(path):
    file = [line for line in open(path, 'r').readlines()]
    outputs = []
    sys = ''
    gold = ''
    for line in file:
        if re.search('^SYS:', line):
            sys = re.sub('^SYS: ', '', line)
        elif re.search('^GOLD:', line):
            gold = re.sub('^GOLD: ', '', line)
            outputs.append((sys, gold))

    return outputs


def get_acc(file):
    return sum([1 if x == y else 0 for x, y in file]) / len(file)


def get_inaccuracies(file):
    inacc = []
    for x, y in file:
        if x != y:
            inacc.append('SYS: ' + x + '\nGOLD: ' + y)
    return inacc

#
# orth = get_file('results/orth_test_withatt.txt')
# print(get_acc(orth))
# orth_inacc = get_inaccuracies(orth)
# for line in orth_inacc:
#     print(line)

# ipa = get_file('results/ipa_test_withatt.txt')
# print(get_acc(ipa))
# ipa_inacc = get_inaccuracies(ipa)
# for line in ipa_inacc:
#     print(line)

# feat = get_file('results/features_test_withatt.txt')
# print(get_acc(feat))
# feat_inacc = get_inaccuracies(feat)
# for line in feat_inacc:
#     print(line)

comb = get_file('combined_epitran_results_2.txt')
print(get_acc(comb))
feat_inacc = get_inaccuracies(comb)
for line in feat_inacc:
    print(line)