import re
import numpy as np


def format_and_reverse_number(n):
    n1 = f'{n:,}' + ('-' if n < 0 else '')
    ss = n1.split(',')
    res = ''
    for i, s in enumerate(ss[::-1]):
        if i > 0:
            while i % 3 == 0:
                res = '`' + res
                i /= 3
            res = '`' + res
        res = s + res
    return res[::-1]

def generate_input(a, b, input_only=False):
    c = a + b
    a_rev = format_and_reverse_number(a)
    b_rev = format_and_reverse_number(b)
    c_rev = format_and_reverse_number(c)
    example = ''
    prompt = f'Add A={a} and B={b}. ' \
             f'Reversing the order of digits of A={a}, ' \
             f'we obtain'
    answer = f' A_={a_rev}. ' \
             f'Reversing the order of digits of B={b}, we obtain B_={b_rev}. ' \
             f'Column addition procedure for digit-reversed numbers ' \
             f'A_={a_rev} and B_={b_rev} will give C_={c_rev}. ' \
             f'Reversing the digits of C_={c_rev} gives the answer C={c}.'
    if input_only:
        return example + prompt
    else:
        return example + prompt + answer


def response_to_answer(response):
    res = re.findall(r'C=-?[\d`]+', response)
    if len(res) == 0:
        return {'status': '404', 'result': None}
    prediction = int(res[0][2:].replace('`', ''))
    if len(res) == 1:
        return {'status': '200', 'result': prediction}
    else:
        return {'status': '500', 'result': prediction, 'list': [int(x[2:]) for x in res]}


def generate_pairs(size=1000, seed=42, lengths_distribution=None):
    np.random.seed(seed)

    if lengths_distribution is None:
        lengths_distribution = np.array([1] + [10 * n for n in range(1, 11)] + [1000 // n for n in range(11, 51)])
        lengths_distribution = lengths_distribution / lengths_distribution.sum()
    lengths = np.random.choice(np.arange(1, 1 + len(lengths_distribution)), p=lengths_distribution, size=[2, size])

    mask = np.random.choice([True, False], p=[2/3, 1/3], size=size)
    lengths[1, mask] = lengths[0, mask]

    digits = np.random.randint(0, 10, size=lengths.sum())
    digits = ''.join([str(d) for d in digits])

    limits = [0] + list(np.cumsum(lengths.flatten()))
    left, right = limits[:-1], limits[1:]

    numbers = [int(digits[l:r]) for l, r in zip(left, right)]
    res = np.array(numbers, dtype=object).reshape([2, size])

    res *= np.random.choice([-1, 1], size=res.shape)
    return res


class AdditionDataset:
    def __init__(self, tokenizer, gen_input=None, gen_pairs=None, resp2ans=None, size=1000, seed=42):
        self.tokenizer = tokenizer
        self.gen_input = gen_input if gen_input is not None else generate_input
        self.gen_pairs = gen_pairs if gen_pairs is not None else generate_pairs
        self.response_to_answer = resp2ans if resp2ans is not None else response_to_answer

        self.pairs = self.gen_pairs(size=size, seed=seed)
        self.texts = [self.gen_input(a, b) for (a, b) in zip(*self.pairs)]
