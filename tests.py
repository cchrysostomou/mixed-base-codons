import numpy as np
from .codon_design_genetic_algo import CodonDesigner

def convert_bits_to_array_slowly(bits):
    seqs = []
    for __, i in enumerate(bits):    
        for _, r in enumerate(np.arange(0, i.shape[0], 8)):
            val = i[r:r+8]
            seqs.append((np.power(2, [7, 6, 5, 4, 3, 2, 1, 0]) * val).sum())
    new_seqs = []
    for __, i in enumerate(np.arange(0, len(seqs), 12)):
        new_seqs.append([])
        subset = seqs[i:i+12]
        for ___, j in enumerate(np.arange(0, len(subset), 4)):
            new_seqs[__].append([])
            acgt = subset[j:j+4]
            new_seqs[__][___].extend(acgt)
    return np.array(new_seqs).swapaxes(1, 2)


def check_variants():
    x1 = CodonDesigner.generate_bitcount(8, 10, 4)
    assert x1.shape[1]/(4 * 3) == 8, x1
    assert x1.shape[0] == 10, x1.shape

    x1_num_test = convert_bits_to_array_slowly(x1)
    x1_num = CodonDesigner.bit_arr_to_num(x1, 8, 4)

    assert (x1_num_test == x1_num).all()


