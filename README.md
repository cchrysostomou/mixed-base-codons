# mixed-base-codons
Implementation of genetic algorithm that determines the ratio of A/C/T/G at each codon position required to achieve a desired amino acid distribution

Implements a genetic algorithm desribed by [Craig, R. et al](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2811015/#B11). The purpose of the algorithm is to help users design oligos such that each position contains a mixed ratio at a given position. The allowed resolution of each base (i.e. 1% each base or 5% or 10%) in a position can be defined by the user. The algorithm attempts to optimize a set of mixed codon distributions that best matches the desired amino acid distribution.

# Usage

Standard usage of the class

```
from mixed_base_codon_design.codon_design_genetic_algo import CodonDesigner, packbit_custom
import numpy array as np

# define a desired amino acid distribution
aa_dist = np.array([[0.0125, 0.05  , 0.0125, 0.0125, 0.0125, 0.0125, 0.1   , 0.0125,
        0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.09  , 0.25  ,
        0.31  , 0.0125, 0.0125, 0.0125, 0.0125]])

# define the parameters for running the algorithm
v1 = CodonDesigner(
    pop_size=100, # create population of 300 variants
    forced_bitsize=8, # set encoding to be 8bits
    elite_fraction=0.1, # keep top 10% of solutions in each iteration
    elite_mut_density=0.5, elite_mut_rate=0.1, mut_density=0.5, mut_rate=0.5, cross_rate=0.8, double_cross_rate=0.1, numeric_percent_resolution=1, thresh=1e-8, # desired threshold cutoff
    max_time=300 # stop after 300 seconds
)
# run the algorithm using desired distribution
v1.run_genetic_algorithm(aa_dst, weights=None, round_aa_freq_precision=5)

# re update with a new objective function
v1._reset_hyperparameter('obj_fxn', 'cosine')
v1.run_genetic_algorithm(aa_dst, weights=None, round_aa_freq_precision=5)

# only allow for degenerate base distributions (i.e. N/S/W)
v1._reset_encoding('degen')
v1.run_genetic_algorithm(aa_dst, weights=None, round_aa_freq_precision=5)
```

