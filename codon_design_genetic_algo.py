from Bio.Seq import Seq
from itertools import combinations, product
import numpy as np
import time
import warnings
import signal
from .objective_functions import obj_fxn_map

# used for performing the cartesian probabability of amino acid usage
cartesian_coordinates = np.array([x1 for x1 in product([0,1,2,3],[0,1,2,3],[0,1,2,3])])
num_codon_pos=3

default_DNA = list('ACGT')

# get AA alphabet from codons
default_AA_CODON_ALPHABET = [
    str(Seq(''.join(x)).translate())
    for x in product(list(default_DNA), list(default_DNA), list(default_DNA))
]

default_unique_aa = sorted(set(default_AA_CODON_ALPHABET))
default_codon_map = [
    default_unique_aa.index(c) for c in default_AA_CODON_ALPHABET
]

print(default_unique_aa)

def packbit_custom(bitmatrix, bitsize, b):
    """
    Performs same function as np.packbits except does not require the uint8 datatype (more than 8 bits allowed)
    """
    if b is None:
        b = np.power(2, np.arange(bitsize-1, -1, -1))
    
    # NP PACK BITS WILL NOT WORK if bit size is > 8        
    return (b.T * bitmatrix.reshape(1, -1, bitsize)).sum(axis=2).reshape(bitmatrix.shape[0], 3, 4).swapaxes(1,2)


class CodonDesigner(object):
    @staticmethod
    def generate_bitcount(bitsize, num_seqs, alphabet_size):
        """
        Generates a distribution counts for each letter within ACGT (or alphabet length) at a specific position
        
        Codon is represented as (4 ways to distribute letters across three positions): 
            p1 p2 p3
            #A  #A  #A
            #C  #C  #C
            #G  #G  #G
            #T  #T  #T
            
        This will randomly generate number between 0 -> 255 (bitsize = 8) for each letter at each codon position. What will be returned is the following dimensionality:
            Axis 0 (size = num_seqs) = each row represents a unique sequence
            Axis 1 (size = bitsize * alphabet_size * num_codon_pos) = distribution of each letter at each codon position
                    #A@p1     #C@p1    #G@p1    #T@p1   ..... #A@p3   #C@p3   #G@p3    #T@p3 
            seq1   00110001  00111011 11110001 11111001     11111101 00110011 00110001 00110101
            seq2
            .
            .
            .
            seq N

        """
        return np.random.choice(np.array([False, True], dtype=np.bool), bitsize*alphabet_size*num_seqs*num_codon_pos).reshape(bitsize*alphabet_size*num_codon_pos, -1).T

    @staticmethod
    def bit_rep_to_num(bitmatrix, bitsize, alphabet_size):
        """
        Converts the bitmatrix where each bit represents whether a letter in the alphabet is present (1 or 0) X times where X = bitsize
        
        The following dimensionality is returned:
            
            Axis 0 (size = num_seqs/shape[0] of bitmatrix)
            Axis 1 (size = len(alphabet) = > A/C/G/T)
            Axis 2 (size = # of codon positions) => Codon position 1, 2, or 3
        """
        return bitmatrix.reshape(-1, bitsize).sum(axis=1).reshape(bitmatrix.shape[0], num_codon_pos, alphabet_size).swapaxes(1, 2)

    @staticmethod
    def bit_arr_to_num(bitmatrix, bitsize, alphabet_size, b=None, zeroPad=None):
        """
            Converts the bitmatrix created in 'generate_bitcount' into numerical form where now each letter's weight/frequency is represented at each codon position. 
            
            The following dimensionality is returned:
            
            Axis 0 (size = num_seqs/shape[0] of bitmatrix)
            Axis 1 (size = len(alphabet) = > A/C/G/T)
            Axis 2 (size = # of codon positions) => Codon position 1, 2, or 3
        """
        # return (base.T * bitmatrix.reshape(bitmatrix.shape[0], -1, bitsize)).sum(axis=2)
        if bitsize > 8:
            res = packbit_custom(bitmatrix, bitsize, b)
            return res
        elif bitsize < 8:     
            # packbit function does not like non 8bit encoded  (it left pads 0 rather than right pads 0 which we need)       
            padding = 8 - bitsize
            if zeroPad is None:
                rows = int(bitmatrix.shape[1] / bitsize) * bitmatrix.shape[0]
                zeroPad = np.zeros((rows, padding), np.uint8)                
            res = np.packbits(
                np.hstack([
                    zeroPad[:, :padding],
                    bitmatrix.reshape(-1, bitsize)
                ]).reshape(1, -1, 8),
                axis=2
            ).reshape(
                bitmatrix.shape[0], num_codon_pos, alphabet_size
            ).swapaxes(1,2)
        else:        
            res = np.packbits(
                bitmatrix.reshape(1, -1, bitsize),
                axis=2
            ).reshape(
                bitmatrix.shape[0], num_codon_pos, alphabet_size
            ).swapaxes(1,2)                
        return res

    @staticmethod
    def renormalize_data(nt_counts, resolution):
        """
        Renormalizes the distribution such that the frequencies of each nucleotide are within a desired resolution (i.e. 1% interval or 2%)

        Resolution:
            defines how to report the frequency (every 1%, 2%, 5%...0.1%..etc)
        """
        return (np.round(
            nt_counts / ((np.maximum(1, nt_counts.sum(axis=1)) / (100/resolution))[:, np.newaxis])
        ) * resolution)

        # return 100 * (np.round(nt_counts / (nt_counts.sum(axis=1))[:, np.newaxis], 2))

    @staticmethod
    def get_codon_freq(codon_usage):
         return (codon_usage / ((np.maximum(1, codon_usage.sum(axis=1)))[:, np.newaxis])) * 100

    @staticmethod
    def nt_to_aa_dist(codon_usage, unique_aa_set=None, codon_aa_map=None, round_freq_to_decimal=None):
        """
            Given a distribution of nucleotides in each codon position, return the distribution of amino acids
        """ 
        
        unique_aa_set = unique_aa_set or default_unique_aa
        codon_aa_map = codon_aa_map or default_codon_map

        # make sure its in frequency
        codon_freq = CodonDesigner.get_codon_freq(codon_usage)

        #Now multiply all combinations of ACTG,ACTG,ACTG at each position (generate 64 possible combinations at each position)
        codon_probs = np.prod(
            np.dstack(
                [
                    codon_freq[:, cartesian_coordinates[:,0], 0],
                    codon_freq[:, cartesian_coordinates[:,1], 1], 
                    codon_freq[:, cartesian_coordinates[:,2], 2]
                ]
            ), axis=2
        )

        # map probability at each codon to a respective cumulative probability for an amino acid
        new_arr =  np.array([0] * len(unique_aa_set) * codon_probs.shape[0], dtype=np.float).reshape(codon_probs.shape[0], -1)
        for ij in range(codon_probs.shape[1]):
            new_arr[:, codon_aa_map[ij]] += codon_probs[:, ij]
        
        aa_freq = new_arr / (np.maximum(new_arr.sum(axis=1), 1)[:,np.newaxis])
        if round_freq_to_decimal is None:
            return aa_freq
        else:
            return np.round(aa_freq, round_freq_to_decimal)

    def __init__(self, encoding='numeric', numeric_percent_resolution=1, forced_bitsize=None, bits_are_ratios=False, *args, **kwargs):
        """
            Initalize the codon design parameters

            bits_are_ratios: allows you to determine how to weight the ratio of each base to one another. For example if bitsize is 1 and bits are ratios is present then we can represent distributions as:
                A = 1, C = 1, G = 1, T = 0 OR A = 1, C=0, G=0,T=0. If bit size were 4 then we could achieve ratios of 4:1 such that A=4,C=0,G=0,T=1, OR A=4,C=4,G=4,T=4. This is really only recommended when we want to allow degenerate base code using a bitsize of 1            
        """

        self._reset_encoding(encoding, numeric_percent_resolution, forced_bitsize, bits_are_ratios)
        self.DNA_ALPHABET = list('ACGT')
        self.alphabet_size = len(self.DNA_ALPHABET)
        self.bad_distribution_fitness = float('Inf')
        self.round_aa_freq_precision = None
        # get AA alphabet from codons
        self.AA_CODON_ALPHABET = [
            str(Seq(''.join(x)).translate()) 
            for x in product(list(self.DNA_ALPHABET), list(self.DNA_ALPHABET), list(self.DNA_ALPHABET))
        ]

        # collpase degenerate codons into unique AA
        self.AA_ALPHABET = sorted(set(self.AA_CODON_ALPHABET))
        # map which amino acid is encoded by each codon
        self.codon_index = [self.AA_ALPHABET.index(c) for c in self.AA_CODON_ALPHABET]
        self._reset_all_hyperparameters(*args, **kwargs)

    def _reset_all_hyperparameters(self, pop_size=100, obj_fxn='cosine', mut_density=0.3, mut_rate=0.2, cross_rate=0.9, double_cross_rate=0.02, thresh=1e-3, elite_fraction=0.1, max_time=None, max_iter=None, elite_mut_density=0, elite_mut_rate=0.05):
        self.pop_size = pop_size
        self.obj_fxn = obj_fxn_map[obj_fxn]
        # self.round_fxn = obj_fxn_map[round_fxn_name]
        self.mut_density = mut_density
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.double_cross_rate = double_cross_rate
        self.thresh = thresh
        self.elite_fraction = elite_fraction
        self.max_time = max_time or float('Inf')
        self.max_iter = max_iter or float('Inf')
        self.elite_mut_density = elite_mut_density
        self.elite_mut_rate = elite_mut_rate        

    def _reset_hyperparameter(self, name, value):        
        if name == 'pop_size':
            self.empty_zeros = np.zeros((name, 8), dtype=np.uint8)
        elif name == 'obj_fxn':            
            value = obj_fxn_map[value]            
        # assert self._hasattr_(name), 'Error the attribute, ' + name + ', does not exist'        
        self.__setattr__(name, value)

    def _reset_encoding(self, encoding, numeric_percent_resolution, forced_bitsize=None, bits_are_ratios=False):        
        if encoding == 'degen':                    
            # nucleotide frequencies will be encoded using standard ACGTSWD... degenerate encoding
            # this will mean that one value (numeric value of 0 means nothing in this space and results in a trivial solution a=0,c=0,t=0,andg=0...should be selected against of course...)            
            
            # when nt count is represnted as a bit, then it means that that a "1" represents the occurrence of a letter at a specific COUNT position
              #P1 #P2 #P3 
            # A 1  1  1
            # C 1  1  0
            # G 0  1  0
            # T 0  1  0

              #P1                     #P2
            #A 111011011110110111011  000000000000000000000
            self.nt_count_represented_as_bit = True
            self.bitsize = 1

        elif encoding=='numeric':
            assert numeric_percent_resolution < 100, 'Error, the numerical percent resolution must be less than 100%'
            assert numeric_percent_resolution > 0, 'Error, the numerical percent resolution must be greater than 0%'
            assert 100.0/numeric_percent_resolution == int(100.0/numeric_percent_resolution), 'Error, the numerical percent resolution must be divisible by 100 (i.e. 1%, 2%, 5%, 0.2%...etc)'

            max_numerical_value = int(100/numeric_percent_resolution)    
            if bits_are_ratios:
                # using this memory will ignore rounding but also biases distributions such that they can only represent integer ratios between one antoher
                self.nt_count_represented_as_bit = True
                self.bitsize = max_numerical_value
            else:
                # using this method is more compact (i.e. we can express 1->50 using lower bitvalues except now 22% of the space will not encode valid solutions) (in other words, the sum of the nucleotide frequencies will not add up to 0)
                self.bitsize = int(np.log2(max_numerical_value)) + 1
                if forced_bitsize is not None:
                    if forced_bitsize < self.bitsize:
                        warnings.warn('Warning the forced bitsize is smaller than the minimum size estimated for proper resolution. User asked for bitsize, {0}, while expected bitsize to be at least, {1}'.format(forced_bitsize, self.bitsize))
                    self.bitsize = forced_bitsize                    
                self.nt_count_represented_as_bit = False                        
        else:
            raise Exception('Unknown format. Currently only allow numeric/degen encoding of frequencies. Reported: ' + encoding)
        self.encoding = encoding
        self.bitsize = int(self.bitsize)
        self.numeric_percent_resolution = numeric_percent_resolution
        self.b = np.power(2, np.arange(self.bitsize, 0, -1))

    def make_population(self):
        self.variants = self.generate_bitcount(
            self.bitsize,
            self.pop_size,
            self.alphabet_size
        )
        return self.variants
    
    def population_point_mutation(self, bitmatrix_members_to_mutate, rate):
        """
        Perform random mutations in sequences where we flip the bits (0 becomes 1 at random sections in a codon)
        bitmatrix_members_to_mutate follows structure defined in generate_bitcount above
        
        mr = rate at which any bit in the population can be mutated and flipped
        
        """
        tmp = np.random.choice([True, False], p=[rate, 1 - rate], size=bitmatrix_members_to_mutate.shape)
        bitmatrix_members_to_mutate[tmp] = ~bitmatrix_members_to_mutate[tmp]
        return bitmatrix_members_to_mutate

    def single_crossover(self, bitmatrix_members_to_mutate, pairwise_crossovers):
        breakpoint = np.random.choice(np.arange(1, bitmatrix_members_to_mutate.shape[1] - 1), len(pairwise_crossovers)).reshape(-1, 1)
        
        data = np.hstack([pairwise_crossovers, breakpoint])    
        
        for c in data:      
            assert(c[2]<bitmatrix_members_to_mutate.shape[1])
            tmp1 = bitmatrix_members_to_mutate[c[0], :]
            tmp2 = bitmatrix_members_to_mutate[c[1], :]          
            bitmatrix_members_to_mutate[c[0], c[2]:] = tmp2[c[2]:]
            bitmatrix_members_to_mutate[c[1], c[2]:] = tmp1[c[2]:]
            
        return bitmatrix_members_to_mutate

    def double_crossover(self, bitmatrix_members_to_mutate, pairwise_crossovers):
        breakpoint = np.random.choice(np.arange(1, bitmatrix_members_to_mutate.shape[1] - 1), len(pairwise_crossovers) * 2).reshape(-1, 2)
        
        data = np.hstack([pairwise_crossovers, breakpoint])    
        
        for c in data:      
            s1 = min(c[2], c[3])
            s2 = max(c[2], c[3])
            
            if s1 == s2:
                continue
                
            tmp1 = bitmatrix_members_to_mutate[c[0], :]
            tmp2 = bitmatrix_members_to_mutate[c[1], :]          
            bitmatrix_members_to_mutate[c[0], s1:s2+1] = tmp2[s1:s2+1]
            bitmatrix_members_to_mutate[c[1], s1:s2+1] = tmp1[s1:s2+1]
            
        return bitmatrix_members_to_mutate
    
    def mutate_population(self, bitmatrix, density, rate):
        """
        Randomly select members from a population to then perform point mutations
        """
        # THIS WILL BE AN INPLACE MUTATION!
        mutate_rows = np.random.choice([True, False], p=[density, 1 - density], size=bitmatrix.shape[0])
        # point mutations
        bitmatrix[mutate_rows, :] = self.population_point_mutation(bitmatrix[mutate_rows, :], rate)    
        return bitmatrix
    
    def generate_offspring(self, num_offspring, bitmatrix, combos):        
        no_mut_rate = 1.0 - self.cross_rate - self.double_cross_rate        
        assert no_mut_rate >= 0.0 
        
        perform_crossover = np.random.choice([0, 1, 2], p=[no_mut_rate, self.cross_rate, self.double_cross_rate], size=int(num_offspring / 2))
        pairs_to_cross = combos[np.random.choice(combos.shape[0], size=int(num_offspring/2), replace=False)]
        
        data = np.hstack([perform_crossover.reshape(-1, 1), pairs_to_cross])
        
        new_bitmatrix = bitmatrix[:num_offspring, :].copy()
        
        for i, d in enumerate(data):       
            if d[0] == 0:
                new_bitmatrix[i, :] = bitmatrix[d[1], :]
                new_bitmatrix[i + perform_crossover.shape[0], :] = bitmatrix[d[2], :]
            elif d[0] == 1:
                breakpoint = np.random.choice(np.arange(1, bitmatrix.shape[1] - 1), 1)
                b1 = breakpoint[0]
                new_bitmatrix[i, :b1] = bitmatrix[d[1], :b1]
                new_bitmatrix[i, b1:] = bitmatrix[d[2], b1:]
                new_bitmatrix[i + perform_crossover.shape[0], :b1] = bitmatrix[d[2], :b1]
                new_bitmatrix[i + perform_crossover.shape[0], b1:] = bitmatrix[d[1], b1:]
            else:
                breakpoint = np.sort(np.random.choice(np.arange(1, bitmatrix.shape[1] - 1), 2))
                b1 = breakpoint[0]
                b2 = breakpoint[1]
                new_bitmatrix[i, :] = bitmatrix[d[1], :]
                new_bitmatrix[i, b1:b2+1] = bitmatrix[d[2], b1:b2+1]            
                new_bitmatrix[i + perform_crossover.shape[0], :] = bitmatrix[d[2], :]
                new_bitmatrix[i + perform_crossover.shape[0], b1:b2+1] = bitmatrix[d[1], b1:b2+1]

        return new_bitmatrix
    
    def get_nt_dist(self, variants=None):
        if variants is None:
            variants = self.variants

        if self.nt_count_represented_as_bit is True:
            # no need to re-normalize the data because representing each base as relative ratio rather than frequency        
            codon_numeric = CodonDesigner.get_codon_freq(
                self.bit_rep_to_num(variants, self.bitsize, self.alphabet_size)
            )            
        else:            
            # convert bits into numerical values, and then calculate relative frequencies. Renormalize data such that each frequency is present in allowed resolution
            codon_numeric = self.renormalize_data(
                self.bit_arr_to_num(
                    variants, self.bitsize, self.alphabet_size, self.b
                ),
                self.numeric_percent_resolution
            )
            # FORCE the sum of the values to be 100
            codon_numeric[:, 0, :] = 100.0 - codon_numeric[:, 1:, :].sum(axis=1)
            # woops, it was less than 0, so lets truncate the first row and then ...
            codon_numeric[codon_numeric < 0] = 0
            # update the last row so we are back to 100            
            codon_numeric[:, 3, :] = 100 - codon_numeric[:, :3, :].sum(axis=1)

            # assert (codon_numeric >= 0).all(), codon_numeric
        return codon_numeric

    def evaluate_fitness(self, desired_dist, weights=None):    
        codon_numeric = self.get_nt_dist(self.variants)

        # based on frequency of each nucleotide calculate the aa frequency
        nt_dist_to_aa = self.nt_to_aa_dist(codon_numeric, self.AA_ALPHABET, self.codon_index, round_freq_to_decimal=self.round_aa_freq_precision)  # make sure rows are AA and columns are each population variant                        
        
        fitness = np.ones((nt_dist_to_aa.shape[0]))
        
        # identify which distributions of nucleotides add up to 1 (all other ones are not realisable)
        above_one = (codon_numeric.reshape(codon_numeric.shape[0], -1) >= 0).all(axis=1)
        
        sum_100 = (codon_numeric.sum(axis=1) == 100).all(axis=1)
        proper_distributions = (above_one & sum_100)

        fitness[proper_distributions] = self.obj_fxn(
            nt_dist_to_aa[proper_distributions, :],
            desired_dist,
            weights=weights
        )

        # assert fitness.max() < self.bad_distribution_fitness, (fitness, fitness.max(), self.bad_distribution_fitness)

        fitness[~proper_distributions] = self.bad_distribution_fitness
        self.distribution_failure += (~proper_distributions).sum()
        
        return fitness

    def iter_genetic_algorithm(self, desired_dist, weights=None):
        desired_num_elite_to_store = int(self.elite_fraction * self.pop_size)            
        self.fit_vs_iter = []
        self.distribution_failure = 0     
        self.success = False                                   
      
        # 1) Generate a population
        self.make_population()
                
        unique_combinations = np.array([
            np.array(c) for c in combinations(np.arange(self.pop_size), 2)
        ])
        yield ('HIT NEXT AGAIN TO START ALGORITHM', time.time())
        while True:            
            # 2) Evaluate fitness of each member
            fitness = self.evaluate_fitness(desired_dist)
            # sort fitness of each member            
            fitness_sorted_arg = np.argsort(fitness)
            # print(self.bad_distribution_fitness)                  

            num_elite_to_store = min(fitness.shape[0] - np.argmax(fitness[fitness_sorted_arg] == self.bad_distribution_fitness), desired_num_elite_to_store)
            
            # this is the best winner
            min_fitness = fitness[fitness_sorted_arg[0]]            
            
            self.fit_vs_iter.append(min_fitness)

            response = yield min_fitness

            if response == 'STOP':
                self.variants = self.variants[fitness_sorted_arg, :]
                break

            # step 4 perform selection/ remove low fitness members
            fitness_weight = (1 - fitness)
            fitness_weight -= fitness_weight.max()
            # print(fitness_weight.max())
            fitness_weight = np.exp(fitness_weight)/(np.exp(fitness_weight).sum())
            # print(num_elite_to_store)            

            if num_elite_to_store > 0:
                # ALWAYS store the top variant (DONT MUTATE IT!)
                top_variant = self.variants[fitness_sorted_arg[0], :]
            else:
                top_variant = self.variants[0:0,:]

            
            self.elite_variants = np.vstack([
                top_variant,
                self.mutate_population(
                    self.variants[fitness_sorted_arg[1:num_elite_to_store], :],
                    self.elite_mut_density,
                    self.elite_mut_rate
                )
            ])
            
            self.elite_variants = np.unique(self.elite_variants, axis=0)     

            # use size = pop_size so that we dont have to calculate unique combinations each time (its fixed for that pop size)
            selected_winners = np.random.choice(np.arange(fitness.shape[0]), size=self.pop_size, p=fitness_weight, replace=True)
            
            fitness = fitness[selected_winners]        
            fitness_index = np.argsort(fitness)
            self.variants = self.variants[fitness_index, :]

            num_offspring = self.pop_size - self.elite_variants.shape[0]
            # step 5: generate offspring            
            new_variants = self.mutate_population(
                self.generate_offspring(num_offspring, self.variants, unique_combinations),
                self.mut_density,
                self.mut_rate
            )

            new_variants = np.unique(new_variants, axis=0)

            new_population = self.generate_bitcount(
                self.bitsize,
                self.pop_size - new_variants.shape[0] - self.elite_variants.shape[0],
                self.alphabet_size
            )   

            # print(self.pop_size - new_variants.shape[0] - self.elite_variants.shape[0])            
            self.variants = np.vstack([
                self.elite_variants,
                new_variants,
                new_population
            ])
            
            assert self.variants.shape[0] == self.pop_size            
        
    def run_genetic_algorithm(self, 
        desired_dist, weights=None, report_top=10, round_aa_freq_precision=4
    ):        
        self.round_aa_freq_precision = round_aa_freq_precision
        if self.max_iter != float('Inf'):
            report_type = 'iter'
            iter_report = int(0.05 * self.max_iter)
            print('This program will stop after ' + str(self.max_iter) + ' iterations')
        elif self.max_time != float('Inf'):
            report_type = 'time'
            iter_report = int(0.05 * self.max_time)
            print('This program will stop after ' + str(self.max_time) + ' seconds')
        else:
            iter_report = 30
            report_type = 'time'
            print('This program will not stop after a set number of iterations or seconds')
        start_time = time.time()
        iter_count = 0
        
        assert np.round(desired_dist.sum(), round_aa_freq_precision) == 1, 'Error the provided distribution must sum to 1'
        desired_dist = np.round(desired_dist/desired_dist.sum(), round_aa_freq_precision)
        

        iter_algo = self.iter_genetic_algorithm(desired_dist, weights)                
        self.response = 'run'

        def exit_gracefully(signal, frame):
            self.response = 'STOP'            
            print('CTRL-C pressed. Exiting loops!', min_fitness, self.thresh)                

        # if user presses ctrl-c then break out of loop but still report current results
        signal.signal(signal.SIGINT, exit_gracefully)

        starting_message = next(iter_algo)
        # print(starting_message)
        prev_time = start_time
        while True:
            try:
                min_fitness = iter_algo.send(self.response)
            except StopIteration:
                break

            current_time = time.time() - start_time

            if min_fitness <= self.thresh:
                # step 3: terminate if reach threshold 
                self.success = True
                print('Found a solution!', min_fitness, self.thresh)
                self.response = 'STOP'
            elif current_time > self.max_time:
                print('Time ran out!', current_time, min_fitness)
                self.response = 'STOP'
            elif iter_count > self.max_iter:
                print('Max iteration reached!', self.max_iter, min_fitness)
                self.response = 'STOP'
                        
            iter_count += 1
            if report_type == 'iter':
                if iter_count % iter_report == 0:
                    print(iter_count, iter_report, min_fitness, self.response)
            elif report_type == 'time':
                if time.time() - prev_time >= iter_report:
                    print(time.time() - start_time, iter_report, min_fitness, self.response)
                    prev_time = time.time()
        
        return self.report_results(desired_dist, report_top)

    def report_results(self, desired_dist, report_top):                                
        final_variants = self.get_nt_dist(self.variants[:report_top,:])
        # distances = self.evaluate_fitness(desired_dist)[:report_top]
        variants_aa_dist = self.nt_to_aa_dist(final_variants, round_freq_to_decimal=self.round_aa_freq_precision)        
        return final_variants, self.success, [[(k, p, dp) for k, p, dp in zip(self.AA_ALPHABET, variants_aa_dist[v], desired_dist[0])] for v in range(variants_aa_dist.shape[0])], self.fit_vs_iter
