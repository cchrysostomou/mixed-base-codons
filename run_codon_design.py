from codon_design_genetic_algo import CodonDesigner, obj_fxn_map, default_unique_aa
import argparse
import pickle
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Identify best set of solutions for mixed bases in IDT')
    parser.add_argument('desired', type=str, help='Provide the desired amino acid distributions in the following format AA_freq:AA_freq...')    
    parser.add_argument('population_size', type=int, help='Number of members in population to generate')    
    parser.add_argument('--pickled_result_location', type=str, help='Location where to store the final pickled file result')
    parser.add_argument('--obj_fxn', default='square_dist', type=str, help='Name of objective function to use', choices=obj_fxn_map.keys())    
    parser.add_argument('--mutation_density', default=0.3, type=float, help='Fraction of population to perform point mutations')
    parser.add_argument('--mutation_rate', default=0.2,  type=float, help='Fraction of positions to mutate')
    parser.add_argument('--single_crossover_rate', default=0.8, type=float, help='Fraction of population to perform single crossover recombination')
    parser.add_argument('--double_crossover_rate', default=0.01, type=float,help='Fraction of population to perform double crossover recombination')
    parser.add_argument('--elite_fraction', default=0.05, type=float, help='Fraction of best scores to store without performing mutational analysis')
    parser.add_argument('--thresh', default=1e-3, type=float, help='Minimum threshold in fitness function to stop algorithm')
    parser.add_argument('--max_time', default=None, type=float, help='If defined then represnts the maximum number of seconds the algorithm is allowed to perform')
    parser.add_argument('--max_iter', default=1e6, type=int, help='If defined then represnts the maximum number iterations the algorithm is allowed to attempt')
    parser.add_argument('--encoding', choices=['numerical', 'degen'], default='numerical', type=str)
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--elite_mutation_density', default=0.0, type=float, help='If defined, then allow elite members to undergo point mutations')
    parser.add_argument('--elite_mutation_rate', default=0.1, type=float, help='If defined, then allow elite members to undergo point mutations')
    parser.add_argument('--report_top', default=10, type=int, help='Number of total solutions to report')

    aa_to_pos = {l: p for p, l in enumerate(default_unique_aa)}

    args = parser.parse_args()
    
    parsed_dist = { aa.split('_')[0] : float(aa.split('_')[1]) for aa in args.desired.split(':') }
    desired_dist = np.array([0] * len(default_unique_aa), dtype=float)
    for k, v in parsed_dist.items():
        desired_dist[aa_to_pos[k]] = v
    
    assert desired_dist.sum() == 1, 'Error the provided amino acid distribution must sum to 1! ' + str(desired_dist)
    
    print(desired_dist)

    optimizer = CodonDesigner(
        pop_size=args.population_size, forced_bitsize=8, elite_fraction=args.elite_fraction, 
        elite_mut_density=args.elite_mutation_density, elite_mut_rate=args.elite_mutation_rate, 
        mut_density=args.mutation_density, mut_rate=args.mutation_rate, 
        cross_rate=args.single_crossover_rate, double_cross_rate=args.double_crossover_rate,
        numeric_percent_resolution=args.resolution, encoding=args.encoding, thresh=args.thresh,
        max_time=args.max_time, max_iter=args.max_iter
    )

    result = optimizer.run_genetic_algorithm(
        args.desired, round_aa_freq_precision=5, report_top=args.report_top
    )

    print(result)
    print('Success! ' + str(result[-1][-1]) if result[1] else 'Iteration stopped at ' + str(result[-1][-1])) 

    if args.pickled_result_location:
        with open(args.picked_result_location, 'wb') as o:
            o.write(pickle.dumps(result))