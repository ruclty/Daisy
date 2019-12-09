from data import create_gaussian_oracle, gaussian_simulator
from data import ChainMaker, TreeMaker, GeneralMaker, FCMaker
import pandas as pd
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Simulator data generating')
	parser.add_argument('--sample', type = int, help = 'num of samples')
	parser.add_argument('--component', type = int, help = 'num of components')
	parser.add_argument('--dim', type = int, help = 'dim of data')
	parser.add_argument('--distribution', type = str, help = 'grid or gridr')
	
	args = parser.parse_args()
	dist_type = args.distribution
	num_samples = args.sample
	n_dim = args.dim
	num_component = args.component
	
	
	if dist_type == "grid" or dist_type == "gridr":
		mus, sigmas = create_gaussian_oracle(dist_type, num_component, n_dim)
		gaussian_simulator(mus, sigmas, num_samples, "simulator_data.csv", shuffle = True, ratios = [0.2,0.8])
	else:
		supported_distributions = {'chain': ChainMaker, 'tree': TreeMaker,'fc':FCMaker,'general': GeneralMaker}
		
		maker = supported_distributions[dist_type]()
		samples = maker.sample(num_samples)
		
		df = pd.DataFrame(samples)
		df.to_csv("simulator_data.csv", index=None)
			
	
