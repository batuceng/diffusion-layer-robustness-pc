import os
import argparse
import numpy as np
import subprocess

import warnings
warnings.filterwarnings("ignore")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--t_list_min', type=str, default='0,0,0,0,0')
parser.add_argument('--t_list_max', type=str, default='100,100,100,100,100')
parser.add_argument('--t_list_step', type=str, default='5,5,5,5,5')
parser.add_argument('--searh_method', type=str, default='random',
                    choices=['random', 'grid'],
                    help='Exhausitive search method for optimization')

# Must give
parser.add_argument('-model', type=str, 
                    choices=['pointnet', 'pointnet2', 'dgcnn', 'curvenet', 'pct', 'pointmlp'],
                    help='Model to use, [pointnet, pointnet2, dgcnn, curvenet, pct, pointmlp]')
parser.add_argument('-attack', type=str, 
                    choices=['add', 'cw', 'drop', 'knn', 'pgd', 'pgdl2'],)
parser.add_argument('-cuda_device', type=int, choices=[0,1,2,3])

# Additional args
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_size', type=int, default=np.inf)

parser.add_argument('--iters', type=int, default=np.inf)

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
np.random.seed(args.seed)

t_list_min = [int(i.strip()) for i in args.t_list_min.split(",")]
t_list_max = [int(i.strip()) for i in args.t_list_max.split(",")]
t_list_steps = [int(i.strip()) for i in args.t_list_step.split(",")]

lower_bound = np.array(t_list_min)
upper_bound = np.array([max_val//step_size for max_val,step_size in zip(t_list_max, t_list_steps)])+1
step_size = np.array(t_list_steps)

# Move One folder above to access test_distance.py
os.chdir(os.path.join(os.path.dirname(__file__),".."))

# Command strings
cuda_str = f'CUDA_VISIBLE_DEVICES={args.cuda_device}'
model_str = f'--model {args.model}'
attack_str = f'--attack {args.attack}'
batch_size_str = f'--batch-size {args.batch_size}'
# Apply testsize if different from inf
test_size_str = ''
if args.test_size != np.inf:
    test_size_str = f'--test_size {args.test_size}'

iter_count = 0
while iter_count<args.iters:
    # Random sample t values from uniform dist
    t_list = np.random.randint(lower_bound, upper_bound)*step_size
    if args.model in ["pointnet","pointnet2"]:
        t_list[-1] = 0
        
    #Create command line
    t_list_str = f"-t_list {','.join(map(str, t_list))}"
    command = f"{cuda_str} python test_distance.py {t_list_str} {model_str} {attack_str} {batch_size_str} {test_size_str}"
    print(command)
    # Exec Command
    return_code = subprocess.call(command, shell=True)
    
    # Increment counter
    iter_count += 1
    print(f"Iter:{iter_count}, args:{model_str} {attack_str} {batch_size_str} {test_size_str}")
