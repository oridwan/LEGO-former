import os
import argparse
import pandas as pd
from tqdm import tqdm
from time import time
from multiprocessing import Pool
from functools import partial
from pyxtal import pyxtal
from pyxtal.lego.builder import builder
from pyxtal.db import database_topology

def process_rep(rep, discrete, discrete_cell, discrete_res):
    xtal = pyxtal()
    # Remove the extra energy and labels
    mod = (len(rep) - 7) % 4
    if mod > 0: rep = rep[:-mod]
    try:
        xtal.from_tabular_representation(rep,
                                         normalize=False,
                                         discrete=discrete,
                                         discrete_cell=discrete_cell,
                                         N_grids=discrete_res)
        if xtal.valid and len(xtal.atom_sites) > 0:
           return rep, sum(xtal.numIons)
    except:
        print(f"Failed to process: {rep}")
    return None

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Relaxation code.")
    parser.add_argument('--ncpu', type=int, default=1,
                        help='N_cpu for parallel computation')
    parser.add_argument('--csv', help='csv file path')
    parser.add_argument('--begin', type=int, default=0, help='end count')
    parser.add_argument('--end', type=int, default=-1, help='end count')
    parser.add_argument('--source', default='data/source/sp2_sacada.db', 
                        help='path to source database')
    parser.add_argument('--prototype', default='graphite',
                        help='prototype for reference environment')
    parser.add_argument('--CN', type=int, default=3,
                        help='coordination number for reference environment')

    # Check if use_mpi is invoked
    rank, size = 0, 1
    print(f"Current rank-{rank}, size-{size}")

    # Parse arguments
    t0 = time()
    args = parser.parse_args()
    f = args.csv
    ncpu = args.ncpu
    begin, end = args.begin, args.end

    xtal = pyxtal()
    xtal.from_prototype(args.prototype)
    cif_file = xtal.to_pymatgen()

    name = f.split('/')[-1].split('.')[0]
    os.makedirs(name, exist_ok=True)

    # 1. Load and get valid structures sorted by number of atoms
    df = pd.read_csv(f)
    val = df['x0'].max()
    if val < 5 + 1e-3:
        discrete, discrete_res = False, None
    elif val < 50 + 1e-3:
        discrete, discrete_res = True, 50
    else:
        discrete, discrete_res = True, 100

    if abs(df['a'][0] - round(df['a'][0])) < 1e-6 and abs(df['c'][0] - round(df['c'][0])) < 1e-6 :
        discrete_cell = True
    else:
        discrete_cell = False

    full_data = df.to_numpy()
    if end == -1:
        full_data = full_data[begin:]
    else:
        full_data = full_data[begin:end]

    # Split the data equally among the ranks
    chunk_size = len(full_data) // size
    chunks = [full_data[i*chunk_size:(i+1)*chunk_size]
            for i in range(size)]
    data = chunks[0]
    N0 = len(data)
    print(f"Rank-{rank} receives {N0} structures from {f}")

    # Use multiprocessing to speed up the processing of each rep
    partial_process_rep = partial(process_rep,
                                  discrete=discrete,
                                  discrete_cell=discrete_cell,
                                  discrete_res=discrete_res)
    lists = []
    with Pool(ncpu) as pool:
        for result in tqdm(pool.imap(partial_process_rep, data),
                           total=len(data),
                           desc=f"Processing Rank-{rank}"):
            if result is not None:
                lists.append(result)

    # list of sorted (xtal, numIons)
    sorted_lists = sorted(lists, key=lambda x: x[-1])
    reps = [l[0] for l in sorted_lists]
    N1 = len(reps)
    print(f"Rank-{rank} receives {N1} structures for optimization ")

    # 2. Setup builder and run optimization
    bu = builder(['C'], [1], rank=rank, prefix=f'{name}/mof')
    bu.set_descriptor_calculator(mykwargs={'rcut': 2.1})
    bu.set_reference_enviroments(cif_file)
    bu.set_criteria(CN={'C': [args.CN]})
    xtals = bu.optimize_reps(reps, ncpu=ncpu,
                             minimizers=[('Nelder-Mead', 100),
                                         ('L-BFGS-B', 400),
                                         ('L-BFGS-B', 200)],
                             N_grids=discrete_res)
    N2 = len(xtals)
    bu.db.update_row_energy('GULP', ncpu=ncpu, calc_folder=f"{name}/gulp_{rank}")
    N3 = bu.db.get_db_unique(f'{name}/unique_{rank}.db')
    t = int((time()-t0)/60)
    print(f'R-{rank} N0/N1/N2/N3: {N0}/{N1}/{N2}/{N3} in {t} min/{ncpu} cores')
    local_data = (N0, N1, N2, N3)

    # 3. Merge all db files
    os.system(f'mv {name}/unique_0.db {name}/final.db')

    # 4. Write metrics
    db = database_topology(args.source, log_file=f'{name}/sp2.log')
    overlaps = db.check_overlap(f'{name}/final.db')
    N4 = len(overlaps)
    with open(f'{name}/metric.txt', 'w') as f:
        f.write(f'Source data:     {args.csv}\n')
        f.write(f'Elapsed minutes: {t:12d}\n')
        f.write(f'N_parallel_cpus: {ncpu:12d}\n')
        f.write(f'N_total_count:   {N0:12d}\n')
        f.write(f'N_valid_xtal:    {N1:12d}\n')
        f.write(f'N_valid_env:     {N2:12d}\n')
        f.write(f'N_unique_xtal:   {N3:12d}\n')
        f.write(f'N_train:         {N4:12d}\n')