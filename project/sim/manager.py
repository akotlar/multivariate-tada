from random import choice
from os import system
import os
import pathlib
from multiprocessing import Pool


def sim_function(args):
    folder, vals = args
    prev1, prev2, mean_af, pi1, pi2, pi12, h2_1, h2_2, rho = vals
    pathlib.Path(folder).mkdir(exist_ok=True)
    outfile = os.path.join(folder, "results.csv")
    if os.path.exists(outfile) and os.stat(outfile) > 0:
        print(f"{outfile} exists, skipping")
        return
    res = system(f"cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib) && perl ./simulate_rare_alleles.pl {prev1} {prev2} 200000 20000 {mean_af} {pi1} {pi2} {pi12} {h2_1} {h2_2} {rho} {outfile}")
    print(f"Done with: {res}")


if __name__ == "__main__":
    prev1s = [.02]
    prev2s = prev1s
    mean_afs = [1e-4]
    pi1s = [.05, .1, .2, .4]
    pi2s = pi1s
    pi_12s = [.01, .05, .1, .2, .4]
    h2_1s = [.2, .4, .6, .8]
    h2_2s = h2_1s
    rhos = [0, .2, .4, .6, .8]

    max = len(prev1s) * len(prev2s) * len(pi1s) * len(pi2s) * len(pi_12s) * len(h2_1s) * len(h2_2s) * len(rhos)

    seen = {}
    while True:
        prev1 = choice(prev1s)
        prev2 = choice(prev2s)
        mean_af = choice(mean_afs)
        pi1 = choice(pi1s)
        pi2 = choice(pi2s)
        pi12 = choice(pi_12s)
        h2_1 = choice(h2_1s)
        h2_2 = choice(h2_2s)
        rho = choice(rhos)

        folder = f"pd1_{prev1}_pd2_{prev2}_af_1e-4_pi1_{pi1}_pi2_{pi2}_pi12_{pi12}_h2_1_{h2_1}_h2_2_{h2_2}_rho_{rho}"
        if folder in seen:
            if len(seen) == max:
                break
        seen[folder] = (prev1, prev2, mean_af, pi1, pi2, pi12, h2_1, h2_2, rho)

    args = []
    for k, v in seen.items():
        args.append((k, v))

    with Pool(32) as p:
        p.map(sim_function, args)