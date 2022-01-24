# Helix (only if --model=B):
helixMinBP=2
helixMaxBP=10
helixMaxIL=0
helixMinPu=0
helixMaxE=0
helixFullE=1

# Seed:
noSeed=0
seedBP=7

# Interaction:
mode=H               # prediction mode 
model=X              # interaction model 
acc=C                # accessibility computation
intLoopMax=10        # interaction site 

# Output:
outMode=N
outNumber=2
outOverlap=B
threads=1
# Helix (only if --model=B):
# --helixMinBP 2         # minimal number of base pairs inside a helix 
                                # (arg in range [2,4])
# --helixMaxBP 10        # maximal number of base pairs inside a helix 
                                # (arg in range [2,20])
# --helixMaxIL 0         # maximal size for each internal loop size in a 
                                # helix (arg in range [0,2]).
# --helixMinPu 0         # minimal unpaired probability (per sequence) of 
                                # considered helices (arg in range [0,1]).
# --helixMaxE 0          # maximal energy (excluding) a helix may have 
                                # (arg in range [-999,999]).
# --helixFullE           # if given (or true), the overall energy of a 
                                # helix (including E_init, ED, dangling ends, ..)
                                # will be used for helixMaxE checks; otherwise 
                                # only loop-terms are considered.

# Seed:
# --noSeed               # if given (or true), no seed is enforced within 
                                # the predicted interactions
# --seedTQ arg                  comma separated list of explicit seed base pair
#                                 encoding(s) in the format 
#                                 startTbpsT&startQbpsQ, e.g. '3|||.|&7||.||', 
#                                 where 'startT/Q' are the indices of the 5' seed
#                                 ends in target/query sequence and 'bpsT/Q' the 
#                                 respective dot-bar base pair encodings. This 
#                                 disables all other seed constraints and seed 
#                                 identification.
# --seedBP 7             # number of inter-molecular base pairs within the
                                # seed region (arg in range [2,20])

# Interaction:
# --mode H               # prediction mode : 
                                #  'H' = heuristic (fast and low memory), 
                                #  'M' = exact (slow), 
                                #  'S' = seed-only
# --model X              # interaction model : 
                                #  'S' = single-site, minimum-free-energy 
                                # interaction (interior loops only), 
                                #  'X' = single-site, minimum-free-energy 
                                # interaction via seed-extension (interior loops 
                                # only), 
                                #  'B' = single-site, helix-block-based, 
                                # minimum-free-energy interaction (blocks of 
                                # stable helices and interior loops only), 
                                #  'P' = single-site interaction with minimal 
                                # free ensemble energy per site (interior loops 
                                # only)
# --acc C                # accessibility computation :
                                #  'N' no accessibility contributions
                                #  'C' computation of accessibilities (see --accW
                                # and --accL)
# --intLoopMax 10        # interaction site : maximal number of unpaired 
                                # bases between neighbored interacting bases to 
                                # be considered in interactions (arg in range 
                                # [0,30]; 0 enforces stackings only)

# Output:
# --outMode N            # output mode :
                                #  'N' normal output (ASCII char + energy),
                                #  'D' detailed output (ASCII char + 
                                # energy/position details),
                                #  'C' CSV output (see --outCsvCols),
                                #  'E' ensemble information
# --outNumber 2          # number of (sub)optimal interactions to report 
                                # (arg in range [0,1000])
# --outOverlap B         # suboptimal output : interactions can overlap 
                                #  'N' in none of the sequences, 
                                #  'T' in the target only, 
                                #  'Q' in the query only, 
                                #  'B' in both sequences

# General:
# --threads 1            # maximal number of threads to be used for 
                                # parallel computation of query-target 
                                # combinations. A value of 0 requests all 
                                # available CPUs. Note, the number of threads 
                                # multiplies the required memory used for 
                                # computation! (arg in range [0,2])
# --personality arg             IntaRNA personality to be used, which defines 
#                                 default values, available program arguments and
#                                 tool behavior