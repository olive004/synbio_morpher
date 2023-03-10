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
threads=0
outCsvCols=*
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


# CSV behavior
# Using the argument --outCsvCols, the user can specify what columns are printed to the output using a comma-separated list of colIds. Available colIds are

#     id1 : id of first sequence (target)
#     id2 : id of second sequence (query)
#     seq1 : full first sequence
#     seq2 : full second sequence
#     subseq1 : interacting subsequence of first sequence
#     subseq2 : interacting subsequence of second sequence
#     subseqDP : hybrid subsequences compatible with hybridDP
#     subseqDB : hybrid subsequences compatible with hybridDB
#     start1 : start index of hybrid in seq1
#     end1 : end index of hybrid in seq1
#     start2 : start index of hybrid in seq2
#     end2 : end index of hybrid in seq2
#     hybridDP : hybrid in VRNA dot-bracket notation (interaction sites only)
#     hybridDPfull : hybrid in VRNA dot-bracket notation (full sequence length)
#     hybridDB : hybrid in dot-bar notation (interactin sites only)
#     hybridDBfull : hybrid in dot-bar notation (full sequence length)
#     bpList : list of hybrid base pairs, e.g. '(4,3):(5,2):(7,1)'
#     E : overall interaction energy
#     ED1 : ED value of seq1
#     ED2 : ED value of seq2
#     Pu1 : probability to be accessible for seq1
#     Pu2 : probability to be accessible for seq2
#     E_init : initiation energy
#     E_loops : sum of loop energies (excluding E_init)
#     E_dangleL : dangling end contribution of base pair (start1,end2)
#     E_dangleR : dangling end contribution of base pair (end1,start2)
#     E_endL : penalty of closing base pair (start1,end2)
#     E_endR : penalty of closing base pair (end1,start2)
#     E_hybrid : energy of hybridization only = E - ED1 - ED2
#     E_norm : length normalized energy = E / ln(length(seq1)*length(seq2))
#     E_hybridNorm : length normalized energy of hybridization only = E_hybrid / ln(length(seq1)*length(seq2))
#     E_add : user defined energy correction term incorporated into E
#     w : Boltzmann weight of E, e.g. used for partition function computation
#     seedStart1 : start index of the seed in seq1 (* see below)
#     seedEnd1 : end index of the seed in seq1 (* see below)
#     seedStart2 : start index of the seed in seq2 (* see below)
#     seedEnd2 : end index of the seed in seq2 (* see below)
#     seedE : overall energy of the seed only (including seedED etc) (* see below)
#     seedED1 : ED value of seq1 of the seed only (excluding rest) (* see below)
#     seedED2 : ED value of seq2 of the seed only (excluding rest) (* see below)
#     seedPu1 : probability of seed region to be accessible for seq1 (* see below)
#     seedPu2 : probability of seed region to be accessible for seq2 (* see below)
#     Eall : ensemble energy of all considered interactions (-RT*log(Zall))
#     Zall : partition function of all considered interactions
#     Eall1 : ensemble energy of all considered intra-molecular structures of seq1 (given its accessibility constraints)
#     Eall2 : ensemble energy of all considered intra-molecular structures of seq2 (given its accessibility constraints)
#     EallTotal : total ensemble energy of all considered interactions including the ensemble energies of intra-molecular structure formation (Eall+Eall1+Eall2)
#     Etotal : total energy of an interaction including the ensemble energies of intra-molecular structure formation (E+Eall1+Eall2)
#     Zall1 : partition function represented by Eall1 (exp(-Eall1/RT))
#     Zall2 : partition function represented by Eall2 (exp(-Eall2/RT))
#     P_E : probability of an interaction (site) within the considered ensemble
#     RT : normalized temperature used for Boltzmann weight computation