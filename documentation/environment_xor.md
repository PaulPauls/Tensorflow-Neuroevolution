## TFNE Documentation for XOR Environment ##

--------------------------------------------------------------------------------

#### Specification ####

The XOR environment is a benchmark test environment for Neuroevolution
algorithms as it's a very simple and fast environment that can only be solved
if the algorithm evolves and maintains a genome that evolved nodes beyond the
minimal topology. The genomes phenotype should simulate the following mapping:

Var   | - | - | - | - 
------|-------|--------|--------|------
**x** | [0,0] | [0, 1] | [1, 0] | [1,1]
**y** | [0]   | [1]    | [1]    | [0]
