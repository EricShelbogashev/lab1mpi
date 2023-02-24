Compilation && Execution script
`mpicxx -lm -o out main.c && mpiexec -n <proc_number> --oversubscribe out`