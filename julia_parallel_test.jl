using Distributed
@everywhere println("process: $(myid()) on host $(gethostname())")
