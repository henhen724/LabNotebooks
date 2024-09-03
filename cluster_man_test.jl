using Distributed, ClusterManagers

addprocs(SlurmManager(2), partition="normal", t="00:5:00")

hosts=[]
pids=[]
for i in workers()
	println(i)
	host, pid = fetch(@spawnat i (gethostname(), getpid()))
	push!(hosts, host)
	push!(pids, pid)
end

println(hosts)
println(pids)

for i in workers()
	rmprocs(i)
end
