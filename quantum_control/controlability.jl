using QuantumOptics

sb = SpinBasis(1 // 2)
Sx = sigmax(sb)
Sy = sigmay(sb)
Sz = sigmaz(sb)

length(sb)

function are_ops_linear_independent(ops_list, basis)
    n = basis
    genrated_ops = ops_list
    while length(genrated_ops) >= n^2 - 1
        for op in ops_list

        end
    end
    return true
end

function is_controlable(ops_list)

end