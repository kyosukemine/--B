from TICC_solver import TICC

fname = "TICC用データ"
ticc = TICC(window_size=100, number_of_clusters=6, lambda_parameter=11e-2, beta=1000, maxIters=10, threshold=2e-5,
            write_out_file=False, prefix_string="output_folder/", num_proc=1)
(cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

print(cluster_assignment)
np.savetxt( "Results", cluster_assignment, fmt='%d', delimiter=',')