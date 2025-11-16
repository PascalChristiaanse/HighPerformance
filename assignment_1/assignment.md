## HPC Lab — (0) Introductory MPI exercises
## HPC Lab — (0) Introductory MPI exercises
### (0.a) Ping-pong: message size vs time

	messages of different sizes.
	1, 2, 4, ..., 2^20 elements (i.e., 1, 2, 4, ..., 1,048,576) so the effect of data size
	becomes visible.
	`MPI_Recv`.

Fit an approximation for the communication time:

$$
t_{\text{comm}}(m) = \alpha + \beta\,m
$$

where:


Notes:

	happens, compute $\alpha$ and $\beta$ for each segment separately.
	the 1-element measurement when fitting the model for $t_{\text{comm}}(m)$.

Repeat the measurement with the sender and receiver on two different nodes (for
example, using `srun --nodes=2`).

Optional (extra): Compare communication measurements when using
`MPI_Sendrecv` instead of separate `MPI_Send` and `MPI_Recv`.

### (0.b) Parallel matrix–matrix multiplication

	communicated between processes (e.g., block row/column distribution, scattering,
	gather, etc.).
	$P = 1, 2, 8, 24, 48, 64$.