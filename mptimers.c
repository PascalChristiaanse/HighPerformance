void start_timer()
{
	if (!timer_on)
	{
	MPI_Barrier(MPI_COMM_WORLD);
	ticks = clock();
	wtime = MPI_Wtime(); 
	timer_on = 1;
	}
}

 
void resume_timer()
{
	if (!timer_on)
	{
	ticks = clock()  -  ticks; 
	wtime = MPI_Wtime() - wtime; 
	timer_on = 1;
	}
}

void stop_timer()
{
	if (timer_on)
	{
	ticks = clock()  -  ticks; 
	wtime = MPI_Wtime() - wtime; 
	timer_on = 0;
	}
}

void print_timer()
{
	if (timer_on)
	{
	stop_timer();
	printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n",
	proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime); 
	resume_timer();
	}
	else
	printf("(%i) Elapsed Wtime %14.6f s (%5.1f%% CPU)\n",
	proc_rank, wtime, 100.0 * ticks * (1.0 / CLOCKS_PER_SEC) / wtime);
}
