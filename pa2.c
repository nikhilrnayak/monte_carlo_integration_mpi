#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>

#define c_pi acos(-1)
#define c_e 2.718281828

void initialize_data(int);
double estimate_g(float, float, long long);
void collect_results(double *);
double fx(double);
double myrand(long long, double);

int main(int argc, char **argv)
{
	float lower_bound = atof(argv[1]);
	float upper_bound = atof(argv[2]);
	long long N = atoi(argv[3]);
	int myrank, nproc;
	double result;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	initialize_data(myrank);
	result = estimate_g(lower_bound, upper_bound, N);
	collect_results(&result);
	MPI_Finalize();
}

void initialize_data(int x)									//setting seed
{
	time_t t;
	srand((unsigned)(time(&t) + (100000 * x)));
}
	
double estimate_g(float lower_bound, float upper_bound, long long N)
{
	long long i;
	double xa;
	double result = 0;
	double coeff = (upper_bound - lower_bound)/N;
	int limit;
	int rnk, sz;
	MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
	MPI_Comm_size(MPI_COMM_WORLD, &sz);
	if(N % sz == 0)										// if no. samples modulo div no. of procs is 0 then no. of comps per proc is N / size.
	{
		limit = N / sz;
		xa = lower_bound + rnk * ((upper_bound - lower_bound) / sz);			// start index for each proc is computed here
	}
	else if(N % sz != 0)									// if no. samples modulo div no. of procs is non 0 then no. of comps per proc is N / size plus
	{											// the remaining number of comps divided among the procs based on rank
		limit = N / sz;
		if(rnk < (N % sz))
		{
			limit++;
			xa = lower_bound + rnk * ((upper_bound - lower_bound) / sz) + rnk * coeff;	// start index for each proc is computed here
		}
		else
		{
			xa = lower_bound + rnk * ((upper_bound - lower_bound) / sz) + (N % sz) * coeff;	// start index for each proc is computed here
		}
	}
	for(i = 0; i < limit; i++)
	{
		result = result + fx(xa); 						// summation of f(xi)
		xa = xa + myrand(N, coeff);						// calculation of xi
	}
	return result * coeff;
}

void collect_results(double *result)
{
	int rank;
	double buf1 = *result, buf2;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Reduce(&buf1, &buf2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0)	
		printf("result is %1.10lf\n", buf2);
}


double fx(double x)									//f(xi) generator.
{
	double denom = exp(pow(2*x, 2.0));
	double numer = 8*sqrt(2*c_pi);
	double result = numer/denom;
	return result;
}

double myrand(long long n, double step)							//randomizer function. This function generates a value between 0 and 2*((end - start)/no. of samples)
{											// this value changes for every iteration in a proc and is added to xa in every iteration.
	double step_f = step;
	double result;
	long long div = 1, temp = n, div2 = 1;
	while((temp / 10) >= 1)
	{
		temp = temp / 10;
		div2 = div2 * 10;
	}
	if(div2 > 10000)
	{
		while( step_f < (div2 / 10000))
		{
			step_f = step_f * 10;
			div = div * 10;
		}
	}
	else
	{
		while( step_f < div2 )
		{
			step_f = step_f * 10;
			div = div * 10;
		}
	}
	result = (double)(rand() % (long)(2 * step_f));
	result = result / div;
	return result;
}
