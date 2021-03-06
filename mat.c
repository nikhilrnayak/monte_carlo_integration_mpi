#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <stdlib.h>

int cnt = 0;
int **A, **B;

void scatter_data( int **, int );
void mask_operation( int **A, int N, int *Ap );
void gather_results( int *Ap, int N );
void initialize_data( int **, int );
//*Ap stands for processed matrix

int main( int argc, char **argv )
{

	
	MPI_Init( &argc, &argv ); 
	int N = atof( argv[ 1 ] );
	int i, j, myrank, nproc;
	MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
	if(N < 3)
	{
		if( myrank == 0)
		{
			printf("N should be greater thn 3\n");
			exit(1);
		}
	}
	int *Ap;
	A = ( int ** )malloc( N * sizeof( int * ) );
	B = ( int ** )malloc( N * sizeof( int * ) );
	MPI_Comm_size( MPI_COMM_WORLD, &nproc );
	Ap = ( int * )malloc( ( ( ( N - 2 ) * ( N - 2 ) / nproc ) + 1 ) * sizeof( int ) );
	for( i = 0; i < N; i++ )
	{
		A [ i ] = ( int * )malloc( N * sizeof( int ) );
		B [ i ] = ( int * )malloc( N * sizeof( int ) );
	}
	initialize_data( A, N );
	scatter_data( A, N );
	mask_operation( B, N, Ap );
	gather_results( Ap, N ); 
	if( myrank == 0 )									// print result
	{
		printf("\n");
		for( i = 0; i < N; i++ )
		{
			for( j = 0; j < N; j++ )
			{
				printf("%d\t", A [ i ] [ j ]);
			}
			printf("\n");
		}
	}
	printf("\n");
	MPI_Finalize( ); 
	return 0;
 
}

void initialize_data( int **A, int N )							//Matrix is initialized here
{
	time_t t;
	int r_val, myrank, i, j;
	srand( ( unsigned ) time( &t ) );
	MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
	if( myrank == 0 )
	{
		for( i = 0; i < N; i++ )
		{
			for( j = 0; j < N; j++ )
			{
				A [ i ] [ j ] = rand( ) % 10000;
				printf("%d\t", A [ i ] [ j ]);				//print original martix
			}
			printf("\n");
		}
	}
}

void scatter_data( int **A, int N )							//scatter the original matrix
{
	int i, x, nproc, *disp_v, *count;
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	disp_v = ( int * )malloc( nproc * sizeof( int ) );
	count = ( int * )malloc( nproc * sizeof( int ) );
	for( i = 0; i < nproc; i++)
	{
		disp_v[i] = 0;
		count[i] = N;
	}
	for( i = 0; i < N; i++)
	{
		MPI_Scatterv( &A[i][0], count, disp_v, MPI_INT, &B[i][0], N, MPI_INT, 0, MPI_COMM_WORLD ); 
	}

}

void mask_operation( int **A, int N, int *Ap )				//performs the mask operation
{
	int i, j, k = 0, myrank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	for( i = 1; i < ( N - 1 ); i++ )
	{
		for( j = 1; j < ( N - 1 ); j++ )
		{
			if( ( 1 + ( i * N ) + j ) % nproc == myrank )					//mask operation is performed by a proc if it satisfies the condiion.
			{
				Ap[ k ] = ( A [ i - 1 ] [ j - 1 ] + A [ i - 1 ] [ j ] + A [ i - 1 ] [ j + 1 ] + A [ i ] [ j - 1 ] + ( 2 * A [ i ] [ j ] ) + 
									A [ i ] [ j + 1 ]  + A [ i + 1 ] [ j - 1 ] + A [ i + 1 ] [ j ] + A [ i + 1 ] [ j + 1 ] ) / 10;
				k++;
				cnt = k;								//no. of mask elements generated by proc n. 
			}
		}
	}
}


void gather_results( int *Ap, int N )									//gathers the result
{
	int i, j, k, myrank, nproc;
	int *recv_buf = (int *)malloc( ( N - 2 ) * ( N - 2 ) * sizeof( int ) );
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	int *recv_cnt = ( int * )malloc( nproc * sizeof( int ) );					
	int *disp = ( int * )malloc( nproc * sizeof( int ) );
	disp[ 0 ] = 0;
	MPI_Gather( &cnt, 1, MPI_INT, recv_cnt, 1, MPI_INT, 0, MPI_COMM_WORLD );				//gathers the no. of elements to be received by each proc
	if( myrank == 0 )
	{
		for( i = 0; i < ( nproc - 1 ); i++)								//calcs. the displacement
			disp[ i + 1 ] = recv_cnt [ i ] + disp[ i ]; 
	}
	MPI_Gatherv(Ap, cnt, MPI_INT, recv_buf, recv_cnt, disp, MPI_INT, 0, MPI_COMM_WORLD );			//gathers the computed mask elements
	if( myrank == 0 )
	{
		printf("\n");
		for( i = 0; i < nproc; i++)									//reinitializes the matrix. I'm reusing this for proc disp. to put back
			recv_cnt[i] = 0;									//the computed mask elements to thier positions.
		for( i = 1; i < ( N - 1 ); i++ )								//this nested loop determines the return position of the gathered values.
		{
			for( j = 1; j < ( N - 1 ); j++)
			{
				for( k = 0; k < nproc; k++)
				{
					if( ( 1 + ( i * N ) + j ) % nproc == k )
					{
						A[i][j] = recv_buf[ disp[ k ] + recv_cnt[ k ] ];
						recv_cnt[ k ]++;
					}
				}
			}
		}
	}

}



