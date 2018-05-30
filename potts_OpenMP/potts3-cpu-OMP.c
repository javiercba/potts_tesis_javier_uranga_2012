/*
 * potts3, an optimized CPU implementation of q-state Potts model.
 * For an L*L system of the Q-state Potts model, this code starts from an
 * initial ordered state (blacks=0, whites=0), it fixes the temperature temp
 * to TEMP_MIN and run TRAN Monte Carlo steps (MCS) to attain equilibrium,
 * then it runs TMAX MCS taking one measure each DELTA_T steps to perform
 * averages. After that, it keeps the last configuration of the system and
 * use it as the initial state for the next temperature, temp+DELTA_TEMP.
 * This process is repeated until some maximum temperature TEMP_MAX is reached.
 * The whole loop is repeated SAMPLES times to average over different
 * realizations of the thermal noise.
 * The outputs are the averaged energy, magnetization and their related second
 * and fourth moments for each temperature.
 * Copyright (C) 2010 Ezequiel E. Ferrero, Juan Pablo De Francesco,
 * Nicolás Wolovick, Sergio A. Cannas
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This code was originally implemented for: "q-state Potts model metastability
 * study using optimized GPU-based Monte Carlo algorithms",
 * Ezequiel E. Ferrero, Juan Pablo De Francesco, Nicolás Wolovick,
 * Sergio A. Cannas
 * http://arxiv.org/abs/1101.0876
 */

/* Parallel Version: Major changes. 
 * New features were included, in order to distribute processing into 
 * Linux clusters using OpenMpi library, and get maximum paralization on each node, 
 * using OpenMP library for thread handling,
 * Javier Nicolas Uranga, Postgraduate Thesis in Distributed Systems, 
 * National University of Cordoba, Argentina, UNC-FAMAF, January 2012.
 * http://www.famaf.unc.edu.ar/wp-content/uploads/2014/04/8-Javier-Uranga.pdf
 */

#include <stdlib.h>  /* rand */
#include <math.h>    /* expf */
#include <stdio.h>   /* printf */
#include <string.h>  /* strlen, memset */
#include <sys/time.h> /* gettimeofday */
#include <time.h> /* time */
#include <limits.h> /* UINT_MAX */
#include <assert.h>
#include <omp.h>



// Default parameters
#ifndef Q
#define Q 9 // spins
#endif

#ifndef L
#define L 2048 // matrix side size
#endif

#ifndef SAMPLES
#define SAMPLES 1 // averages
#endif

#ifndef TRAN
#define TRAN 2000 // updates before hysterisis cycle
#endif

#ifndef TMAX
#define TMAX 800 // updates with the same temperature
#endif

#ifndef TEMP_MIN
#define TEMP_MIN 0.7 // minimum temperature
#endif

#ifndef TEMP_MAX
#define TEMP_MAX 0.75 // maximum temperature
#endif

#ifndef DELTA_TEMP
#define DELTA_TEMP 0.00005 // temperature rate of change
#endif

#ifndef DELTA_T
#define DELTA_T 10 // sampling period for energy and magnetization
#endif

// Functions

// Maximum
#define MAX(a,b) (((a)>(b))?(a):(b))

// Internal definitions and functions
// out vector size, it is +1 since we reach TEMP_MAX
//TODO: check if +1 is needed
#define NPOINTS (1+(int)((TEMP_MAX-TEMP_MIN)/DELTA_TEMP))
#define N (L*L) // system size
#define SAFE_PRIMES_FILENAME "safeprimes_base32.txt"
#define SEED (time(NULL)) // random seed
#define MICROSEC (1E-6)
#define WHITES 0
#define BLACKS 1

// For profiling purposes
#undef DETERMINISTIC_UPDATE

//#define PROFILE_SPINFLIP
#undef PROFILE_SPINFLIP

typedef unsigned char byte;

// temperature, E, E^2, E^4, M, M^2, M^4
struct statpoint {
	double t;
	double e; double e2; double e4;
	double m; double m2; double m4;
};

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

// Maximum expected number of threads in OMP
#define MAX_NUM_THREADS 512
// state of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
static unsigned long long x[MAX_NUM_THREADS];
// multipliers (constants once initialized)
static unsigned int a[MAX_NUM_THREADS];

// The grid: global arrays are more optimization prone
static byte whites[L/2][L];
static byte blacks[L/2][L];


#include "MCMLrng.c"

#define E_STENCIL 5 // the 2-dim first neighbour stencil has 5 possible energies, 0 to 4

#define CHUNKSIZE 1

static void updateKernel(const float temp,
		const unsigned int color,
		byte write[L/2][L],
		byte read[L/2][L]) {

	static float table_expf_temp[E_STENCIL] = {0.0f}; //table of probability acceptance for energy increase
	static float table_temp = 0.0f; // temp this table was pre-computed

	if (table_temp!=temp) { // recompute probability table
		unsigned int e = 0;
		table_temp = temp;
		for (e=1; e<E_STENCIL; e++) { // positive energies for a 2-dim 1 neigh stencil
			// expf, sinf, etc. have serious performance issues in Debian-based libm/amd64
			table_expf_temp[e] = exp((float)(-(float)e)/temp);
		}
	}

	#pragma omp parallel shared(read,write,table_expf_temp)
	{
	unsigned int tid = omp_get_thread_num();
	unsigned long long x_l = x[tid]; // load RNG state into local storage
	unsigned int a_l = a[tid];
	#pragma omp for schedule(static, CHUNKSIZE) //#pragma omp for
	for (unsigned int i=0; i<L/2; i++) {
		int h_before, h_after, delta_E;
		byte spin_old, spin_new;
		byte spin_neigh_x, spin_neigh_y, spin_neigh_z, spin_neigh_w;
		for (unsigned int j=0; j<L; j++) {
			spin_old = write[i][j];

			// computing h_before
			spin_neigh_x = read[i][j];
			spin_neigh_y = read[i][(j+1+L)%L];
			spin_neigh_z = read[i][(j-1+L)%L];
			spin_neigh_w = read[(i+(2*(color^(j%2))-1)+L/2)%(L/2)][j];
			h_before = -(spin_old==spin_neigh_x) - (spin_old==spin_neigh_y) -
				    (spin_old==spin_neigh_z) - (spin_old==spin_neigh_w);

			// new spin
			spin_new = (spin_old + (byte)(1 + rand_MWC_co(&x_l, &a_l)*(Q-1))) % Q;

			// h after taking new spin
			h_after = -(spin_new==spin_neigh_x) - (spin_new==spin_neigh_y) -
				   (spin_new==spin_neigh_z) - (spin_new==spin_neigh_w);

			delta_E = h_after - h_before;
			float p = rand_MWC_co(&x_l, &a_l);
			// expf, sinf, etc. have performance issues in libm/amd64
#ifdef DETERMINISTIC_UPDATE
			int change = delta_E<=0 || p<=table_expf_temp[delta_E];
			write[i][j] = (change)*spin_new + (1-change)*spin_old;
#else
			// if the energy increases, the change is stochastic
			if (delta_E<=0 || p<=table_expf_temp[delta_E]) {
				write[i][j] = spin_new;
			}
#endif
		}
	}
	x[tid] = x_l; // store again the RNG state
	a[tid] = a_l;
	} // #pragma omp parallel
}


static void update(const float temp, byte whites[L/2][L], byte blacks[L/2][L]) {
	// whites update, read from blacks
#ifdef PROFILE_SPINFLIP
	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};
	// start timer
	assert(gettimeofday(&start, NULL)==0);
#endif
	updateKernel(temp, WHITES, whites, blacks);
#ifdef PROFILE_SPINFLIP
	// stop timer
	assert(gettimeofday(&end, NULL)==0);
	assert(timeval_subtract(&elapsed, &end, &start)==0);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("PROFILE_SPINFLIP: %f ns per spinflip\n", secs/(L*L/2) * 1.0e9);
#endif

	// blacks update, read from whites
	updateKernel(temp, BLACKS, blacks, whites);
}


static double calculateKernel(byte white[L/2][L],
			      byte black[L/2][L],
			      unsigned int* M_max) {

	
	byte spin;
	byte spin_neigh_n, spin_neigh_e, spin_neigh_s, spin_neigh_w;
	
	unsigned int i,j;
	unsigned int E=0;
	unsigned int M[Q]={0};
        unsigned int Mt[Q]={0};

	
       #pragma omp parallel shared(white,black,E,M)  firstprivate(Mt) private(i, j, spin, spin_neigh_n, spin_neigh_e, spin_neigh_s, spin_neigh_w)
       {
	   
		          //t = omp_get_thread_num();
		          //printf ("soy el hilo %u de %u \n",t, T);
		
                         #pragma omp for reduction(+:E) schedule(static, CHUNKSIZE)
			  for ( i=0; i<L/2; i++) {  
					for ( j=0; j<L; j++) {
						spin = white[i][j];
						spin_neigh_n = black[i][j];
						spin_neigh_e = black[i][(j+1)%L];
						spin_neigh_w = black[i][(j-1+L)%L];
						spin_neigh_s = black[(i+(2*(j%2)-1)*1+L/2)%(L/2)][j];
						
						
						E += (spin==spin_neigh_n)+(spin==spin_neigh_e)+(spin==spin_neigh_w)+(spin==spin_neigh_s);
						
						Mt[spin] += 1; //M[spin] += 1;//;
						
						spin = black[i][j];
						
						Mt[spin] += 1; //M[spin] += 1;//
						
						
					} //del 2do for
					
				} //del 1er for // AQUI HAY UNA BARRERA IMPLICITA del omp-for!! salvo que nowait sea especificado


				 for (i=0; i<Q; i++) {
				          #pragma omp atomic
					  M[i] += Mt[i];
			
				  }
	
	}//*************************hay barrera aca tambien ****************FIN de la region del pragma parallel *************************
	

	*M_max = 0;

	for (i=0; i<Q; i++) {
		
		*M_max = MAX(*M_max, M[i]);
	}
	
	return -((double)E);

	
}



static void cycle(byte whites[L/2][L], byte blacks[L/2][L],
	   const double min, const double max,
	   const double step, const unsigned int calc_step,
	   struct statpoint stats[]) {

	unsigned int index = 0;
	int modifier = 0;
	double temp = 0.0;

	//assert ((step > 0 && min < max) || (step < 0 && min > max));

	modifier = (step > 0) ? 1 : -1;

	for (index=0, temp=min; modifier*temp <= modifier*max;
	     index++, temp+=step) {

		// equilibrium phase
		for (unsigned int j=0; j<TRAN; j++) {
			update(temp, whites, blacks);
		}

		// sample phase
		unsigned int measurments = 0;
		double e=0.0, e2=0.0, e4=0.0, m=0.0, m2=0.0, m4=0.0;
		for (unsigned int j=0; j<TMAX; j++) {
			update(temp, whites, blacks);
			if (j%calc_step==0) {
				double energy = 0.0, mag = 0.0;
				unsigned int M_max = 0;
				energy = calculateKernel(whites, blacks, &M_max);
				mag = (Q*M_max/(1.0*N) - 1) / (double)(Q-1);
				e  += energy;
				e2 += energy*energy;
				e4 += energy*energy*energy*energy;
				m  += mag;
				m2 += mag*mag;
				m4 += mag*mag*mag*mag;
				measurments++;
			}
		}
		assert(index<NPOINTS);
		//TODO: struct multiassignment C idiom?
		stats[index].t = temp;
		stats[index].e += e/measurments;
		stats[index].e2 += e2/measurments;
		stats[index].e4 += e4/measurments;
		stats[index].m += m/measurments;
		stats[index].m2 += m2/measurments;
		stats[index].m4 += m4/measurments;
	}
}


static void sample(byte whites[L/2][L], byte blacks[L/2][L], struct statpoint stat[]) {
	// set the matrix to 0
	memset(whites, '\0', L*L/2);
	memset(blacks, '\0', L*L/2);

	// cycle increasing temperature
	cycle(whites, blacks,
	      TEMP_MIN, TEMP_MAX, DELTA_TEMP, DELTA_T,
	      stat);
}


static int ConfigureRandomNumbers(void) {
	unsigned long long seed = (unsigned long long) SEED;

	// Init RNG's
	int error = init_RNG(x, a, MAX_NUM_THREADS, SAFE_PRIMES_FILENAME, seed);

	return error;
}


int main(void)
{
	// the stats
	struct statpoint stat[NPOINTS] = { {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} };

	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	// parameters checking
	assert(2<=Q); // at least Ising
	assert(Q<(1<<(sizeof(byte)*8))); // do not overflow the representation
	assert(TEMP_MIN<=TEMP_MAX);
	assert(DELTA_T<TMAX); // at least one calculate()
	assert(TMAX%DELTA_T==0); // take equidistant calculate()
	assert(L%2==0); // we can halve height
	assert((L*L/2)*4L<UINT_MAX); //max energy that is all spins are the same, fits into a ulong

	// print header
	printf("# Q: %i\n", Q);
	printf("# L: %i\n", L);
	printf("# Number of Samples: %i\n", SAMPLES);
	printf("# Minimum Temperature: %f\n", TEMP_MIN);
	printf("# Maximum Temperature: %f\n", TEMP_MAX);
	printf("# Temperature Step (DELTA_TEMP): %.12f\n", DELTA_TEMP);
	printf("# Transient Equilibration Time: %i\n", TRAN);
	printf("# Transient Time in each Cycle: %i\n", TMAX);
	printf("# Time between Data Acquiring: %i\n", DELTA_T);
	printf("# Number of Points: %i\n", NPOINTS);

	// start timer
	assert(gettimeofday(&start, NULL)==0);

	if (ConfigureRandomNumbers()) {
		return 1;
	}

	// stop timer
	assert(gettimeofday(&end, NULL)==0);
	assert(timeval_subtract(&elapsed, &end, &start)==0);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Configure RNG Time: %f\n", secs);

	// start timer
	assert(gettimeofday(&start, NULL)==0);

	for (unsigned int i = 0; i < SAMPLES; i++) {
		sample(whites, blacks, stat);
	}

	// stop timer
	assert(gettimeofday(&end, NULL)==0);
	assert(timeval_subtract(&elapsed, &end, &start)==0);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Total Simulation Time: %lf\n", secs);

	printf("# Temp\tE\tE^2\tE^4\tM\tM^2\tM^4\n");
	for (unsigned int i=0; i<NPOINTS; i++) {
		printf ("%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",
			stat[i].t,
			stat[i].e/((double)N*SAMPLES),
			stat[i].e2/((double)N*N*SAMPLES),
			stat[i].e4/((double)N*N*N*N*SAMPLES),
			stat[i].m/SAMPLES,
			stat[i].m2/SAMPLES,
			stat[i].m4/SAMPLES);
	}

	return 0;
}


/*
 * http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
 * Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 * return 1 if the difference is negative, otherwise 0.
 */

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {
	/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}
	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait. tv_usec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}
