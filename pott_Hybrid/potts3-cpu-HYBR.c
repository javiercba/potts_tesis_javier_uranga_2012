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
 * http://www.cs.famaf.unc.edu.ar/ssd/?q=node/336
 */


/*


mpicc -O3 -std=c99 -Wall -Wextra -ffast-math -march=core2 -funroll-loops -lm -fopenmp potts3-cpu-HYBR.c -o potts3-cpu-HYBR-516-1x6 -DQ=9 -DNODESX=1 -DNODESY=6 -DL=516 -DSAMPLES=1 -DTEMP_MIN=0.71f -DTEMP_MAX=0.72f -DDELTA_TEMP=0.005f -DTRAN=20 -DTMAX=80 -DDELTA_T=5

export OMP_NUM_THREADS=8

time mpirun -np 6 -machinefile mymachinefile -x OMP_NUM_THREADS ./potts3-cpu-HYBR-516-1x6



*/

#include <stdlib.h>  
#include <math.h>    
#include <stdio.h>   
#include <string.h>  
#include <sys/time.h> 
#include <time.h> 
#include <limits.h> 
#include <assert.h>
#include <omp.h>
#include <mpi.h>

//-----------------------------------------------------parametros del programa ----------------------------------------------------------------------------------

#ifndef NODESX
#define NODESX 1 
#endif

#ifndef NODESY
#define NODESY 5 
#endif

#ifndef Q
#define Q 9 
#endif

#ifndef L
#define L 2048
#endif


#ifndef SAMPLES
#define SAMPLES 1 
#endif

#ifndef TRAN //cantidad de veces de ejecucion del cilco de eq fase
#define TRAN 2000 
#endif

#ifndef TMAX	//cantidad de veces MCS que se mide 1 solo temp 
#define TMAX 800 
#endif

#ifndef DELTA_T
#define DELTA_T 10 
#endif

#ifndef TEMP_MIN
#define TEMP_MIN 0.7 
#endif

#ifndef TEMP_MAX
#define TEMP_MAX 0.75 
#endif

#ifndef DELTA_TEMP    //es DELTA_TEMP=R=step de temp
#define DELTA_TEMP 0.00005 
#endif


//------------------------------------------------------fin parametros --------------------------------------------------------------------------------

#define MAX(a,b) (((a)>(b))?(a):(b))

#define NPOINTS (1+(int)((TEMP_MAX-TEMP_MIN)/DELTA_TEMP))

#define SAFE_PRIMES_FILENAME "safeprimes_base32.txt"
#define SEED (time(NULL)) 
#define MICROSEC (1E-6)
#define WHITES 0
#define BLACKS 1


#undef DETERMINISTIC_UPDATE

#undef PROFILE_SPINFLIP



typedef unsigned char byte;


struct statpoint {
	double t;
	double e; double e2; double e4;
	double m; double m2; double m4;
};


#define E_STENCIL 5 

//-------------------------------------------------------MPI------------------------------------------------------------------------------------


unsigned int NODE_THREADS;
unsigned long long* x;
unsigned int* a;
unsigned long long*  X;
unsigned int* A;


#define ROOT 0 //MPI: define el id del nodo Master

int myrank_mpi, nprocs_mpi; 

#define N (L*L)


//-------------------------------------------------------MPI topologia ------------------------------------------------------------

#define NODESX_AMPL  (NODESX+2)
#define NODESY_AMPL  (NODESY+2)

int TOPO[NODESX_AMPL][NODESY_AMPL];

byte neighbors[4]={0}; //nodos: norte, sur, oeste, este

//-------------------------------------------------------------MPI----lattice---------------------------------------------------------

typedef struct{

	byte 		data;		//el valor del spin
	byte 		isGhost;	//dice si la celda es o no es de tipo ghost
	unsigned int 	ii;		//es el i en la matriz original: tablero de ajedrez sin compactar, pero rodeado por 2 filas y 2 cols ghost

}cell;


#define Lmpix ((L/NODESX)+2)  					      
#define Lmpiy ((L/NODESY)+2)

static cell whites[Lmpix/2][Lmpiy]; 
static cell blacks[Lmpix/2][Lmpiy]; 

//------------------------------------------MPI-------comunicacion---------------------------------------------------------------------


//definiciones de tags para mensajes entre nodos, 
//muchas veces un mismo nodo es vecino de 2 cardinales de otro, entonces la unica forma de diferenciar el mensaje es con el uso de tags
#define MSG_DATA_N 10
#define MSG_DATA_S 20
#define MSG_DATA_E 30
#define MSG_DATA_O 40


//variables de estado usadas por las funciones MPI_Recv
MPI_Status statusn;
MPI_Status statuss;
MPI_Status statuso;
MPI_Status statuse;

//buffers de salida usados por las funciones MPI_Recv
byte gnt[Lmpiy]={0};
byte gst[Lmpiy]={0};
byte got[Lmpix]={0};
byte get[Lmpix]={0};

//buffers de entrada usados por las funciones MPI_Isend
byte grn[Lmpiy]={0};
byte grs[Lmpiy]={0};
byte gro[Lmpix]={0};
byte gre[Lmpix]={0};

//Definiciones requeridas por las funciones MPI_Isend y MPI_Wait
MPI_Request ireqnSend, ireqsSend, ireqoSend, ireqeSend;
MPI_Status istatnSend, istatsSend, istatoSend, istateSend;

#define NOGHOST 1
#define GHOST   2

#define CHUNKSIZE 1

//----------------------------------------------para el calculo de la magnetismo--------------------------------------------------------------------

unsigned int Ma[Q];
unsigned int *T;

//---------------------------------------------prototipos -------------------------------------------------------------------------------------------

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

static void initLattice(void);

float rand_MWC_co(unsigned long long* x, unsigned int* a);
float rand_MWC_oc(unsigned long long* x, unsigned int* a);
int init_RNG(unsigned long long *x, unsigned int *a, const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit);
void LoadRandomDebug(void);

static void ghostProcessing(void);

//****************************************************************************************************************************************************
//********************************************************* Fin declaraciones ************************************************************************
//****************************************************************************************************************************************************


//-----------------------------------------------------------------RNG--------------------------------------------------------------------


float rand_MWC_co(unsigned long long* x, unsigned int* a)
{
		//Generate a random number [0,1)
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return ((float) ((unsigned int)(*x))) / (float)0x100000000;// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)

}//end rand_MWC_co

float rand_MWC_oc(unsigned long long* x, unsigned int* a)
{
		//Generate a random number (0,1]
		return 1.0f-rand_MWC_co(x,a);
}//end rand_MWC_oc


//TODO: si molesta, sacar el const
int init_RNG(unsigned long long *x, unsigned int *a,
             const unsigned int n_rng, const char *safeprimes_file, unsigned long long xinit)

{
	FILE *fp = NULL;
	int fscanf_result = 0;
	unsigned int begin = 0u;
	unsigned int fora,tmp1,tmp2;

	if (strlen(safeprimes_file)==0) {
		// Try to find it in the local directory
		safeprimes_file = "safeprimes_base32.txt";
	}

	fp = fopen(safeprimes_file, "r");
	if (fp==NULL) {
		printf("Could not find the file of safeprimes (%s)! Terminating!\n", safeprimes_file);
		return 1;
	}

	fscanf_result = fscanf(fp, "%u %u %u", &begin, &tmp1, &tmp2);
	if (fscanf_result<3) {
		printf("Problem reading first safeprime from file %s\n", safeprimes_file);
		return 1;
	}

	// Here we set up a loop, using the first multiplier in the file to generate x's and c's
	// There are some restictions to these two numbers:
	// 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
	// also [x,c]=[0,0] and [b-1,a-1] are not allowed.

	//Make sure xinit is a valid seed (using the above mentioned restrictions)
	if ((xinit==0ull) | (((unsigned int)(xinit>>32))>=(begin-1)) | (((unsigned int)xinit)>=0xfffffffful)) {
		//xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
		printf("%llu not a valid seed! Terminating!\n",xinit);
		return 1;
	}

	unsigned int i = 0;
	for (i=0; i<n_rng; i++)
	{
		fscanf_result = fscanf(fp, "%u %u %u", &fora, &tmp1, &tmp2);
		if (fscanf_result<3) {
			printf("Problem reading safeprime %d out of %d from file %s\n", i+2, n_rng+1, safeprimes_file);
			return 1;
		}

		a[i] = fora;
		x[i] = 0;
		while ((x[i]==0) | (((unsigned int)(x[i]>>32))>=(fora-1)) | (((unsigned int)x[i])>=0xfffffffful)) {
			//generate a random number
			xinit = (xinit&0xffffffffull)*(begin)+(xinit>>32);

			//calculate c and store in the upper 32 bits of x[i]
			x[i] = (unsigned int) floor((((double)((unsigned int)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
			x[i] = x[i]<<32;

			//generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
			xinit = (xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
			x[i] += (unsigned int) xinit;
		}
		//if(i<10)printf("%llu\n",x[i]);
	}
	fclose(fp);

	return 0;
}



//----------------------------------------------------------------------------------------------------------------------------------------------------


/*
 * http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
 * Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 * return 1 if the difference is negative, otherwise 0.
 */

static int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {

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


	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;


	return x->tv_sec < y->tv_sec;
}



//---------------------------------------------------------calculo de vecinos general -----------------------------------------------------------------


void initNodeMesh(void){

	int i,j;
	for (i=0; i<NODESX_AMPL; i++)
	   for (j=0; j<NODESY_AMPL; j++)
	      TOPO[i][j]=0;

}


void buildToroNodeMesh(void){

	int i,j;
	int count=0;

	for (i=0; i<NODESX; i++){
		for (j=0; j<NODESY; j++){
			TOPO[i+1][j+1]=count;
			count ++;				
		}
	}

	//norte ok
	for (j=1; j <= NODESY; j++)
		TOPO[0][j]=TOPO[NODESX][j];
	//sur
	for (j=1; j <= NODESY; j++)
		TOPO[NODESX_AMPL-1][j]=TOPO[1][j];
	//oeste
	for (i=1; i <= NODESX; i++)
		TOPO[i][0]=TOPO[i][NODESY];
	//este
	for (i=1; i <= NODESX; i++)
		TOPO[i][NODESY_AMPL-1]=TOPO[i][1];

}


void calculateNodeNeighbors(byte V[4], int myrank){

	int i,j;
	
	for (i=1; i<=NODESX; i++){
		for (j=1; j<=NODESY; j++){

			if (myrank == TOPO[i][j]){

				//vecino Norte
				V[0]= TOPO[i-1][j];
				//vecino Sur
				V[1]= TOPO[i+1][j];
				//vecino Oeste
				V[2]= TOPO[i][j-1];
				//vecino Este
				V[3]= TOPO[i][j+1];

				
				return;

			}

		}

	}

}



//----------------------------------------------------MPI-------SETEO DEL LATTICE --------------------------------------------------------------


//Es FUNDAMENTAL ESTA FUNCION!
void defineGhostCells(void){ //ex load_structs

	initLattice(); //inicializa valores .data del struct de whites y blacks

	int i,j, ii;
	ii=0;
	
	for (i=0; i<Lmpix; i++){
		for (j=0; j<Lmpiy; j++){

			ii=(((i+j)%2)*Lmpix+i)/2; 

			if (ii<Lmpix/2){
				whites[ii][j].ii=i;		 //carga el i en la matriz original sin compactar: mapeo inverso

				if ((i==0) || (j==0) || (i==Lmpix-1) || (j==Lmpiy-1)) 	//define la condicion de ghost de la celda, 
											//i,j son de la matriz orig sin compactar
					whites[ii][j].isGhost=GHOST;
				else 
					whites[ii][j].isGhost=NOGHOST;

			}
			else if (ii>=Lmpix/2){

				blacks[ii-(Lmpix/2)][j].ii=i;  //carga el i en la matriz original sin compactar: mapeo inverso

				if ((i==0) || (j==0) || (i==Lmpix-1) || (j==Lmpiy-1)) 	//define la condicion de ghost de la celda, 
											//i,j son de la matriz orig sin compactar
					blacks[ii-(Lmpix/2)][j].isGhost=GHOST;
				else 
					blacks[ii-(Lmpix/2)][j].isGhost=NOGHOST;
			}

			
		}//2do for
	}//1er for


}


//------------------------------------------------ MPI CARGA DE GHOST------------------------------------------------------------



// esta funcion es para cargar los vectores para transmitir

// Esta funcion manipula las filas y columnas que forman el limite real de la matriz y no el limite ampliado que esta formado por filas y cols ghost
// por eso si la matriz ampliada de LmpixLmpi va desde 0 a Lmpi-1 para las filas y desde 0 a Lmpi-1 para las columnas
// hay que transmitir la fila/col 1 ya que la fila/col 0 es ghost,
// y hay que transmitir la fila/col Lmpi-2 ya que la fila/col Lmpi-1 es ghost

//TODO: APLICAR_OPENMP 
void ghostLoading(void){

  		int i, j, ii;

		for (j=1; j<Lmpiy-1; j++){ // j va desde 1 a Lmpi-2 inclusive
			//norte
			i=1;			
			ii=(((i+j)%2)*Lmpix+i)/2;	//??	//del paper, (i,j) se mapea en (ii,j)
					 
				if (ii<(Lmpix/2))
					 gnt[j]= whites[ii][j].data;
				else if (ii>=(Lmpix/2))// deberia ser un >= en vez de > ?????????
					gnt[j]= blacks[ii-(Lmpix/2)][j].data;
			//sur
			i=Lmpix-2;		
			ii=(((i+j)%2)*Lmpix+i)/2;	//??	//del paper, (i,j) se mapea en (ii,j)

				if (ii<(Lmpix/2))
					 gst[j]= whites[ii][j].data;
				else if (ii>=(Lmpix/2))	// deberia ser un >= en vez de > ?????????
					gst[j]= blacks[ii-(Lmpix/2)][j].data;
		}

		for (i=1; i<Lmpix-1; i++){	// i va desde 1 a Lmpi-2 inclusive
			//oeste
			 j=1;			
			 ii=(((i+j)%2)*Lmpix+i)/2;	//??	//del paper, (i,j) se mapea en (ii,j)

			 if (ii<(Lmpix/2))
			 	 got[i]= whites[ii][j].data;
			 else if (ii>=(Lmpix/2))// deberia ser un >= en vez de > ?????????
				 got[i]= blacks[ii-(Lmpix/2)][j].data;
			 //este
			 j=Lmpiy-2;		
			 ii=(((i+j)%2)*Lmpix+i)/2;	//??	//del paper, (i,j) se mapea en (ii,j)
					 
			 if (ii<(Lmpix/2))
			 	 get[i]= whites[ii][j].data;
			 else if (ii>=(Lmpix/2))// deberia ser un >= en vez de > ?????????
				 get[i]= blacks[ii-(Lmpix/2)][j].data;
		}


}


//-------------------------------------------MPI funciones de comunicacion GHOST-----------------------------------------------------


//envio asincrono
void asyncSend(){

	if (neighbors[0] != myrank_mpi){
		MPI_Isend(gnt, Lmpiy, MPI_BYTE, neighbors[0], MSG_DATA_N, MPI_COMM_WORLD, &ireqnSend);
	}
	if (neighbors[1] != myrank_mpi){
		MPI_Isend(gst, Lmpiy, MPI_BYTE, neighbors[1], MSG_DATA_S, MPI_COMM_WORLD, &ireqsSend);
	}
	if (neighbors[2] != myrank_mpi){
		MPI_Isend(got, Lmpix, MPI_BYTE, neighbors[2], MSG_DATA_O, MPI_COMM_WORLD, &ireqoSend);
	}
	if (neighbors[3] != myrank_mpi){
		MPI_Isend(get, Lmpix, MPI_BYTE, neighbors[3], MSG_DATA_E, MPI_COMM_WORLD, &ireqeSend);
	}

}

//operacion asincrona: controla q el envio asincrono se finalice para actualizar el buffer de salida
void syncWaitForAsyncSend (void){


	if (neighbors[0] != myrank_mpi){
		MPI_Wait(&ireqnSend, &istatnSend);
		//ahora se ppuede actualizar el buffer de envio en gnt sin race conditions
	}
	if (neighbors[1] != myrank_mpi){
		MPI_Wait(&ireqsSend, &istatsSend);
		//ahora se ppuede actualizar el buffer de envio en gst sin race conditions
	}
	if (neighbors[2] != myrank_mpi){
		MPI_Wait(&ireqoSend, &istatoSend);
		//ahora se ppuede actualizar el buffer de envio en got sin race conditions
	}
	if (neighbors[3] != myrank_mpi){
		MPI_Wait(&ireqeSend, &istateSend);
		//ahora se ppuede actualizar el buffer de envio en get sin race conditions
	}

}


void syncRecv(void){

	
	if (neighbors[0] != myrank_mpi){
		MPI_Recv(grn, Lmpiy, MPI_BYTE, neighbors[0], MSG_DATA_S, MPI_COMM_WORLD, &statusn);
	}else{
		
		for(unsigned int i=0; i<Lmpiy; i++)
				grn[i]=gst[i];
	}

	if (neighbors[1] != myrank_mpi){
		MPI_Recv(grs, Lmpiy, MPI_BYTE, neighbors[1], MSG_DATA_N, MPI_COMM_WORLD, &statuss);
	}else{
		for(unsigned int i=0; i<Lmpiy; i++)
				grs[i]=gnt[i];
	}

	if (neighbors[3] != myrank_mpi){
		MPI_Recv(gre, Lmpix, MPI_BYTE, neighbors[3], MSG_DATA_O, MPI_COMM_WORLD, &statuse);
	}else{
		for(unsigned int i=0; i<Lmpix; i++)
				gre[i]=got[i];
	}

	if (neighbors[2] != myrank_mpi){
		MPI_Recv(gro, Lmpix, MPI_BYTE, neighbors[2], MSG_DATA_E, MPI_COMM_WORLD, &statuso);
	}else{
		for(unsigned int i=0; i<Lmpix; i++)
				gro[i]=get[i];
	}
		
		
}


//comunicacion sin deadlock
void ghostComunication(void){
	
	asyncSend();
	syncRecv();
	syncWaitForAsyncSend();

}



//----------------------------------------------------------------MPI MAPEO GHOST-------------------------------------------------------------

//una vez recibidos los vectores ghost de los vecinos se mapean a las matrices blancas y negras del nodo
void ghostMapping(void){

	 int i, j, ii;

 	 // variando J
	 for ( j=1; j<Lmpiy-1; j++){ // o sea que j va desde 1 a Lmpi-2, tomando los extremos

		//norte
	 	i=0;
	  	ii=(((i+j)%2)*Lmpix+i)/2;

		if (ii<Lmpix/2)
			whites[ii][j].data=grn[j];
		else
			blacks[ii-Lmpix/2][j].data=grn[j];


		//sur
	 	i=Lmpix-1;
	  	ii=(((i+j)%2)*Lmpix+i)/2;
		if (ii<Lmpix/2)
			whites[ii][j].data=grs[j];
		else
			blacks[ii-Lmpix/2][j].data=grs[j];
	}
	
	 // variando i
	 for ( i=1; i<Lmpix-1; i++){ // o sea que i va desde 1 a Lmpi-2, tomando los extremos

		//este
	 	j=Lmpiy-1;
	  	ii=(((i+j)%2)*Lmpix+i)/2;

		if (ii<Lmpix/2)
			whites[ii][j].data=gre[i];
		else
			blacks[ii-Lmpix/2][j].data=gre[i];

		//oeste
	 	j=0;
	  	ii=(((i+j)%2)*Lmpix+i)/2;

		if (ii<Lmpix/2)
			whites[ii][j].data=gro[i];
		else
			blacks[ii-Lmpix/2][j].data=gro[i];

		
	}

}



//--------------------------------------------------------------CORE del SISTEMA-----------------------------------------------------------------------


static void updateKernel(const float temp, const unsigned int color, cell write[Lmpix/2][Lmpiy],  cell read[Lmpix/2][Lmpiy]) {

	//ghostProcessing();

	static float table_expf_temp[E_STENCIL] = {0.0f}; 
	static float table_temp = 0.0f; 

	if (table_temp!=temp) { 
		unsigned int e = 0;
		table_temp = temp;
		for (e=1; e<E_STENCIL; e++) { 

			table_expf_temp[e] = exp((float)(-(float)e)/temp);
		}
	}

	unsigned long long x_l;
	unsigned int a_l;
	#pragma omp parallel shared(read,write,table_expf_temp, x, a) private(x_l, a_l)
	{

		unsigned int tid = omp_get_thread_num();
		assert(tid<NODE_THREADS); 
		x_l = x[tid]; 
		a_l = a[tid];

		
		#pragma omp for schedule(static, CHUNKSIZE)
		for (unsigned int i=0; i<Lmpix/2; i++) {  
			
			int h_before, h_after, delta_E;
			byte spin_old, spin_new;
			byte spin_neigh_x, spin_neigh_y, spin_neigh_z, spin_neigh_w;
			
			for (unsigned int j=0; j<Lmpiy; j++) { 

				//tid = (i*Lmpix+j) % NODE_THREADS; 

				if (write[i][j].isGhost==NOGHOST){

						spin_old = write[i][j].data;

						/*
						spin_neigh_x = read[i][j].data;
						spin_neigh_y = read[i][(j+1+Lmpi)%Lmpi].data; 
						spin_neigh_z = read[i][(j-1+Lmpi)%Lmpi].data; 
						spin_neigh_w = read[(i+(2*(color^(j%2))-1)+Lmpi/2)%(Lmpi/2)][j].data; 
						*/

						//no hace falta %Lmpi ni %L/2 porque en el caso de los bordes, solo tiene que leer de los ghost
						//cada nodo no es un toroide en si mismo
						//y como las filas y columnas de los bordes no se procesan gracias al if-isGhost, 
						//los calculos % tipicos del toro no son necesarios

						spin_neigh_x = read[i][j].data;
						spin_neigh_y = read[i][(j+1)].data; 
						spin_neigh_z = read[i][(j-1)].data; 
						spin_neigh_w = read[(i+(2*(color^(j%2))-1))][j].data; //color XOR (j%2) vale 0 ó 1 Y (2*(color^(j%2))-1)) vale 1 ó -1
								

						h_before = -(spin_old==spin_neigh_x) - (spin_old==spin_neigh_y) -
							    (spin_old==spin_neigh_z) - (spin_old==spin_neigh_w);

						  spin_new = (spin_old + (byte)(1 + rand_MWC_co(&x_l, &a_l)*(Q-1))) % Q;

			
						h_after = -(spin_new==spin_neigh_x) - (spin_new==spin_neigh_y) -
							   (spin_new==spin_neigh_z) - (spin_new==spin_neigh_w);

						delta_E = h_after - h_before;

						float p = rand_MWC_co(&x_l, &a_l);
						

					#ifdef DETERMINISTIC_UPDATE
						int change = delta_E<=0 || p<=table_expf_temp[delta_E];
			  
						write[i][j].data = (change)*spin_new + (1-change)*spin_old;

					#else
			
						if (delta_E<=0 || p<=table_expf_temp[delta_E]) {
			 
							write[i][j].data = spin_new;
						}
					#endif


				}//del if
				else{
						
					continue;
				}

			}//de for2, que mueve las columnas
			
		}//del for1, que mueve las filas

		x[tid] = x_l;
		a[tid] = a_l; 


	} // #pragma omp parallel		


}//de la funcion


static void update(const float temp, cell whites[Lmpix/2][Lmpiy], cell blacks[Lmpix/2][Lmpiy]) {

	ghostProcessing();

        #ifdef PROFILE_SPINFLIP
		MPI_Barrier(MPI_COMM_WORLD);
		double secs = 0.0;
		double recbuf_secs = 0.0;
		struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};
	        assert(gettimeofday(&start, NULL)==0);// start timer

        #endif
	

	updateKernel(temp, WHITES, whites, blacks); //escribe blancas, lee de negras: params :<color, escribe, lee>

	
	#ifdef PROFILE_SPINFLIP
		MPI_Barrier(MPI_COMM_WORLD);
		assert(gettimeofday(&end, NULL)==0);// stop timer
		assert(timeval_subtract(&elapsed, &end, &start)==0);
		secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);

		MPI_Reduce(&secs, &recbuf_secs, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
		if (myrank_mpi==ROOT){
			FILE *fp = NULL;
			char _filename[40];
			sprintf(_filename, "%s%s%i%s%i%s%i%s%i%s", "pottsHYBR_", "spinFlip_", L,"-", NODESX,"-", NODESY, "-", NODE_THREADS, ".txt");
			fp = fopen(_filename, "a"); 
			fprintf(fp, "PROFILE_SPINFLIP: %f ns per spinflip\n", recbuf_secs/(L*L/2) * 1.0e9);
			fclose(fp);
	
		}



	#endif
	
	ghostProcessing();

	updateKernel(temp, BLACKS, blacks, whites); 	//escribe negras, lee blancas: params :<color, escribe, lee>

}

//output_1: valor double de Energia
//output_2: se carga el vector global M[Q], que luego participara de una operacion Gather hacia el ROOT

static double calculateKernel(cell whites[Lmpix/2][Lmpiy], cell blacks[Lmpix/2][Lmpiy]) {
	
	ghostProcessing(); //fundamental agregado!

	byte spin;
	byte spin_neigh_n, spin_neigh_e, spin_neigh_s, spin_neigh_w;
	
	unsigned int i,j;
	unsigned int E=0;

	//unsigned int M[Q]={0};
	for ( i=0; i<Q; i++){
		Ma[i]=0; 
	}

	unsigned int Mt[Q]={0};


	#pragma omp parallel shared(whites,blacks,E,Ma)  firstprivate(Mt) private(i, j, spin, spin_neigh_n, spin_neigh_e, spin_neigh_s, spin_neigh_w)
        {
			  #pragma omp for reduction(+:E) schedule(static, CHUNKSIZE)
			  for ( i=0; i<Lmpix/2; i++) {  
					for ( j=0; j<Lmpiy; j++) {

						if (whites[i][j].isGhost==NOGHOST){ 

							spin = whites[i][j].data;
							
							/*
							spin_neigh_n = blacks[i][j].data;
							spin_neigh_e = blacks[i][(j+1)%Lmpi].data; 
							spin_neigh_w = blacks[i][(j-1+Lmpi)%Lmpi].data; 
							spin_neigh_s = blacks[(i+(2*(j%2)-1)+Lmpi/2)%(Lmpi/2)][j].data; 
							*/

							//no hace falta %Lmpi ni %L/2 porque en el caso de los bordes, solo tiene que leer de los ghost
							//cada nodo no es un toroide en si mismo
							//y como las filas y columnas de los bordes no se procesan gracias al if-isGhost, 
							//los calculos % tipicos del toro no son necesarios

							spin_neigh_n = blacks[i][j].data;
							spin_neigh_e = blacks[i][(j+1)].data;               
							spin_neigh_w = blacks[i][(j-1)].data; 		    
							spin_neigh_s = blacks[(i+(2*(j%2)-1))][j].data; // (2*(j%2)-1)) es 1 ó -1
						
							E += (spin==spin_neigh_n)+(spin==spin_neigh_e)+(spin==spin_neigh_w)+(spin==spin_neigh_s);
						
							Mt[spin] += 1;
							
							spin = blacks[i][j].data;
						
							Mt[spin] += 1;


						}//del if

						else{
							continue;
						}
						
						
					} //del 2do for
					
				 } //del 1er for 

				for (i=0; i<Q; i++) { //bucle exclusivo por openmp
				          #pragma omp atomic
					  Ma[i] += Mt[i];
			
				 }
				
	
	}//****************FIN de la region del pragma parallel *************************
	
	//*M_max = 0;
	//for (i=0; i<Q; i++)
	//	*M_max = MAX(*M_max, M[i]);

	
	return -((double)E);

	
}

//-----------------------------------------------------------------------------------------------------------------------------------------------

static void ghostProcessing(void){


	ghostLoading();
	ghostComunication();
	ghostMapping();

	MPI_Barrier(MPI_COMM_WORLD);

}

unsigned int maxProcessing(void){

	unsigned int S[Q]={0};
	unsigned int TT[nprocs_mpi][Q];

	int fila=0;
	int col=0;

	for ( int j=0;j<nprocs_mpi*Q;j++){			

		TT[fila][col]=T[j];

		if (col==Q-1){
			fila +=1;
			col=0;
		}
		else{
			col +=1;
		}

	}

	for ( int j=0;j<Q;j++)
		for ( int i=0;i<nprocs_mpi;i++)
			S[j]+=TT[i][j];

	unsigned int max = 0;
	for (unsigned int i=0; i<Q; i++) 
		max = MAX(max, S[i]);


	return max;

}


static void cycle(cell whites[Lmpix/2][Lmpiy], cell blacks[Lmpix/2][Lmpiy],
		   const double min, const double max,
		   const double step, const unsigned int calc_step, 
		   struct statpoint stats[]) {

		unsigned int index = 0;
		int modifier = 0;
		double temp = 0.0;

		//TODO:
		//assert ((step > 0 && min < max) || (step < 0 && min > max));

		modifier = (step > 0) ? 1 : -1;

		for (index=0, temp=min; modifier*temp <= modifier*max;
		     index++, temp+=step) {

			// equilibrium phase
			for (unsigned int j=0; j<TRAN; j++) {

				
				//ghostProcessing();
				update(temp, whites, blacks);

			}

			// sample phase
			unsigned int measurments = 0;
			double e=0.0, e2=0.0, e4=0.0, m=0.0, m2=0.0, m4=0.0;
			for (unsigned int j=0; j<TMAX; j++) {


				//ghostProcessing();
				update(temp, whites, blacks);


				if (j%calc_step==0) {

					double energy = 0.0;
					double recbuf_energy=0;

					energy = calculateKernel(whites, blacks);

					//calculo de la energia global
					MPI_Reduce(&energy, &recbuf_energy, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

					//calculo del magnetismo global
					MPI_Gather(Ma,Q,MPI_UNSIGNED,T,Q,MPI_UNSIGNED,ROOT,MPI_COMM_WORLD);//origen, destino


	if (myrank_mpi==ROOT){//------------------------------------------ROOT sumariza---------------------------------------------------------------

						unsigned int M_max = 0;
						M_max = maxProcessing(); //procesa los datos del gather para encontrar el maximo global
						double mag = 0.0;
						mag = (Q*M_max/(1.0*N) - 1) / (double)(Q-1); 

						e  += recbuf_energy;
						e2 += recbuf_energy*recbuf_energy;
						e4 += recbuf_energy*recbuf_energy*recbuf_energy*recbuf_energy;
						m  += mag;
						m2 += mag*mag;
						m4 += mag*mag*mag*mag;

						measurments++;

						
	}//-----------------------------------------------------------------------------------------------------------------------------------------------


				}//del 1er if
			}//del 2do for


	if (myrank_mpi==ROOT){//-------------------------------------ROOT PROMEDIA------------------------------------------------------------------

				assert(index<NPOINTS);
				
				stats[index].t = temp;
				stats[index].e += e/measurments;
				stats[index].e2 += e2/measurments;
				stats[index].e4 += e4/measurments;
				stats[index].m += m/measurments;
				stats[index].m2 += m2/measurments;
				stats[index].m4 += m4/measurments;
				

	}//---------------------------------------------------------------------------------------------------------------------------------------------------


		}//del 1er for


}//de la funcion cycle


//TODO: APLICAR_OPENMP 
static void initLattice(void){

	// set the matrix to 0
	//memset(whites, '\0', Lmpi*Lmpi/2); 
	//memset(blacks, '\0', Lmpi*Lmpi/2); 

	int i,j;
	for (i=0; i<Lmpix/2; i++){
		for (j=0; j<Lmpiy; j++){

			whites[i][j].data=0; 	         //inicializacion de datos, 
			blacks[i][j].data=0;	 	//inicializacion de datos, 


		}
	}


}

static void sample(cell whites[Lmpix/2][Lmpiy], cell blacks[Lmpix/2][Lmpiy], struct statpoint stat[]) {

	initLattice();
	// cycle increasing temperature
	cycle(whites, blacks,
	      TEMP_MIN, TEMP_MAX, DELTA_TEMP, DELTA_T,
	      stat);

}


// MPI funciones de comunicacion colectiva de SEMILLAS, se usa scatter de mpi
static void ConfigureRandomNumbers(void) {

         
	//int error;
	unsigned long long seed;

	unsigned int nrows = NODE_THREADS; //se carga en main en un bloque de openmp
	unsigned int ncols = nprocs_mpi;   

	X = (unsigned long long *)malloc(nrows*ncols*sizeof(unsigned long long)); 
	x = (unsigned long long *)malloc(nrows*sizeof(unsigned long long));   
 
	A = (unsigned int *)malloc(nrows*ncols*sizeof(unsigned int));
	a = (unsigned int *)malloc(nrows*sizeof(unsigned int));
	
	unsigned int i;

	if(myrank_mpi ==ROOT){

		seed = (unsigned long long) SEED; //SEED=time(NULL)

		//TODO: capturar error de la funcion y escribirlo en el archivo log del ROOT
		//error = init_RNG(X, A, nrows*ncols, SAFE_PRIMES_FILENAME, seed);

		init_RNG(X, A, nrows*ncols, SAFE_PRIMES_FILENAME, seed);

	}else{
		
		for(i=0;i<nrows;i++){
			x[i]=0;
			a[i]=0;
		}

	}


	MPI_Scatter(X, nrows, MPI_UNSIGNED_LONG_LONG, x, nrows, MPI_UNSIGNED_LONG_LONG, ROOT, MPI_COMM_WORLD);
	MPI_Scatter(A, nrows, MPI_UNSIGNED, a, nrows, MPI_UNSIGNED, ROOT, MPI_COMM_WORLD);

	//ahora cada thread accede a su semilla y a su multiplicador asi: x[thread_id] y a[thread_id]

	//return error;

}


int main(int argc, char** argv){

			

			MPI_Init(&argc, &argv);
	    		MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
			MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

	if (myrank_mpi==ROOT){//-----------------------------------------ROOT-------------------------------------------------------------------

				T=(unsigned int*) malloc(nprocs_mpi*Q*sizeof(unsigned int)); //vector para el calculo del magnetismo

	}//-------------------------------------------------------------------------------------------------------------------------------------
			
			#pragma omp parallel shared(NODE_THREADS)
			#pragma omp master
			{
				NODE_THREADS=omp_get_num_threads();////NODE_THREADS=8;

			}

			struct statpoint stat[NPOINTS] = { {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0} };
			double secs = 0.0;
			struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

			// parameters checking
			
			assert(2<=Q); 
			assert(Q<(1<<(sizeof(byte)*8))); 
			assert(TEMP_MIN<=TEMP_MAX);
			assert(DELTA_T<TMAX); 
			assert(TMAX%DELTA_T==0); 
			assert(L%2==0); // we can halve height
			assert((L*L/2)*4L<UINT_MAX); //max energy that is all spins are the same, fits into a ulong
			
			//nuevos asserts

			assert(L%NODESX==0);
			assert(L%NODESY==0);
			assert(((L/NODESX)+2)%2==0);
			
			#ifdef PROFILE_SPINFLIP
				if (myrank_mpi==ROOT){
					FILE *fp = NULL;
					char _filename[40];
					sprintf(_filename, "%s%s%i%s%i%s%i%s%i%s", "pottsHYBR_", "spinFlip_", L,"-", NODESX,"-", NODESY, "-", NODE_THREADS, ".txt");
					fp = fopen(_filename, "a"); 
					fprintf(fp, "---------------------------INICIO: PROFILE_SPINFLIP-------------------------------\n"); 
					fclose(fp);
				}
			#endif

	if (myrank_mpi==ROOT){//---------------------------ROOT--print header---and config RNG----------------------------------------------------------------
			
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
	}//-----------------------------------------------------------------------------------------------------------------------------------------------

			ConfigureRandomNumbers();


	if (myrank_mpi==ROOT){//--------------------------------------------------------------------------------------------------------------------

			// stop timer
			assert(gettimeofday(&end, NULL)==0);
			assert(timeval_subtract(&elapsed, &end, &start)==0);
			secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
			printf("# Configure RNG Time: %f\n", secs);

			// start timer
			assert(gettimeofday(&start, NULL)==0);

	}//------------------------------------------------------------------------------------------------------------------------------------------

			initNodeMesh();    				//calculo de topologia
			buildToroNodeMesh();    			//calculo de topologia
			calculateNodeNeighbors(neighbors, myrank_mpi); //calculo de topologia
			

			defineGhostCells();				//carga la definicion ghost e inicializa whites y blacks

			MPI_Barrier(MPI_COMM_WORLD);

			for (unsigned int i = 0; i < SAMPLES; i++) {
				sample(whites, blacks, stat);
			}


	
	//ACA2
	if (myrank_mpi==ROOT){//------------------------------------print results------------------------------------------------------------------------


			// stop timer
			assert(gettimeofday(&end, NULL)==0);
			assert(timeval_subtract(&elapsed, &end, &start)==0);
			secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
			printf("# Total Simulation Time: %lf\n", secs);

			printf("# Temp\tE\tE^2\tE^4\tM\tM^2\tM^4\n"); //^ es XOR-lógico

			
			for (unsigned int i=0; i<NPOINTS; i++) {
				printf ("%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",
					stat[i].t,
					stat[i].e/((double)N*SAMPLES),
					stat[i].e2/((double)N*N*SAMPLES),
					stat[i].e4/((double)N*N*N*N*SAMPLES),
					stat[i].m/SAMPLES,
					stat[i].m2/SAMPLES,
					stat[i].m4/SAMPLES);
			}//del for
			
	}//--------------------------------------------------------------------------------------------------------------------------------------------

			//free(x);
			//free(a);

			MPI_Finalize(); 
			return 0;



}


