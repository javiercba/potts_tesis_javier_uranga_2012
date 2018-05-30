# potts_tesis_javier_uranga_2012
Escalabilidad del Modelo de Potts en entornos distribuidos con OpenMP y MPI

autor: Javier Nicolás Uranga ( javiercba@gmai.com )

Directores: Nicolás Wolovick, Javier Blanco
Presentado como Trabajo Final Integrador de la carrera de posgrado
Especialización en Servicios y Sistemas Distribuidos
Facultad de Matemática, Astronomía y Física
Universidad Nacional de Córdoba, Argentina
Marzo de 2012
http://www.famaf.unc.edu.ar/wp-content/uploads/2014/04/8-Javier-Uranga.pdf

El Objetivo del trabajo es estudiar los modelos más populares sobre paralelización de
algoritmos de cómputo científico en multicore y clusters, resolviendo un problema de grilla y
buscando escalabilidad perfecta en el número de cores y el número de nodos.
Se tomará como base un problema Mecánica Estadística, ya modelado por el grupo GTMC
de FaMAF [3], originalmente codificado en C [7] y posteriormente llevado a CUDA [8]. Se trata de
un algoritmo Monte Carlo que utiliza la dinámica de Metropolis para simular el modelo de Potts.
La simulación estudia la transición de fase ferromagnética-paramagnética variando la
temperatura y buscando la convergencia hacia el orden del sistema en un estado de mínima energía,
utilizando una grilla de espines de dimensión L×L, para q-estados en el modelo de Potts.
Las tecnologías de paralelización utilizadas fueron: OpenMP [1] y MPI [2]. Para tal fin se
crearon seis versiones de código paralelo: dos de MPI-puro, dos OpenMP-puro y dos versiones
Híbridas que combinan las características de OpenMP y MPI. En adelante serán referidas por los
prefijos MPI, OMP e HYBR respectivamente.
El objetivo principal es obtener una implementación correcta en los dos paradigmas
(memoria compartida y pasaje de mensajes) y solucionar los problemas de concurrencia que puedan
surgir (condiciones de carrera, deadlocks). El objetivo secundario es estudiar los problemas de
escalabilidad y proponer las soluciones correspondientes buscando alcanzar la escalabilidad ideal. 


The aim of this work is to study the most popular models for scientific computing
algorithms in multicore clusters.
The starting point is a problem Statistical Mechanics, already modeled by GTMC group at
FaMAF [3], originally coded in C [7] and later moved to CUDA [8]. It is a Monte Carlo algorithm
which uses the Metropolis dynamic to simulate the Potts model.
The simulation studies the ferromagnetic-paramagnetic phase transition, varying
temperature and seeking the convergence towards an ordered state of the system, which is reached
at the minimum energy state, using a grid of size L×L spins for q-states in the Potts model.
The parallelization technologies used are: OpenMP [1] and MPI [2]. Six versions of parallel
code are generated: two pure OpenMP, two pure MPI and two Hybrids which combines the features
of OpenMP and MPI.
The main objective is to obtain a correct implementation in the two paradigms (shared
memory and message passing) and solve concurrency problems that could arise (race conditions,
deadlocks, etc). The secondary objective is to study scalability issues that arise when we look for
solutions to achieve ideal scalability.
