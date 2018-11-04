#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>
// Constants
double kB = 1.3806503e-23;
double e0 = 1.60217662e-19;
double pi = 3.14159265359;
double au = 1.66053904e-27;
double atm = 101325;
double NA = 6.02214086e23;
const double area = 1557e-20;
const double lattice_const = 4.08; // for gold
double x_max;
double y_max;
int max_traj = 20;
double max_traj_inv;
double area_inv;
// Lennard-Jones parameter for argon
double epsilon;
double sigma;
double zSurface = 11.8;
int prv_start = 0;

typedef struct{
  int *traj;
  int *step;
  double *x;
  double *y;
  double *z;
  double *ix;
  double *iy;
  double *vx;
  double *vy;
  double *vz;
  int n;
} ar;

typedef struct{
  double *grid;
  double *pz;
  double dz;
  int l;
} pzz;

pzz* new_pzz(int l){
  pzz *p;
  p = (pzz*) malloc(sizeof(pzz));

  p->grid = (double*) _mm_malloc(l * sizeof(double), 64);
  p->pz = (double*) _mm_malloc(l * sizeof(double), 64);
  p->l = l;
  p->dz = 0.0;

  return p;
}

void make_grid(pzz *p, double start, double end){
  int i;
  int l = p->l;
  double step = (end - start) / l;
  for(i = 0; i < l; i++){
    p->grid[i] = start + i * step;
  }
  p->dz = step;
}

void del_pzz(pzz *p){
  _mm_free(p->grid);
  _mm_free(p->pz);
  free(p);
}

ar* new_ar(int n){
  ar *a;
  a = (ar*) malloc(sizeof(ar));

  a->traj = (int*) _mm_malloc(n * sizeof(int), 64);
  a->step = (int*) _mm_malloc(n * sizeof(int), 64);

  a->x = (double*) _mm_malloc(n * sizeof(double), 64);
  a->y = (double*) _mm_malloc(n * sizeof(double), 64);
  a->z = (double*) _mm_malloc(n * sizeof(double), 64);

  a->ix = (double*) _mm_malloc(n * sizeof(double), 64);
  a->iy = (double*) _mm_malloc(n * sizeof(double), 64);

  a->vx = (double*) _mm_malloc(n * sizeof(double), 64);
  a->vy = (double*) _mm_malloc(n * sizeof(double), 64);
  a->vz = (double*) _mm_malloc(n * sizeof(double), 64);
  a->n = n;

  return a;
}

void del_ar(ar *a){
  _mm_free(a->traj);
  _mm_free(a->step);

  _mm_free(a->x);
  _mm_free(a->y);
  _mm_free(a->z);

  _mm_free(a->ix);
  _mm_free(a->iy);

  _mm_free(a->vx);
  _mm_free(a->vy);
  _mm_free(a->vz);
  free(a);
}

void readfile(const char* name, ar* a){
  FILE* f = fopen(name, "r");
  int i;
  int n = a->n;
  char line[256];
  int percent = 0;
  if(f){
    fgets(line, sizeof(line), f); // omit header
    for(i = 0; i < n; i++){
      if (i % (n / 10) == 0){
        printf("%d percent\n",percent);
        percent += 10;
      }
      fgets(line, sizeof(line), f);
      sscanf(line, "%d, %d, %*d, %lf, %lf, %lf, %lf, %lf, %*f, %lf, %lf, %lf, %*f\n",
            &a->traj[i], &a->step[i],
            &a->x[i], &a->y[i], &a->z[i],
            &a->ix[i], &a->iy[i],
            &a->vx[i], &a->vy[i], &a->vz[i]);
      a->z[i] -= zSurface;
    }
  }
  printf("%d percent\n",percent);
  fclose(f);
}

void writedata(const char* fname, ar *a, int maxline){
  FILE* f = fopen(fname, "w");
  int i;
  for(i = 0; i < maxline; i++){
    fprintf(f, "%d, %d, \
                %lf, %lf, %lf, \
                %lf, %lf, \
                %lf, %lf, %lf\n",
                a->traj[i], a->step[i],
                a->x[i], a->y[i], a->z[i],
                a->ix[i], a->iy[i],
                a->vx[i], a->vy[i], a->vz[i]);
  }
  fclose(f);
}

void printdata_i(ar *a, int i){
    printf("%d, %d, \
                %lf, %lf, %lf, \
                %lf, %lf, \
                %lf, %lf, %lf\n",
                a->traj[i], a->step[i],
                a->x[i], a->y[i], a->z[i],
                a->ix[i], a->iy[i],
                a->vx[i], a->vy[i], a->vz[i]);
}


double force_z(double xi, double xj,
          double yi, double yj,
          double zi, double zj,
          double epsilon, double sigma)
{
  double dx = xi - xj;
  double dy = yi - yj;
  double dz = zi - zj;
  double r = sqrt(dx*dx + dy*dy + dz*dz);
  double r6 = r*r*r*r*r*r;
  double r_7 = 1.0 / r6 / r;
  double r12 = r6 * r6;
  double r_13 = 1.0 / r12 / r;
  double sigma6 = sigma * sigma * sigma * sigma *sigma * sigma;
  double sigma12 = sigma6 * sigma6;
  double f = 24.0 * epsilon * (2.0 * sigma12 * r_13 - sigma6 * r_7);
  double fz = f * dz / r;

  return fz * e0 *1.0e10; //conversion to J / m
}

// may want to optimize sign function by bitwise shifting to obtain the leading (i.e. pseudo-sign) bit
int sign(float n)
{
  int num = (n < 0) ? -1 : 1;
  return num;
}

int within(double x, double a, double b){
  if (a <= x && x < b){
    return 1;
  }else{
    return 0;
  }
}

double weight_dz(double zi, double zj, double z, double dz){
  double r = fabs(zj-zi);
  if (within(zi, z, z+dz) && within(zj, z, z+dz)){
    return fabs(zi - zj) / dz;
  }else if (within(zi, z, z+dz) && !within(zj, z, z+dz)){
    if (zj <= z){
      double b = z - zj;
      assert(r >= b);
      double a = r - b;
      return a / dz;
    }else if (zj > z){
      double b = zj - (z+dz);
      assert(r >= b);
      double a = r - b;
      return a / dz;
    }else{
      printf("interesting... that should not have happened\n");
      return -99999.9;
    }
  }else if (!within(zi, z, z+dz) && !within(zj, z, z+dz)){
      return dz;
  }else{
      printf("something mysterious is going on!\n");
      return 0.0;
  }
}


// caution! sets pz to 0
double pressure_component(ar* a, int time, int traj, int start, int num, double z, double dz){
  // input: particle struct: a
  //        number of particles in current frame (i.e. current trajectory and timestep): num
  //        pressure component: pz (shall be from pzz struct)
  //        current height value for pressure tensor: z

  //TODO add weighting factor due to contribution inside [z, z+dz]
  int i,j;
  double xi, yi, zi;
  double xj, yj, zj;
  double fz;
  double si, sj;
  double weight;
  double pz = 0.0;
  // dz *= 1.0e-10;
  for(i = start; i < num; i++){
    if (a->step[i] != time || a->traj[i] != traj) continue;
    zi = a->z[i];
    si = sign(zi - z);
    xi = a->x[i];
    yi = a->y[i];
    for(j = start; j < num; j++){
      if (a->step[j] != time || a->traj[j] != traj) continue;
      zj = a->z[j];
      sj = sign(zj - z);

      if(si == sj) continue;

      xj = a->x[j];
      yj = a->y[j];
      // for debugging
      double dx = xi - xj;
      double dy = yi - yj;
      double dz = zi - zj;
      double r = sqrt(dx*dx + dy*dy + dz*dz);
      if (r < 2.0){
        printf("critical distance\n");
        printdata_i(a, i);
        printdata_i(a, j);
      }else if(r > 12.0){
        continue;
      }
      fz = force_z( xi, xj,
                    yi, yj,
                    zi, zj,
                    epsilon, sigma);
      weight = weight_dz(zi, zj, z, dz) * 1.0e-10;
      pz += fz * (si - sj) * weight;
    }
  }
  return pz;
}

void pressure_tensor(ar* a, pzz* p, int time, int traj, int *prev){
  int i,k;
  int n = a->n;
  int l = p->l;
  int ctr = 0;
  int start;
  // find useful scope of struct
  for(i = *prev; i < n; i++){
    if (a->step[i] != time || a->traj[i] != traj) continue;
    if (ctr == 0){ start = i; }
    ctr++;
  }
  printf("%d, %d\n",start, start+ctr);
    for(k = 0; k < l; k++){
      p->pz[k] = pressure_component(a, time, traj, start, start+ctr, p->grid[k], p->dz);
    }

  (*prev) = start + ctr;
}

void calc_pressure(ar* a, pzz* p, int time){
  int t;
  for(t = 0; t < max_traj; t++){
    printf("\tTrajectory %d\n", t);
    pressure_tensor(a, p, time, t, &prv_start);
  }
  int l = p->l;
  for (t = 0; t < l; t++){
    p->pz[t] *= 0.25 * area_inv;
  }
}

void write_pressure(pzz* p, const char* fname){
  FILE* f = fopen(fname, "w");
  int i,l;
  l = p->l;
  for(i = 0; i < l; i++){
    fprintf(f, "%lf, %le\n", p->grid[i], p->pz[i]);
  }
  fclose(f);
}




int main(int argc, char** argv){

  x_max = lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
  y_max = lattice_const * 12. / sqrt(2.0);
  max_traj = 20;
  max_traj_inv = 1.0 / max_traj;
  // Lennard-Jones parameter for argon
  epsilon = 1.67e-21 / e0;
  sigma = 3.4;
  area_inv = 1.0 / area;

  printf("Initializing Particle struct\n");
  int NoOfEntries = 10000000;
  ar *a;
  a = new_ar(NoOfEntries);

  printf("Initializing grid\n");
  int l = 150;
  pzz* p;
  p = new_pzz(l);
  make_grid(p, 0.0, 60.0);

  char fname[] = "/home/becker/lammps/flux/A052_TS300K_TP300K_p40datm.csv";
  printf("Reading file %s\n", fname);
  readfile(fname, a);
  // printdata("/home/becker/lammps/testc.txt", a, 10);
  // printf("%d\n", a->step[1000]);
  printf("Calculating pressure tensor\n");
  int time = 2000000;
  calc_pressure(a, p, time);

  char pname[] = "/home/becker/lammps/pressuredata.csv";
  printf("Writing pressure data to file %s\n", pname);
  write_pressure(p, pname);


  del_ar(a);
  del_pzz(p);
  return 0;

}
