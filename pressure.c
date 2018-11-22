#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "md_struct.h"
#include "pressure.h"

// calc contribution in equilibrated state
// this roughly corresponds to (maxtime - 1200000) timesteps
// which is the interval from [500-300, 500] ps = [200, 500] ps

// DELTA_T is set in ptensor.c according to pressure
unsigned int DELTA_T;
#define LATTICE_ATOMS 1296
static int max_traj = 20;
// constants
extern const double kB;
extern const double e0;
extern const double pi;
extern const double au;
extern const double atm;
const double one = 1.0;
// double atm_inv = one /atm;
#define atm_inv 1.0 / atm
extern const double NA;
// lattice properties
extern const double area;
extern const double lattice_const;
extern double zSurface;
extern double x_max; // lattice_const * 9.0 * sqrt(3.0) / sqrt(2.0);
extern double y_max; // lattice_const * 12. / sqrt(2.0);
extern double BoxHeight; // zSurface + 60
extern double Boundary;

extern double max_traj_inv;
extern double area_inv;
// static double c_ln2 = 0.69314718055994530941723212145817656807550;

double m;

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
  double fz = f * dz / r; // calc z component

  return fz * e0 * 1.0e10; // conversion to J / m
}

double force_z_born(double xi, double xj, double yi, double yj, double zi, double zj, double *params){
  double dx = xi - xj;
  double dy = yi - yj;
  double dz = zi - zj;
  double r = sqrt(dx*dx + dy*dy + dz*dz);
  double r_ = 1.0/r;
  double r_2 = 1.0/(r*r);

  double r_6 = r_2 * r_2 * r_2;
  double r_7 = r_6 * r_;
  double r_9 = r_7 * r_2;

  double A = params[0];
  double rho = params[1];
  // double sigma = params[2];
  double C = params[3];
  double D = params[4];

  double f =  A / rho * exp(-r/rho) - 6.0 * C * r_7 + 8.0 * D * r_9;
  double fz = f * dz / r;

  return fz * e0 * 1.0e10; // conversion to J / m from ev/AA (default metal units in lammps)
}

// may want to optimize sign function by bitwise shifting to obtain the leading (pseudo-sign) bit
inline int sign(float n)
{
  int num = (n < 0) ? -1 : 1;
  return num;
}

inline int within(double x, double a, double b){
  if (a <= x && x < b){
    return 1;
  }else{
    return 0;
  }
}
// include width of dz

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

  }else if (!within(zi, z, z+dz) && within(zj, z, z+dz)){
    if (zi <= z){
      double b = z - zi;
      assert(r >= b);
      double a = r - b;
      return a / dz;
    }else if (zi > z){
      double b = zi - (z+dz);
      assert(r >= b);
      double a = r - b;
      return a / dz;
    }else{
      printf("something mysterious is going on!\n");
      return -99999.9;
    }
  }else if (!within(zi, z, z+dz) && !within(zj, z, z+dz)){
    return dz;
  }else{
      printf("Nuts!\n");
      return 0.0;
  }
}

double surface_pressure_component(ar *a, Au *au, int time, int traj, int start, int num, int z, double delta_z, double *params, int *checkpoint){
  int i,j;
  int si, sj;
  double xi,yi,zi;
  double dx, dy, dz;
  double r;
  double xj,yj,zj;
  int N = au->n;
  double fz;
  double weight;
  double pz = 0.0;
  int ctr = 0;
  double cutoff = 12.0;
  for(i = start; i < num; i++){
    if(a->step[i] == time && a->traj[i] == traj && a->z[i] < cutoff){
      if(ctr == 0){
        *checkpoint = i;
        ctr++;
      }
      zi = a->z[i];
      si = sign(zi - z);
      xi = a->x[i];
      yi = a->y[i];
      for(j = 0; j < N; j++){
        zj = au->z[j];
        sj = sign(zj - z);
        if(si == sj) continue;

        xj = au->x[j];
        yj = au->y[j];
        dx = xi - xj;
        dy = yi - yj;
        dz = zi - zj;
        r = sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 2.0){
          printf("surface critical distance\n");
          printdata_i(a, i);
          printdataAu_i(au, j);
        }
        fz = force_z_born( xi, xj,
                      yi, yj,
                      zi, zj,
                      params);
        weight = weight_dz(zi, zj, z, delta_z);
        if (weight < 0){
            printf("Weight cannot be negative!\n");
        }
        // division by area and other prefactors is done later
        pz += fz * (si - sj) * weight;
      }
    } else if ((a->step[i] > time && a->traj[i] == traj) || a->traj[i] > traj) break;
  }
  return pz;
}

// caution! sets pz to 0
double pressure_component(ar* a, int time, int traj, int start, int num, double z, double delta_z, double epsilon, double sigma, int *checkpoint){
  // input: particle struct: a
  //        number of particles in current frame (i.e. current trajectory and timestep): num
  //        returns pressure component: pz (shall be from pzz struct)
  //        current height value for pressure tensor: z

  //TODO add weighting factor due to contribution inside [z, z+dz]
  int i,j;
  double xi, yi, zi;
  double xj, yj, zj;
  double dx, dy, dz, r;
  double fz;
  double si, sj;
  double weight;
  double pz = 0.0;
  int ctr = 0;
  for(i = start; i < num; i++){
    if (a->step[i] == time && a->traj[i] == traj){
      if (ctr == 0){
        *checkpoint = i;
        ctr++;
      }
      zi = a->z[i];
      si = sign(zi - z);
      xi = a->x[i];
      yi = a->y[i];
      for(j = start; j < num; j++){
        if (a->step[j] == time && a->traj[j] == traj){
          zj = a->z[j];
          sj = sign(zj - z);

          if(si == sj) continue;
          // printf("normale\n");

          xj = a->x[j];
          yj = a->y[j];

          dx = xi - xj;
          dy = yi - yj;
          dz = zi - zj;
          r = sqrt(dx*dx + dy*dy + dz*dz);
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
          weight = weight_dz(zi, zj, z, delta_z);
          if (weight < 0){
              printf("Weight cannot be negative!\n");
          }
          // division by area and other prefactors is done later
          pz += fz * (si - sj) * weight;
        } else if ((a->step[j] > time && a->traj[j] == traj) || a->traj[j] > traj){
          break;
        }
      }
    } else if ((a->step[i] > time && a->traj[i] == traj) || a->traj[i] > traj) {
      break;
    }
  }
  return pz;
}

double kinetic_pressure_component(
  ar* a, pzz* p, double m, double z, double time, int traj, int start, int *checkpoint, int num)
{
  int i;
  int ctr = 0;
  double pz = 0.0;
  double dz = p->dz;
  double inv_dz = 1.0 / dz;
  double vz;
  for (i = start; i < num; i++){
    if (a->traj[i] == traj && a->step[i] == time){
      if (ctr == 0){
        *checkpoint = i;
        ctr++;
      }
      vz = a->vz[i] * 100.0; // conversion to m/s

      // fct 'within' constitutes delta function and here it has units of 1/AA=1e10 m^-1
      pz += m*vz*vz * within(a->z[i], z, z+dz) * 1.0e10 * inv_dz;
    } else if ((a->step[i] > time && a->traj[i] == traj) || a->traj[i] > traj){
      break;
    }
  }
  return pz;
}

void surface_pressure_tensor(ar *a, Au *au, pzz *p, int time, int traj, double *ArAuParams){
  int k,t;
  double dz = p->dz;
  double indexSurface = Boundary * 3.0 / dz;
  int start;
  int n = a->n;
  int checkpoint = 0;
  start = 0;

  for (t = time - (DELTA_T * STW); t < time; t+=STW){
    // split the calculation into the most interesting part
    for (k = 0; k < (int)indexSurface; k++){
        p->pz_s[k] += surface_pressure_component(a, au, t, traj, start, n, p->grid[k],
                        dz, ArAuParams, &checkpoint) / ((double) DELTA_T);
    }
    start = checkpoint;
  }
  return;
}

int pressure_tensor(ar* a, pzz* p, int time, int traj, double epsilon, double sigma){
  int k,t;
  int l = p->l;
  double dz = p->dz;
  int start;
  int n = a->n;
  int checkpoint = 0;
  start = 0;

  for (t = time - (DELTA_T * STW); t < time; t+=STW){
    // printf("\t time: %d\n", t);
    for (k = 0; k < l; k++){
      p->pz_u[k] += pressure_component(a, t, traj, start, n, p->grid[k], dz, epsilon, sigma, &checkpoint) / (DELTA_T);
    }
    start = checkpoint;

  }
  return 0;
}

void kinetic_pressure_tensor(ar* a, pzz* p, int time, int traj, double m){
  int k, t;
  int prev = 0;
  int l = p->l;
  // double dz = p->dz;
  int n = a->n;
  // double indexSurface = Boundary * 3.0 / dz;
  // double indexBulk = 40 / dz;
  int checkpoint = 0;
  for (t = time - (DELTA_T * STW); t < time; t+=STW){
    for (k = 0; k < l; k++){
      p->pz_k[k] += kinetic_pressure_component(a, p, m, p->grid[k], t, traj, prev, &checkpoint, n) / (DELTA_T);
    }

    prev = checkpoint;
    // printf("thread %d, kinetic component start point: %d\n", omp_get_thread_num(), prev);
  }
}

void calc_pressure(ar* a, pzz* p, int time, double u, double rho,
  double epsilonArAr, double sigmaArAr, double *ArAuParams, double m){
  int l = p->l;
  // int n = a->n;
  double dz = p->dz;
  int i;
  double tstart, tend;
  tstart = omp_get_wtime();
  // int max_traj_loc = 8;
  #pragma omp parallel
  {
    // int ostart = omp_get_thread_num() * max_traj / omp_get_num_threads();
    // int oend = (omp_get_thread_num() + 1) * max_traj / omp_get_num_threads();
    // ostart = 0;
    // oend = 20;
    int t;
    // declare private variables for better parallelization
    // sounds good, doesn't work
    int time_loc = time;
    double m_loc = m;
    double epsilon_loc = epsilonArAr;
    double sigma_loc = sigmaArAr;
    double *params_loc = ArAuParams;
    pzz *p_loc = new_pzz(l);
    make_pgrid(p_loc, 0.0, (double)l*dz);

    for(i = 0; i < l; i++){
      p_loc->pz_u[i] = 0.0;
      p_loc->pz_s[i] = 0.0;
      p_loc->pz_k[i] = 0.0;
    }

    Au *au_loc = new_Au(LATTICE_ATOMS);
    make_lattice(au_loc);
  #pragma omp for schedule(dynamic,1)
  // each thread does one trajectory
  // since the data for different trajectories at the same time
  // is far apart in the struct (and in the input file)
  // this should still work fine and no false sharing should occur
    for(t = 0; t < max_traj; t++){
      printf("\tTrajectory %d, Thread %d\n", t, omp_get_thread_num());
      pressure_tensor(a, p_loc, time_loc, t, epsilon_loc, sigma_loc);
      kinetic_pressure_tensor(a, p_loc, time_loc, t, m_loc);
      surface_pressure_tensor(a, au_loc, p_loc, time_loc, t, params_loc);
    }
    // del_ar(a_loc);
    del_Au(au_loc);

    // potential
    #pragma omp critical
    {
      for(i = 0; i < l; i++){
        p->pz_u[i] += p_loc->pz_u[i] * 0.25 * area_inv * atm_inv * max_traj_inv;
        p->pz_k[i] += p_loc->pz_k[i] * area_inv * atm_inv * max_traj_inv - rho*u*u;
        p->pz_s[i] += p_loc->pz_s[i] * 0.25 * area_inv * atm_inv * max_traj_inv;
      }
    }
    del_pzz(p_loc);
  }
  tend = omp_get_wtime();
  printf("Time elapsed: %f\n", (double)(tend-tstart));
}

void write_pressure(pzz* p, const char* fname, double *params){
  FILE* f = fopen(fname, "w");
  int i,l;
  double density = params[0];
  double temp_P = params[1];
  double p_reference = density * kB * temp_P * atm_inv;
  l = p->l;
  if(f){
    fprintf(f, "z, p_u, p_k, p_s, p_ref\n");
      for(i = 0; i < l; i++){
        fprintf(f, "%lf, %le, %le, %le, %le\n", p->grid[i], p->pz_u[i], p->pz_k[i], p->pz_s[i], p_reference);
      }
      fclose(f);
  }else{
      printf("Could not open file %s\n", fname);
  }
}
