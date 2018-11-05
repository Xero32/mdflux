#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "p_struct.h"
#include "pressure.h"

#define NO_OF_ENTRIES 10000000

int max_traj = 20;
double max_traj_inv;
double kB;
double e0;
double pi;
double au;
double atm;
double NA;
double area;
double area_inv;
double lattice_const;
double zSurface;
int prv_start = 1;
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


// caution! sets pz to 0
double pressure_component(ar* a, int time, int traj, int start, int num, double z, double delta_z, double epsilon, double sigma){
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
      weight = weight_dz(zi, zj, z, delta_z) * 1.0e-10;
      if (weight < 0){
          printf("Panik!\n");
      }
      pz += fz * (si - sj) * weight;
    }
  }
  return pz;
}

double kinetic_pressure_component(
  ar* a, pzz* p, double m, double u, double rho, double z, double time, int traj, int start)
{
  int n = a->n;
  int i;
  double pz = 0.0;
  double dz = p->dz;
  double vz;
  for (i = start; i < n; i++){
    if (a->traj[i] == traj && a->step[i] == time){
      vz = a->vz[i];
      pz += vz*vz * within(a->z[i], z, z+dz) - rho * u * u;
    }
  }
  return pz;
}

int pressure_tensor(ar* a, pzz* p, int time, int traj, int *prev, double epsilon, double sigma){
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
      p->pz_u[k] = pressure_component(a, time, traj, start, start+ctr, p->grid[k], p->dz, epsilon, sigma);
    } // check if this needs to be "+="

  (*prev) = start+ctr;
  if ((*prev) > NO_OF_ENTRIES){
    return -1;
  }else{
    return 0;
  }
}

void kinetic_pressure_tensor(ar* a, pzz* p, int time, int traj, int prev, double u, double rho, double m){
  int k;
  int l = p->l;

  for(k = 0; k < l; k++){
    p->pz_u[k] += kinetic_pressure_component(a, p, m, u, rho, p->grid[k], time, traj, prev);
  } // check if this needs to be "+="
}

void calc_pressure(ar* a, pzz* p, int time, double u, double rho, double epsilon, double sigma){
  int t, err, num;
  num = max_traj;
  for(t = 0; t < max_traj; t++){
    printf("\tTrajectory %d\n", t);
    err = pressure_tensor(a, p, time, t, &prv_start, epsilon, sigma);
    kinetic_pressure_tensor(a, p, time, t, 0, u, rho, m);
    if(err){
      num = t;
      break;
    }
  }
  int l = p->l;
  for (t = 0; t < l; t++){
    p->pz_u[t] *= 0.25 * area_inv / num;
    p->pz_k[t] *= m * area_inv * 1.0e2 / num;
  }
}

void write_pressure(pzz* p, const char* fname){
  FILE* f = fopen(fname, "w");
  int i,l;
  l = p->l;
  if(f){
      for(i = 0; i < l; i++){
        fprintf(f, "%lf, %le, %le\n", p->grid[i], p->pz_u[i], p->pz_k[i]);
      }
      fclose(f);
  }else{
      printf("Could not open file %s\n", fname);
  }
}
