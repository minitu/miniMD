/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#include "types.h"

#define MAXLINE 256

int input(const char* filename, int& in_nx, int& in_ny, int& in_nz,
    MMD_float& in_t_request, MMD_float& in_rho, int& in_units,
    ForceStyle& in_forcetype, MMD_float& in_epsilon, MMD_float& in_sigma,
    std::string& in_datafile, int& in_ntimes, MMD_float& in_dt,
    int& in_neigh_every, MMD_float& in_force_cut, MMD_float& in_neigh_cut,
    int& in_thermo_nstat)
{
  FILE* fp;
  char line[MAXLINE];

  fp = fopen(filename, "r");

  if (fp == NULL) {
    printf("Failed at fopen\n");
    return -1;
  }

#if PRECISION==1
  fgets(line, MAXLINE, fp);
  fgets(line, MAXLINE, fp);
  fgets(line, MAXLINE, fp);

  if(strcmp(strtok(line, " \t\n"), "lj") == 0) in_units = 0;
  else if(strcmp(strtok(line, " \t\n"), "metal") == 0) in_units = 1;
  else {
    printf("Unknown units option in file at line 3 ('%s'). Expecting either 'lj' or 'metal'.\n", line);
    return -1;
  }

  fgets(line, MAXLINE, fp);

  if(strcmp(strtok(line, " \t\n"), "none") == 0) in_datafile = std::string();
  else {
    char* ptr = strtok(line, " \t");

    if(ptr == NULL) ptr = line;

    in_datafile = std::string(ptr);
  }

  fgets(line, MAXLINE, fp);

  if(strcmp(strtok(line, " \t\n"), "lj") == 0) in_forcetype = FORCELJ;
  else if(strcmp(strtok(line, " \t\n"), "eam") == 0) in_forcetype = FORCEEAM;
  else {
    printf("Unknown forcetype option in file at line 5 ('%s'). Expecting either 'lj' or 'eam'.\n", line);
    return -1;
  }

  fgets(line, MAXLINE, fp);
  sscanf(line, "%e %e", &in_epsilon, &in_sigma);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d %d %d", &in_nx, &in_ny, &in_nz);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d", &in_ntimes);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%e", &in_dt);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%e", &in_t_request);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%e", &in_rho);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d", &in_neigh_every);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%e %e", &in_force_cut, &in_neigh_cut);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d", &in_thermo_nstat);
  fclose(fp);
#else
#if PRECISION==2
  fgets(line, MAXLINE, fp);
  fgets(line, MAXLINE, fp);
  fgets(line, MAXLINE, fp);

  if(strcmp(strtok(line, " \t\n"), "lj") == 0) in_units = 0;
  else if(strcmp(line, "metal") == 0) in_units = 1;
  else {
    printf("Unknown units option in file at line 3 ('%s'). Expecting either 'lj' or 'metal'.\n", line);
    return -1;
  }

  fgets(line, MAXLINE, fp);

  if(strcmp(strtok(line, " \t\n"), "none") == 0) in_datafile = std::string();
  else {
    char* ptr = strtok(line, " \t");

    if(ptr == NULL) ptr = line;

    in_datafile = std::string(ptr);
  }

  fgets(line, MAXLINE, fp);

  if(strcmp(strtok(line, " \t\n"), "lj") == 0) in_forcetype = FORCELJ;
  else if(strcmp(line, "eam") == 0) in_forcetype = FORCEEAM;
  else {
    printf("Unknown forcetype option in file at line 5 ('%s'). Expecting either 'lj' or 'eam'.\n", line);
    return -1;
  }

  fgets(line, MAXLINE, fp);
  sscanf(line, "%le %le", &in_epsilon, &in_sigma);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d %d %d", &in_nx, &in_ny, &in_nz);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d", &in_ntimes);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%le", &in_dt);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%le", &in_t_request);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%le", &in_rho);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d", &in_neigh_every);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%le %le", &in_force_cut, &in_neigh_cut);
  fgets(line, MAXLINE, fp);
  sscanf(line, "%d", &in_thermo_nstat);
  fclose(fp);
#else
  printf("Invalid MMD_float size specified: crash imminent.\n");
  return -1;
#endif
#endif

  in_neigh_cut += in_force_cut;

  return 0;
}
