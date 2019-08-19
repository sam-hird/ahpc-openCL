#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
  }
}


void reduce(local float* local_sums, global float* partial_sums){
  if (get_local_id(0) == 0 && get_local_id(1) == 0) {
    float sum = 0.0f;

    for (int i=0; i<get_local_size(1); ++i) { 
      for (int j = 0; j < get_local_size(0); ++j)
        {
          sum += local_sums[i*get_local_size(0)+j];
        }
    }
    partial_sums[get_group_id(0)+get_group_id(1)*get_num_groups(0)] = sum;
  }
}

kernel void av_velocity(global t_speed* cells, 
                        global int* obstacles, 
                        int nx, 
                        int ny,
                        local  float* local_tot_u,
                        local  float* local_tot_cells,
                        global float* partial_tot_u,
                        global float* partial_tot_cells)
{

  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  /* ignore occupied cells */
  if (!obstacles[ii + jj*nx])
  {
    /* local density total */
    float local_density = 0.f;

    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += cells[ii + jj*nx].speeds[kk];
    }

    /* x-component of velocity */
    float u_x = (cells[ii + jj*nx].speeds[1]
                  + cells[ii + jj*nx].speeds[5]
                  + cells[ii + jj*nx].speeds[8]
                  - (cells[ii + jj*nx].speeds[3]
                     + cells[ii + jj*nx].speeds[6]
                     + cells[ii + jj*nx].speeds[7]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (cells[ii + jj*nx].speeds[2]
                  + cells[ii + jj*nx].speeds[5]
                  + cells[ii + jj*nx].speeds[6]
                  - (cells[ii + jj*nx].speeds[4]
                     + cells[ii + jj*nx].speeds[7]
                     + cells[ii + jj*nx].speeds[8]))
                 / local_density;
    /* accumulate the norm of x- and y- velocity components */
    int index = get_local_id(0)+get_local_id(1)*get_local_size(0);
    local_tot_u[index] = pow((u_x * u_x) + (u_y * u_y),0.5f);
    /* increase counter of inspected cells */
    local_tot_cells[index] = 1.f;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  reduce(local_tot_u, partial_tot_u); 
  reduce(local_tot_cells, partial_tot_cells); 
}

kernel void timestep(global t_speed* cells, 
                global int* obstacles, 
                int nx, 
                int ny,
                local  float* local_tot_u,
                local  float* local_tot_cells,
                global float* partial_tot_u,
                global float* partial_tot_cells,
                global t_speed* tmp_cells,
                float omega)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float c_sq_inv    = 1.f  / c_sq;
  const float c_sq_sq_inv = 0.5f * c_sq_inv * c_sq_inv;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  /* if the cell contains an obstacle */
  if (obstacles[ii + jj*nx])
  {
    //-----------propagate + rebound---------------
    /* determine neighbours with wrap around */
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
    /* propagate densities from neighbouring cells, following
    ** mirrored directions of travel and writing into
    ** scratch space grid */
    tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
    tmp_cells[ii + jj*nx].speeds[1] = cells[x_e + jj*nx].speeds[3]; /* west */
    tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_n*nx].speeds[4]; /* south */
    tmp_cells[ii + jj*nx].speeds[3] = cells[x_w + jj*nx].speeds[1]; /* east */
    tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_s*nx].speeds[2]; /* north */
    tmp_cells[ii + jj*nx].speeds[5] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
    tmp_cells[ii + jj*nx].speeds[6] = cells[x_w + y_n*nx].speeds[8]; /* south-east */
    tmp_cells[ii + jj*nx].speeds[7] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
    tmp_cells[ii + jj*nx].speeds[8] = cells[x_e + y_s*nx].speeds[6]; /* north-west */

  } else {
    //-------propagate + collision + av_vels-------

    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

    //avoid bad access pattern by copying all the relevant speeds into an array
    float currentSpeeds[NSPEEDS] = {cells[ii + jj*nx].speeds[0],
                                    cells[x_w + jj*nx].speeds[1],
                                    cells[ii + y_s*nx].speeds[2],
                                    cells[x_e + jj*nx].speeds[3],
                                    cells[ii + y_n*nx].speeds[4],
                                    cells[x_w + y_s*nx].speeds[5], 
                                    cells[x_e + y_s*nx].speeds[6],
                                    cells[x_e + y_n*nx].speeds[7],
                                    cells[x_w + y_n*nx].speeds[8]};

    /* local density total */
    float local_density = 0.f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += currentSpeeds[kk];
    }

    /* compute x velocity component */
    float u_x = ( currentSpeeds[1]
                - currentSpeeds[3]
                + currentSpeeds[5]
                - currentSpeeds[6]
                - currentSpeeds[7]
                + currentSpeeds[8])
                 / local_density;
    /* compute y velocity component */
    float u_y = ( currentSpeeds[2]
                - currentSpeeds[4]
                + currentSpeeds[5]
                + currentSpeeds[6]
                - currentSpeeds[7]
                - currentSpeeds[8])
                 / local_density;

    /* velocity squared */
    float u_sq = u_x * u_x + u_y * u_y;

    /*pre-compute some parts to avoid many divisions */
    float u_over_c_sq = 0.5f * u_sq * c_sq_inv;

    /* equilibrium densities */
    float d_equ[NSPEEDS];

    
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density * (1.f - u_over_c_sq);

    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f +   u_x  * c_sq_inv + (  u_x  *   u_x ) * c_sq_sq_inv - u_over_c_sq);
    d_equ[2] = w1 * local_density * (1.f +   u_y  * c_sq_inv + (  u_y  *   u_y ) * c_sq_sq_inv - u_over_c_sq);
    d_equ[3] = w1 * local_density * (1.f + (-u_x) * c_sq_inv + ((-u_x) * (-u_x)) * c_sq_sq_inv - u_over_c_sq);
    d_equ[4] = w1 * local_density * (1.f + (-u_y) * c_sq_inv + ((-u_y) * (-u_y)) * c_sq_sq_inv - u_over_c_sq);
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + ( u_x + u_y) * c_sq_inv + (( u_x + u_y) * ( u_x + u_y)) * c_sq_sq_inv - u_over_c_sq);
    d_equ[6] = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv + ((-u_x + u_y) * (-u_x + u_y)) * c_sq_sq_inv - u_over_c_sq);
    d_equ[7] = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv + ((-u_x - u_y) * (-u_x - u_y)) * c_sq_sq_inv - u_over_c_sq);
    d_equ[8] = w2 * local_density * (1.f + ( u_x - u_y) * c_sq_inv + (( u_x - u_y) * ( u_x - u_y)) * c_sq_sq_inv - u_over_c_sq);

    /* local density total */
    local_density = 0.f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      tmp_cells[ii + jj*nx].speeds[kk] = currentSpeeds[kk] + omega * (d_equ[kk] - currentSpeeds[kk]);
      local_density += tmp_cells[ii + jj*nx].speeds[kk];
    }

    /* x-component of velocity */
    u_x = ( tmp_cells[ii + jj*nx].speeds[1]
          - tmp_cells[ii + jj*nx].speeds[3]
          - tmp_cells[ii + jj*nx].speeds[6]
          - tmp_cells[ii + jj*nx].speeds[7]
          + tmp_cells[ii + jj*nx].speeds[5]
          + tmp_cells[ii + jj*nx].speeds[8])
           / local_density;
    /* compute y velocity component */
    u_y = ( tmp_cells[ii + jj*nx].speeds[2]
          - tmp_cells[ii + jj*nx].speeds[4]
          + tmp_cells[ii + jj*nx].speeds[5]
          + tmp_cells[ii + jj*nx].speeds[6]
          - tmp_cells[ii + jj*nx].speeds[7]
          - tmp_cells[ii + jj*nx].speeds[8])
           / local_density;
    /* accumulate the norm of x- and y- velocity components */
    int index = get_local_id(0)+get_local_id(1)*get_local_size(0);
    local_tot_u[index] = pow((u_x * u_x) + (u_y * u_y),0.5f);
    /* increase counter of inspected cells */
    local_tot_cells[index] = 1.f;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  reduce(local_tot_u, partial_tot_u); 
  reduce(local_tot_cells, partial_tot_cells);;
}
