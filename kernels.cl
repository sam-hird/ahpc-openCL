#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9


kernel void accelerate_flow(global float* cells,
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
      && (cells[ii + jj* nx + (nx*ny) * 3] - w1) > 0.f
      && (cells[ii + jj* nx + (nx*ny) * 6] - w2) > 0.f
      && (cells[ii + jj* nx + (nx*ny) * 7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx + (nx*ny) * 1] += w1;
    cells[ii + jj* nx + (nx*ny) * 5] += w2;
    cells[ii + jj* nx + (nx*ny) * 8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx + (nx*ny) * 3] -= w1;
    cells[ii + jj* nx + (nx*ny) * 6] -= w2;
    cells[ii + jj* nx + (nx*ny) * 7] -= w2;
  }
}


void reduce(local float* local_sums, global float* partial_sums)
{
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

void reduce2(local int* local_sums, global int* partial_sums)
{
  if (get_local_id(0) == 0 && get_local_id(1) == 0) {
    int sum = 0;

    for (int i=0; i<get_local_size(1); ++i) { 
      for (int j = 0; j < get_local_size(0); ++j)
        {
          sum += local_sums[i*get_local_size(0)+j];
        }
    }
    partial_sums[get_group_id(0)+get_group_id(1)*get_num_groups(0)] = sum;
  }
}

kernel void av_velocity(global float* cells, 
                        global int* obstacles, 
                        int nx, 
                        int ny,
                        local  float* local_tot_u,
                        local  int* local_tot_cells,
                        global float* partial_tot_u,
                        global int* partial_tot_cells)
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
      local_density += cells[ii + jj*nx + (nx*ny) * kk];
    }

    /* x-component of velocity */
    float u_x = (cells[ii + jj*nx + (nx*ny) * 1]
                  + cells[ii + jj*nx + (nx*ny) * 5]
                  + cells[ii + jj*nx + (nx*ny) * 8]
                  - (cells[ii + jj*nx + (nx*ny) * 3]
                     + cells[ii + jj*nx + (nx*ny) * 6]
                     + cells[ii + jj*nx + (nx*ny) * 7]))
                 / local_density;
    /* compute y velocity component */
    float u_y = (cells[ii + jj*nx + (nx*ny) * 2]
                  + cells[ii + jj*nx + (nx*ny) * 5]
                  + cells[ii + jj*nx + (nx*ny) * 6]
                  - (cells[ii + jj*nx + (nx*ny) * 4]
                     + cells[ii + jj*nx + (nx*ny) * 7]
                     + cells[ii + jj*nx + (nx*ny) * 8]))
                 / local_density;
    /* accumulate the norm of x- and y- velocity components */
    int index = get_local_id(0)+get_local_id(1)*get_local_size(0);
    local_tot_u[index] = sqrt((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    local_tot_cells[index] = 1;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  reduce(local_tot_u, partial_tot_u); 
  reduce2(local_tot_cells, partial_tot_cells); 
}

kernel void timestep( global float* cells, 
                      global float* tmp_cells, 
                      global int* obstacles, 
                      int nx, 
                      int ny,
                      local  float* local_tot_u,
                      local  int* local_tot_cells,
                      global float* partial_tot_u,
                      global int* partial_tot_cells,
                      float omega)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float c_sq_inv    = 3.f;
  const float c_sq_sq_inv = 4.5f;
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  const int currentIndex = ii + jj * nx;
  /* determine neighbours with wrap around */
  const int y_n = (jj + 1) % ny;
  const int x_e = (ii + 1) % nx;
  const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* if the cell contains an obstacle */
  if (obstacles[currentIndex])
  {
    //-----------propagate + rebound---------------
    /* propagate densities from neighbouring cells, following
    ** mirrored directions of travel and writing into
    ** scratch space grid */
    tmp_cells[0 * (nx*ny) + currentIndex] = cells[0 * (nx*ny) + ii  + jj *nx]; /* central cell, no movement */
    tmp_cells[1 * (nx*ny) + currentIndex] = cells[3 * (nx*ny) + x_e + jj *nx]; /* west */
    tmp_cells[2 * (nx*ny) + currentIndex] = cells[4 * (nx*ny) + ii  + y_n*nx]; /* south */
    tmp_cells[3 * (nx*ny) + currentIndex] = cells[1 * (nx*ny) + x_w + jj *nx]; /* east */
    tmp_cells[4 * (nx*ny) + currentIndex] = cells[2 * (nx*ny) + ii  + y_s*nx]; /* north */
    tmp_cells[5 * (nx*ny) + currentIndex] = cells[7 * (nx*ny) + x_e + y_n*nx]; /* south-west */
    tmp_cells[6 * (nx*ny) + currentIndex] = cells[8 * (nx*ny) + x_w + y_n*nx]; /* south-east */
    tmp_cells[7 * (nx*ny) + currentIndex] = cells[5 * (nx*ny) + x_w + y_s*nx]; /* north-east */
    tmp_cells[8 * (nx*ny) + currentIndex] = cells[6 * (nx*ny) + x_e + y_s*nx]; /* north-west */

  } else {
    //-------propagate + collision + av_vels-------

    //avoid bad access pattern by copying all the relevant speeds into an array
    const float currentSpeed0 = cells[0 * (nx*ny) + ii  +  jj*nx];
    const float currentSpeed1 = cells[1 * (nx*ny) + x_w +  jj*nx];
    const float currentSpeed2 = cells[2 * (nx*ny) + ii  + y_s*nx];
    const float currentSpeed3 = cells[3 * (nx*ny) + x_e +  jj*nx];
    const float currentSpeed4 = cells[4 * (nx*ny) + ii  + y_n*nx];
    const float currentSpeed5 = cells[5 * (nx*ny) + x_w + y_s*nx];
    const float currentSpeed6 = cells[6 * (nx*ny) + x_e + y_s*nx];
    const float currentSpeed7 = cells[7 * (nx*ny) + x_e + y_n*nx];
    const float currentSpeed8 = cells[8 * (nx*ny) + x_w + y_n*nx];

    const float local_density = currentSpeed0 + currentSpeed1 + currentSpeed2 
                              + currentSpeed3 + currentSpeed4 + currentSpeed5 
                              + currentSpeed6 + currentSpeed7 + currentSpeed8;
    /* compute x velocity component */
    const float u_x = ( currentSpeed1
                - currentSpeed3
                + currentSpeed5
                - currentSpeed6
                - currentSpeed7
                + currentSpeed8)
                 / local_density;
    /* compute y velocity component */
    const float u_y = ( currentSpeed2
                - currentSpeed4
                + currentSpeed5
                + currentSpeed6
                - currentSpeed7
                - currentSpeed8)
                 / local_density;

    /* velocity squared */
    const float u_sq = u_x * u_x + u_y * u_y;

    /*pre-compute some parts to avoid many divisions */
    const float u_over_c_sq = 0.5f * u_sq * c_sq_inv;
    /* equilibrium densities */

    
    /* zero velocity density: weight w0 */
    const float d_equ0 = w0 * local_density * (1.f - u_over_c_sq);

    /* axis speeds: weight w1 */
    const float d_equ1 = w1 * local_density * (1.f +   u_x  * c_sq_inv + (  u_x  *   u_x ) * c_sq_sq_inv - u_over_c_sq);
    const float d_equ2 = w1 * local_density * (1.f +   u_y  * c_sq_inv + (  u_y  *   u_y ) * c_sq_sq_inv - u_over_c_sq);
    const float d_equ3 = w1 * local_density * (1.f + (-u_x) * c_sq_inv + ((-u_x) * (-u_x)) * c_sq_sq_inv - u_over_c_sq);
    const float d_equ4 = w1 * local_density * (1.f + (-u_y) * c_sq_inv + ((-u_y) * (-u_y)) * c_sq_sq_inv - u_over_c_sq);
    /* diagonal speeds: weight w2 */
    const float d_equ5 = w2 * local_density * (1.f + ( u_x + u_y) * c_sq_inv + (( u_x + u_y) * ( u_x + u_y)) * c_sq_sq_inv - u_over_c_sq);
    const float d_equ6 = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv + ((-u_x + u_y) * (-u_x + u_y)) * c_sq_sq_inv - u_over_c_sq);
    const float d_equ7 = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv + ((-u_x - u_y) * (-u_x - u_y)) * c_sq_sq_inv - u_over_c_sq);
    const float d_equ8 = w2 * local_density * (1.f + ( u_x - u_y) * c_sq_inv + (( u_x - u_y) * ( u_x - u_y)) * c_sq_sq_inv - u_over_c_sq);

    /* local density total */
    float av_density = 0.f;
    
    tmp_cells[0 * (nx*ny) + currentIndex] = currentSpeed0 + omega * (d_equ0 - currentSpeed0);
    av_density += tmp_cells[0 * (nx*ny) + currentIndex];
    tmp_cells[1 * (nx*ny) + currentIndex] = currentSpeed1 + omega * (d_equ1 - currentSpeed1);
    av_density += tmp_cells[1 * (nx*ny) + currentIndex];
    tmp_cells[2 * (nx*ny) + currentIndex] = currentSpeed2 + omega * (d_equ2 - currentSpeed2);
    av_density += tmp_cells[2 * (nx*ny) + currentIndex];
    tmp_cells[3 * (nx*ny) + currentIndex] = currentSpeed3 + omega * (d_equ3 - currentSpeed3);
    av_density += tmp_cells[3 * (nx*ny) + currentIndex];
    tmp_cells[4 * (nx*ny) + currentIndex] = currentSpeed4 + omega * (d_equ4 - currentSpeed4);
    av_density += tmp_cells[4 * (nx*ny) + currentIndex];
    tmp_cells[5 * (nx*ny) + currentIndex] = currentSpeed5 + omega * (d_equ5 - currentSpeed5);
    av_density += tmp_cells[5 * (nx*ny) + currentIndex];
    tmp_cells[6 * (nx*ny) + currentIndex] = currentSpeed6 + omega * (d_equ6 - currentSpeed6);
    av_density += tmp_cells[6 * (nx*ny) + currentIndex];
    tmp_cells[7 * (nx*ny) + currentIndex] = currentSpeed7 + omega * (d_equ7 - currentSpeed7);
    av_density += tmp_cells[7 * (nx*ny) + currentIndex];
    tmp_cells[8 * (nx*ny) + currentIndex] = currentSpeed8 + omega * (d_equ8 - currentSpeed8);
    av_density += tmp_cells[8 * (nx*ny) + currentIndex];
    

    /* x-component of velocity */
    const float u_x2 = ( tmp_cells[1 * (nx*ny) + currentIndex]
          - tmp_cells[3 * (nx*ny) + currentIndex]
          - tmp_cells[6 * (nx*ny) + currentIndex]
          - tmp_cells[7 * (nx*ny) + currentIndex]
          + tmp_cells[5 * (nx*ny) + currentIndex]
          + tmp_cells[8 * (nx*ny) + currentIndex])
           / av_density;
    /* compute y velocity component */
    const float u_y2 = ( tmp_cells[2 * (nx*ny) + currentIndex]
          - tmp_cells[4 * (nx*ny) + currentIndex]
          + tmp_cells[5 * (nx*ny) + currentIndex]
          + tmp_cells[6 * (nx*ny) + currentIndex]
          - tmp_cells[7 * (nx*ny) + currentIndex]
          - tmp_cells[8 * (nx*ny) + currentIndex])
           / av_density;
    /* accumulate the norm of x- and y- velocity components */
    int index = get_local_id(0)+get_local_id(1)*get_local_size(0);
    local_tot_u[index] = sqrt((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    local_tot_cells[index] = 1;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  reduce(local_tot_u, partial_tot_u); 
  reduce2(local_tot_cells, partial_tot_cells); 
  
}