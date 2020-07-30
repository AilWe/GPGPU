__kernel void mat_mul(const int N, __global float *input1, __global float *input2, __global float *output){
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0.0f;
    for (k=0; k < N; k++){
      tmp += input1[i*N+k] * input2[k*N+j];
    }
    output[i*N+j] = tmp;
}
