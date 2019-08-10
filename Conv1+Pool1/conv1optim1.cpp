#include "conv1pool1.h"

float expf(float x) {
 x = 1.0 + x / 1024;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x;
 return x;
}

float Conv_5x5(float input[25], float kernel[25]){
	int x,y;
	float result;
	float add1,add2,add3,add4,add0;
	L1:for(y = 0; y < 5; y++){
#pragma HLS PIPELINE rewind
#pragma HLS UNROLL factor=5
//		L2:for(x = 0; x < 5; x++){
//#pragma HLS PIPELINE II=2
//#pragma HLS UNROLL factor=5
//			result += input[x+y*5] * kernel[x+y*5];
		add0 += input[0+y*5] * kernel[0+y*5];
		add1 += input[1+y*5] * kernel[1+y*5];
		add2 += input[2+y*5] * kernel[2+y*5];
		add3 += input[3+y*5] * kernel[3+y*5];
		add4 += input[4+y*5] * kernel[4+y*5];
//		}
	}
result = add0+add1+add2+add3+add4;
	return result;
}

//float Conv_5x5(float input[25], float kernel[25]){
// int x,y;
// float mul_result[25];
// float result = 0;
//
// float sum4[12];
// float sum3[6];
// float sum2[3];
// float sum1[2];
//
//// L1:for(y = 0; y < 5; y++){
////  L2:for(x = 0; x < 5; x++){
////   result += input[x+y*5] * kernel[x+y*5];
////  }
//// }
// for(int i = 0; i < 25; i++)
//  {
// #pragma HLS LATENCY max=0
// #pragma HLS UNROLL
//   mul_result[i] = input[i] * kernel[i];
//  }
// sum4[0] = mul_result[0] + mul_result[1];
// sum4[1] = mul_result[2] + mul_result[3];
// sum4[2] = mul_result[4] + mul_result[5];
// sum4[3] = mul_result[6] + mul_result[7];
// sum4[4] = mul_result[8] + mul_result[9];
// sum4[5] = mul_result[10] + mul_result[11];
// sum4[6] = mul_result[12] + mul_result[13];
// sum4[7] = mul_result[14] + mul_result[15];
// sum4[8] = mul_result[16] + mul_result[17];
// sum4[9] = mul_result[18] + mul_result[19];
// sum4[10] = mul_result[20] + mul_result[21];
// sum4[11] = mul_result[22] + mul_result[23];
//
// sum3[0] = sum4[0] + sum4[1];
// sum3[1] = sum4[2] + sum4[3];
// sum3[2] = sum4[4] + sum4[5];
// sum3[3] = sum4[6] + sum4[7];
// sum3[4] = sum4[8] + sum4[9];
// sum3[5] = sum4[10] + sum4[11];
//
// sum2[0] = sum3[0] + sum3[1];
// sum2[1] = sum3[2] + sum3[3];
// sum2[2] = sum3[4] + sum3[5];
//
// sum1[0] = sum2[0] + sum2[1];
// sum1[1] = sum2[2] + mul_result[24];
// return sum1[0]+sum1[1];
//}

//kernel 5x5x6 = 25x6 = 150
void ConvLayer_1(float input[1024],float * C1_value,float * weights){
	int i_y,i_x,matrix_y,matrix_x;
	int k_num,mat_i = 0;
	top_loop:for(int k_num = 0; k_num < 6; k_num+=1){
		//TODO 内存kernel
		float matrix_2[25];
		for(mat_i = 0;mat_i<25;mat_i++){
			matrix_2[mat_i] = weights[mat_i + k_num*25];
		}
		i_y_loop:for(i_y = 0; i_y < 28; i_y++){
//#pragma HLS PIPELINE rewind

			i_x_loop:for(i_x = 0; i_x < 28; i_x++){
 #pragma HLS PIPELINE rewind
//                #pragma HLS PIPELINE II=1
//#pragma HLS UNROLL factor=5
				float matrix[25];
				int pic_value_index = i_x + i_y * 32;
				matrix_loop:for(matrix_y = 0; matrix_y <5; matrix_y++){
					caculate:for(matrix_x = 0; matrix_x <5; matrix_x++){
#pragma HLS PIPELINE rewind
#pragma HLS UNROLL factor=5
//						图片索引  0 ~ 24
						int matrix_index = matrix_x + matrix_y * 5;
//						图片像素索引 0 ~ 1024,与matrix_x,matrix_y相关,x、y=32
						int input_value_index = pic_value_index + matrix_x + matrix_y * 32;
						matrix[matrix_index] = input[input_value_index];
					}
				}
				int out_pic_index = i_x + i_y * 28 + k_num * 784;
				C1_value[out_pic_index] = Conv_5x5(matrix,matrix_2);
			}
		}
	}
}

float AvgPool_2x2(float input[4]){
	float sum1;
	float sum2[4];
	int i;
//	for(i = 0; i < 4 ; i++){
//#pragma HLS LATENCY max=0
// #pragma HLS UNROLL
//		res += input[i];
//	}
//	res /= 4;
	sum2[0]=input[0]+input[2];
	sum2[1]=input[1]+input[3];
	sum1 = sum2[0]+sum2[1];
	return sum1/4;
}

float sigmoid(float x)
{
    return (1 / (1 + expf(-x)));
}

void AvgpoolLayer_2(float input[4704],float *A2_value){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;
	for(k_num = 0; k_num < 6; k_num++){
		for(i_y = 0; i_y < 27; i_y+=2){
			for(i_x = 0;  i_x < 27; i_x+=2){
				float matrix[4];
				int index_now = i_x + i_y * 28 + k_num * 784;
				for(matrix_y = 0; matrix_y < 2; matrix_y++){
					for(matrix_x = 0; matrix_x < 2; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * 28 ;
						matrix[matrix_x + matrix_y*2] = input[input_index];
					}
				}
				A2_value[count] = sigmoid(AvgPool_2x2(matrix));
				count++;
			}
		}
	}
}

/* //kernel 5x5x6x16 = 25x6x16 =2400
void ConvLayer_3(float input[1176],float *C3_value,float * weights){
	int k_num,nk_num,i_y,i_x,matrix_x,matrix_y;
	int mat_i;
    for(nk_num = 0; nk_num < 16; nk_num++){
		for(i_y = 0; i_y < 10; i_y++){
			for(i_x = 0; i_x < 10; i_x++){
				float res = 0;
				float res_total_6 = 0;
				float matrix[25];
				int index_now = i_x + i_y * 10 + nk_num * 100;
				for(k_num = 0; k_num < 6; k_num++){
					float matrix_2[25];
					for(mat_i = 0;mat_i<25;mat_i++){
						int weights_index = mat_i + k_num*25 + (nk_num+1)*150;
						matrix_2[mat_i] = weights[weights_index];
					}
					for(matrix_y = 0; matrix_y <5; matrix_y++){
						for(matrix_x = 0; matrix_x <5; matrix_x++){
							int matrix_index = matrix_x + matrix_y * 5;
							int input_value_index = index_now + matrix_x + matrix_y * 14;
							matrix[matrix_index] = input[input_value_index];
						}
					}
					res_total_6 += Conv_5x5(matrix,matrix_2);
				}
				C3_value[index_now] = res_total_6;
			}
		}
	}
}

void AvgpoolLayer_4(float input[1600],float *A4_value){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;
	for(k_num = 0; k_num < 16; k_num++){
		for(i_y = 0; i_y < 10; i_y+=2){
			for(i_x = 0;  i_x < 10; i_x+=2){
				float matrix[4];
				int index_now = i_x + i_y * 10 + k_num * 100;
				for(matrix_y = 0; matrix_y < 2; matrix_y++){
					for(matrix_x = 0; matrix_x < 2; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * 10 ;
						matrix[matrix_x + matrix_y*2] = input[input_index];
					}
				}
				A4_value[count] = sigmoid(AvgPool_2x2(matrix));
				count++;
			}
		}
	}
}

//kernel 400x120 = 48000
void FullyConnLayer_5(float input[400],float *F5_value,float * weights){
	int i_y,i_x;
	for(i_y = 0; i_y < 120; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 400; i_x++){
			int index = i_x + i_y * 400;
			res += input[i_x] * weights[index + 2550];
		}
		F5_value[i_y] = res;
	}
}
//kernel 84x120 = 10080
void FullyConnLayer_6(float input[120],float *F6_value,float * weights){
	int i_y,i_x;
	for(i_y = 0; i_y < 84; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 120; i_x++){
			int index = i_x + i_y * 120;
			res += input[i_x] * weights[index + 50550];
		}
		F6_value[i_y] = res;
	}
}

//kernel 10x120 = 1200
void FullyConnLayer_7(float input[84],float *F6_value,float * weights){
	int i_y,i_x;
	for(i_y = 0; i_y < 10; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 84; i_x++){
			int index = i_x + i_y * 84;
			res += input[i_x] * weights[index + 60630];
		}
		F6_value[i_y] = res;
	}
}

int Softmax_1_8(float input[10],float *probability,float *res){
	int index;
	float sum = 0;
	for(index = 0; index < 10; index++ ){
		probability[index] = expf(input[index]/1000);
		sum += probability[index];
	}
	int max_index = 0;
	for(index = 0; index < 10; index++ ){
			res[index] = probability[index]/sum;
			float res1 = res[index];
			float res2 = res[max_index];
			if(res1 > res2){
				max_index = index;
			}
	}
	return max_index; 
}*/


void LeNet1(volatile float *addrMaster,volatile float *addrSlave){
#pragma HLS INTERFACE m_axi depth=1174/* 62855 */ port=addrMaster offset=slave bundle=input
#pragma HLS INTERFACE m_axi depth=1176 port=addrSlave offset=slave bundle=output
//#pragma HLS INTERFACE s_axilite port=addrSlave bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	// 32x32 iamge
	float photo[1024];
	//layer1 weights  5x5x6 = 25x6 = 150
	//layer3 weights  5x5x6x16 = 25x6x16 =2400
	//layer5 weights 400x120 = 48000
	//layer6 weights 84x120 = 10080
	//layer7 weights 10x120 = 1200
	float data[1174];
	//The output of each layer
	float C1_value[4704];
	float A2_value[1176];
/* 	float C3_value[1600];
	float A4_value[400];
	float F5_value[120];
	float F6_value[84];
	float F7_value[10]; */

	float probability[10];
	float res[10];
	int loop1_i;
	//memory copy from BRAM to FPGA's RAM
	memcpy(data,(const float*)addrMaster,1174*sizeof(float));
	//get the image data
	for(loop1_i = 0; loop1_i<1024; loop1_i++){
		photo[loop1_i] = data[loop1_i+150];
	}
	//calulation of each layer
	ConvLayer_1(photo,C1_value,data);                  //122,304   FLOPs
	AvgpoolLayer_2(C1_value,A2_value);                 //5,880
	memcpy((float*)addrSlave,(const float*)A2_value,1176*sizeof(float));
	/* 	ConvLayer_3(A2_value,C3_value,data);               //151,600
	AvgpoolLayer_4(C3_value,A4_value);                 //2,000
	FullyConnLayer_5(A4_value,F5_value,data);          //48,120 全连层计算量不大但是参数使用非常�?
	FullyConnLayer_6(F5_value,F6_value,data);          //10,080
	FullyConnLayer_7(F6_value,F7_value,data);          //840
	*r = Softmax_1_8(F7_value,probability,res);        // */
}
