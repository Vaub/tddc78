#include "threshold_filter.h"
typedef unsigned int uint;

void thresfilter(const uint threshold, const int size, 
		const Pixel* buffer, Pixel* pass_output){

    for(int i = 0; i < size; ++i){
        Pixel current_pix = buffer[i];
    	unsigned int sum = (unsigned int)(current_pix.r + current_pix.g + current_pix.b);
        
        current_pix.r = current_pix.g = current_pix.b = (threshold > sum) ? 0:255;
     	pass_output[i] = current_pix;
    }
}
