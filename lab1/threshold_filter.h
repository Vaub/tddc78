//
// Created by Vincent Aube on 2016-04-26.
//

#ifndef LAB1_THRESHOLD_FILTER_H
#define LAB1_THRESHOLD_FILTER_H

#include "types.h"

void thresfilter(const unsigned int threshold, const int size, 
		const Pixel* buffer, Pixel* pass_output);

#endif //LAB1_THRESHOLD_FILTER_H


