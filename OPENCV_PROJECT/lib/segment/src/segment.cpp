//get_segmentaion.cpp

#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"
#include "get_segmentation.h"


void get_segmentation(float sigma, float k, int min_size, const char *input_path, const char *output_path)
{
        
    printf("loading input image.\n");
    image<rgb> *input = loadPPM(input_path);
        
    printf("processing\n");

    int num_ccs; 
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs); 
    savePPM(seg, output_path);

}
