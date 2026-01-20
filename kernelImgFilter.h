#ifndef KERNEL_IMG_FILTER_H
#define KERNEL_IMG_FILTER_H

#include <vector>
#include <string>


/************************************************************
******************* Fltro Edge 3x3 e 5x5 ********************
************************************************************/
constexpr float edge_3x3[]={-1.0,-1.0,-1.0,
                            -1.0, 8.0,-1.0,
                            -1.0,-1.0,-1.0
};
constexpr float edge_5x5[]={-1.0,  -1.0, -1.0, -1.0,-1.0,
                            -1.0,  -1.0, -1.0, -1.0,-1.0,
                            -1.0,-1.0, 24.0, -1.0,-1.0,
                            -1.0,-1.0,-1.0,-1.0,-1.0,
                            -1.0,-1.0,-1.0,-1.0,-1.0
};

/************************************************************
****************** Fltro Sharpen 3x3 e 5x5 ******************
************************************************************/
constexpr int sharpen_3x3[]={-1,-1,-1,
                             -1, 9,-1,
                             -1,-1,-1
};
constexpr int sharpen_5x5[]={-1,-1,-1,-1,-1,
                             -1,-1,-1,-1,-1,
                             -1,-1,25, -1, -1,
                             -1,-1,-1,-1,-1,
                             -1,-1,-1,-1,-1
};

/************************************************************
***************** Fltro Laplaciano 3x3 e 5x5 ****************
************************************************************/
constexpr int laplace_3x3[]={0,-1,0,
                   -1,4,-1,
                   0,-1,0
                  };
constexpr int laplace_5x5[]={0,0,-1,0,0,
                   0,-1,-2,-1,0,
                   -1,-2,17, -2, -1,
                   0,-1,-2,-1,0,
                   0,0,-1,0,0
                   };

/************************************************************
***************** Fltro Gaussiano 3x3 e 5x5 *****************
************************************************************/
constexpr float gaussian_3x3[]={1.0/16,	2.0/16,	1.0/16,
                                 2.0/16,	4.0/16,	2.0/16,
                                 1.0/16,	2.0/6,	1.0/16
};
constexpr float gaussian_5x5[]={1.0/273,4.0/273,7.0/273,4.0/273,1.0/273,
                                 4.0/273,16.0/273,26.0/273,16.0/273,4.0/273,
                                 7.0/273,26.0/273,41.0/273,26.0/273,7.0/273,
                                 4.0/273,16.0/273,26.0/273,16.0/273,4.0/273,
                                 1.0/273,4.0/273,7.0/273,4.0/273,1.0/273
};

/************************************************************
****************** Classe kernelImgFilter *******************
************************************************************/
class kernelImgFilter {
public:
    kernelImgFilter();

    ~kernelImgFilter() {
        kernelData.clear();
        std::vector<float>().swap(kernelData);
    }

    std::string name;
    bool buildFilter(std::string filter, int kernel_size);
    void display() const;
    std::string getName() const;
    int getSize() const;
    std::vector<float> getKernelData() const;

private:
    bool buildIdentity();
    bool buildEdgeDetect();
    bool buildSharp();
    bool buildGaussian();
    bool buildLaplacian();
    std::vector<float> kernelData;
    int size;                       
};

#endif // KERNEL_IMG_FILTER_H
