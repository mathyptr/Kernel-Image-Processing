#ifndef KERNEL_IMG_FILTER_H
#define KERNEL_IMG_FILTER_H

#include <vector>
#include <string>


constexpr float SHARPEN_SURROUND = -1.0f;

constexpr float EDGE_CENTER = 8.0f;
constexpr float EDGE_SURROUND = -1.0f;


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

constexpr float gaussian1_3x3[]={0.0751,	0.1238,	0.0751,
                                0.1238,	0.2042,	0.1238,
                                0.0751,	0.1238,	0.0751
};
constexpr float gaussian1_5x5[]={0.0030,	0.0133,	0.0219,	0.0133,	0.0030,
                                0.0133,	0.0596,	0.0983,	0.0596,	0.0133,
                                0.0219,0.0983,0.1621,0.0983,0.0219,
                                0.0133,0.0596,0.0983,0.0596,0.0133,
                                0.0030,0.0133,0.0219,0.0133,0.0030
};

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

    bool Init(std::vector<float>& kernel, float centerValue, float surroundValue);
    bool buildIdentity();
    bool buildEdgeDetect();
    bool buildSharp();
    bool buildGaussian(float sigma);
    bool buildLaplacian();
    std::vector<float> kernelData;
    int size;                       
};

#endif // KERNEL_IMG_FILTER_H
