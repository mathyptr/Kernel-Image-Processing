#ifndef KERNEL_IMG_FILTER_H
#define KERNEL_IMG_FILTER_H

#include <vector>
#include <string>


constexpr float SHARPEN_CENTER = 9.0f;
constexpr float SHARPEN_SURROUND = -1.0f;

constexpr float EDGE_CENTER = 8.0f;
constexpr float EDGE_SURROUND = -1.0f;

constexpr float LAPLACE_CENTER = 4.0f;
constexpr float LAPLACE_SURROUND = -1.0f;

class kernelImgFilter {
public:
    kernelImgFilter();

    ~kernelImgFilter() {
        kernelData.clear();
        std::vector<float>().swap(kernelData);
    }
    std::string name;
    bool buildFilter(std::string filter);

    void display() const;
    std::string getName() const;
    int getSize() const;

    std::vector<float> getKernelData() const;

private:

    bool Init(std::vector<float>& kernel, float centerValue, float surroundValue, int size);
    bool buildEdgeDetect();
    bool buildSharp();
    bool buildGaussian(int size, float sigma);
    bool buildLaplacian();

    std::vector<float> kernelData;  
    int size;                       
};

#endif // KERNEL_IMG_FILTER_H
