#ifndef IMG_PROC_CUDA_H
#define IMG_PROC_CUDA_H

#include <vector>
#include "kernelImgFilter.h"
#include "imgProc.h"

enum class CudaMemoryType
{
    GLOBAL_MEM,    // Memoria globale 
    CONSTANT_MEM,  // Memoria costante 
    SHARED_MEM     // Memoria condivisa 
};


class ImgProcCuda
{
public:
    ImgProcCuda(ImgProc& imageProc);
    ~ImgProcCuda();
    int getWidth() const;
    int getHeight() const;
    bool saveImageToFile(const char* filepath) const;
    bool setImageData(const std::vector<float>& data, int width, int height);
    bool applyFilter(const kernelImgFilter& filter, const CudaMemoryType memType);
private:
    std::vector<float> applyFilterCore(const kernelImgFilter& filter) const;
    std::vector<float> createPaddedImg(int paddingY, int paddingX) const;

    ImgProc imgProcInput;
    ImgProc imgProc;
    std::vector<float> imgDataXXX;
    int height;
    int width;
};

#endif // IMG_PROC_H_CUDA
