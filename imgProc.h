#ifndef IMG_PROC_H
#define IMG_PROC_H

#include <vector>
#include "kernelImgFilter.h"



class ImgProc
{
public:
  
    ImgProc();


    ~ImgProc();

 
    int getWidth() const;


    int getHeight() const;

  
    int getChannels() const;

 
    bool loadImageFromFile(const char* filepath);

  
    bool saveImageToFile(const char* filepath) const;

  
    bool setImageData(const std::vector<float>& data, int width, int height);

    std::vector<float> createPaddedImg(int paddingY, int paddingX) const;

    std::vector<float> getImageData() const;

//    bool ParallelFilter(ImgProc& output, const kernelImgFilter& filter, const CudaMemoryType memType);

private:

    std::vector<float> applyFilterCore(const kernelImgFilter& filter) const;

    std::vector<float> imgData;  
    int height;      
    int width;              
};

#endif // IMG_PROC_H
