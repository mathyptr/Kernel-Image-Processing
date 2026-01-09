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
    bool loadImageFromFile(const char* filepath);
    bool setImageData(const std::vector<float>& data, int width, int height);
    std::vector<float> buildPaddedImg(int paddingY, int paddingX) const;
    std::vector<float> getImageData() const;

private:
    std::vector<float> applyFilterCore(const kernelImgFilter& filter) const;
    std::vector<float> imgData;
    int height;      
    int width;              
};

#endif // IMG_PROC_H
