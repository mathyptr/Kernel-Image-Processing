#ifndef IMG_PROC_OMP_H
#define IMG_PROC_OMP_H

#include <vector>
#include "kernelImgFilter.h"
#include "imgProc.h"

class ImgProcOMP
{
public:
    ImgProcOMP(ImgProc& imgProcInput);
    ~ImgProcOMP();
    int getWidth() const;
    int getHeight() const;
    bool saveImageToFile(const char* filepath) const;
    bool applyFilter(const kernelImgFilter& filter, int numThreads) ;

private:
    std::vector<float> applyFilterCore(const kernelImgFilter& filter) const;
    ImgProc imgProcInput;
    ImgProc imgProc;
    int width;                     // Larghezza dell'immagine in pixel
    int height;                    // Altezza dell'immagine in pixel
};

#endif // IMG_PROC_OMP_H