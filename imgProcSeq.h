#ifndef IMG_PROC_SEQ_H
#define IMG_PROC_SEQ_H

#include <vector>
#include "kernelImgFilter.h"
#include "imgProc.h"

class ImgProcSeq
{
public:

    ImgProcSeq(ImgProc& imgProcInput);

    ~ImgProcSeq();

    int getWidth() const;

    int getHeight() const;

    int getChannels() const;

   bool saveImageToFile(const char* filepath) const;


    bool applyFilterSequential(const kernelImgFilter& filter) ;


private:

    std::vector<float> applyFilterCore(const kernelImgFilter& filter) const;

    ImgProc imgProcInput;
    ImgProc imgProc;
    int width;                     // Larghezza dell'immagine in pixel
    int height;                    // Altezza dell'immagine in pixel

};

#endif // IMG_PROC_SEQ_H