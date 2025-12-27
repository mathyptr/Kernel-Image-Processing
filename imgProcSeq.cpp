#include "imgProcSeq.h"
#include <png++/png.hpp>
#include <algorithm>
#include <iostream>


ImgProcSeq::ImgProcSeq(ImgProc& inputImage) : width(0), height(0) {
    imgProcInput=inputImage;
}


ImgProcSeq::~ImgProcSeq() {

}

bool ImgProcSeq::applyFilterSequential( const kernelImgFilter& filter)
{

    int kernelSize = filter.getSize();
    int radius = kernelSize / 2;
    std::vector<float> kernel = filter.getKernelData();

    auto padded = imgProcInput.createPaddedImg(radius, radius);

    int height=imgProcInput.getHeight();
    int width=imgProcInput.getWidth();

    std::vector<float> result(width * height);


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;


            for (int ky = -radius; ky <= radius; ++ky) {
                for (int kx = -radius; kx <= radius; ++kx) {

                    int px = x + kx + radius;
                    int py = y + ky + radius;


                    float pixelValue = padded[py * (width + 2 * radius) + px];

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];

                    sum += pixelValue * kernelValue;
                }
            }

            result[y * width + x] = std::min(255.0f, std::max(0.0f, sum));
        }
    }

    imgProc.setImageData(result, width, height);
    return true;
}

bool ImgProcSeq::saveImageToFile(const char* filepath) const {
    try {
        int height=imgProcInput.getHeight();
        int width=imgProcInput.getWidth();
        std::vector<float> imgData= imgProc.getImageData();
        png::image<png::gray_pixel> image(width, height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                image[y][x] = static_cast<png::gray_pixel>(imgData[y * width + x]);
            }
        }
        image.write(filepath);
        std::cout << "Immagine salvata in: " << filepath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore nel salvataggio dell'immagine: " << e.what() << std::endl;
        return false;
    }
}
