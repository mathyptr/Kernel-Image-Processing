// image_processor.cu
#include "imgProc.h"
#include <png++/png.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16


#define MAX_KERNEL_SIZE 25

/*********************************************************************
********* Classe per la gestione delle immagini. Fornisce le **********
********* funzionalità per caricare, salvare e effettuare il **********
********* padding di un'immagine.                            **********
*********************************************************************/
ImgProc::ImgProc() : width(0), height(0) {}


ImgProc::~ImgProc() {
    imgData.clear();
    std::vector<float>().swap(imgData);
}

/*******************************************************
******** Metodo per il loading di un'immagine   ********
*******************************************************/
bool ImgProc::loadImageFromFile(const char* filepath) {
    try {
        png::image<png::gray_pixel> image(filepath);
        width = image.get_width();
        height = image.get_height();

        imgData.resize(width * height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                imgData[y * width + x] = static_cast<float>(image[y][x]);
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore nel caricamento dell'immagine: " << e.what() << std::endl;
        return false;
    }
}

/******************************************************************************
*********** Metodo per creare una versione dell'immagine con padding **********
******************************************************************************/
std::vector<float> ImgProc::buildPaddedImg(int paddingY, int paddingX) const {
    int paddedWidth = width + 2 * paddingX;
    int paddedHeight = height + 2 * paddingY;
    std::vector<float> padded(paddedWidth * paddedHeight);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded[(y + paddingY) * paddedWidth + (x + paddingX)] =
                    imgData[y * width + x];
        }
    }

    for (int y = 0; y < paddingY; ++y) {
        std::copy_n(&padded[paddingY * paddedWidth], width,
            &padded[y * paddedWidth + paddingX]);
        std::copy_n(&padded[(height + paddingY - 1) * paddedWidth], width,
            &padded[(paddedHeight - y - 1) * paddedWidth + paddingX]);
    }


    for (int y = 0; y < paddedHeight; ++y) {
        for (int x = 0; x < paddingX; ++x) {
            padded[y * paddedWidth + x] = padded[y * paddedWidth + paddingX];
            padded[y * paddedWidth + paddedWidth - 1 - x] =
                padded[y * paddedWidth + paddedWidth - paddingX - 1];
        }
    }

    return padded;
}

/*******************************************************
********** Metodo getter dell'attributo width ***********
*******************************************************/
int ImgProc::getWidth() const
{
    return width;
}

/*******************************************************
********* Metodo getter dell'attributo height **********
*******************************************************/
int ImgProc::getHeight() const
{
    return height;
}

/*******************************************************
********* Metodo setter dell'attributo imgData *********
*******************************************************/
bool ImgProc::setImageData(const std::vector<float>& data, int w, int h) {
    if (data.size() != w * h) {
        std::cerr << "Dimensioni dati non valide" << std::endl;
        return false;
    }
    width = w;
    height = h;
    imgData = data;
    return true;
}

/*******************************************************
******** Metodo getter dell'attributo imgData **********
*******************************************************/
std::vector<float> ImgProc::getImageData() const {
    return imgData;
}