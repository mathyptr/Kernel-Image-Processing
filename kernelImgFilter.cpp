#include "kernelImgFilter.h"
#include <iostream>
#include <cmath>
#include <iomanip>

#include "util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/************************************************
 *** Costruttore della classe kernelImgFilter ***
 *** Inizializza un kernel vuoto              ***
 ***********************************************/
kernelImgFilter::kernelImgFilter() {
    kernelData.clear();
    size = 0;
}

 /********************************************************
 ***** Metodo per visualizzare come matrice il filtro ****
 ********************************************************/
void kernelImgFilter::display() const {
    if (size == 0 || kernelData.empty()) {
        std::cout << "Il kernel non è inizializzato" << std::endl;
        return;
    }
    std::cout << "##############################" << std::endl;
    std::cout << "##### Matrice del Filtro #####\n";
    std::cout << "##############################" << std::endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << std::fixed << std::setprecision(4)
                      << kernelData[j + i * size] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "##############################" << std::endl;
}

/*******************************************************
********** Metodo getter dell'attributo size ***********
*******************************************************/
int kernelImgFilter::getSize() const {
    return size;
}

/*******************************************************
********** Metodo getter dell'attributo name ***********
*******************************************************/
std::string kernelImgFilter::getName() const{
    return name;
}

/*******************************************************
***** Metodo per creare un filtro di edge detection ****
*******************************************************/
bool kernelImgFilter::buildEdgeDetect() {
    std::cout << "Filtro edge detection...build..." << std::endl;
    kernelData.resize(size * size);

    float *pun;
    if(size==3)
        pun=(float *)edge_3x3;
    else
        pun=(float *)edge_5x5;

    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = *pun++;

    return true;
}

/*******************************************************
********* Metodo per creare un filtro Identità *********
*******************************************************/
bool kernelImgFilter::buildIdentity() {
    std::cout << "Filtro Identity...build..." << std::endl;
    kernelData.resize(size * size);
    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = 0;
    kernelData[size*size/2] = 1;
    return true;
}

/*******************************************************
*********** Metodo per creare un filtro sharp **********
*******************************************************/
bool kernelImgFilter::buildSharp() {
    std::cout << "Filtro sharp...build..." << std::endl;
    kernelData.resize(size * size);

    int *pun;
    if(size==3)
        pun=(int*)sharpen_3x3;
    else
        pun=(int*)sharpen_5x5;

    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = *pun++;

    return true;
}

/*******************************************************
********* Metodo per creare un filtro gaussiano ********
*******************************************************/
bool kernelImgFilter::buildGaussian() {
    std::cout << "Filtro gaussiano...build..." << std::endl;

    if (size % 2 == 0 || size < 3) {
        std::cerr << "Il Kernel deve avere una dimensione dispari e maggiore di 2" << std::endl;
        return false;
    }

    kernelData.resize(size * size);

    float *pun;
    if(size==3)
        pun=(float*)gaussian_3x3;
    else
        pun=(float*)gaussian_5x5;

    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = *pun++;
    return true;
}

/*******************************************************
********* Metodo per creare un filtro Laplaciano *******
*******************************************************/
bool kernelImgFilter::buildLaplacian() {
    std::cout << "Filtro Laplaciano...build..." << std::endl;

    kernelData.resize(size * size);

    int *pun;
    if(size==3)
        pun=(int*)laplace_3x3;
    else
        pun=(int*)laplace_5x5;

    for (auto it = begin (kernelData); it != end (kernelData); ++it)
        *it = *pun++;

    return true;
}



std::vector<float> kernelImgFilter::getKernelData() const {
    return kernelData;
}


bool kernelImgFilter::buildFilter(std::string filter,int kernel_size) {
    name=filter;
    size=kernel_size;
    bool res=false;
    if (name == GAUSSIAN_FILTER_STR ) {
        res=buildGaussian();
    }
    else if (name == SHARPEN_FILTER_STR ) {
        res=buildSharp();
    }
    else if (name == EDGE_FILTER_STR ) {
        res=buildEdgeDetect();
    }
    else if (name == LAPLACIAN_FILTER_STR ) {
        res=buildLaplacian();
    }
    else if (name == IDENTITY_FILTER_STR ) {
        res=buildIdentity();
    }
    return res;
}


