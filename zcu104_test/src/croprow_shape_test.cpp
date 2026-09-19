#include <net.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    const char* param_path = "model/model.ncnn.param";
    const char* bin_path   = "model/model.ncnn.bin";
    const char* image_path = "input/test.png";

    ncnn::Net net;

    if (net.load_param(param_path) != 0) {
        std::cerr << "Failed to load param\n";
        return -1;
    }

    if (net.load_model(bin_path) != 0) {
        std::cerr << "Failed to load model\n";
        return -1;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

    int target_size = 320;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        img.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        img.cols,
        img.rows,
        target_size,
        target_size
    );

    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    if (ex.input("in0", in) != 0) {
        std::cerr << "Failed to set input: in0\n";
        return -1;
    }

    ncnn::Mat out0;
    ncnn::Mat out1;

    int ret0 = ex.extract("out0", out0);
    int ret1 = ex.extract("out1", out1);

    if (ret0 == 0) {
        std::cout << "out0 shape: w=" << out0.w
                  << ", h=" << out0.h
                  << ", c=" << out0.c
                  << ", dims=" << out0.dims << std::endl;
    } else {
        std::cerr << "Failed to extract out0\n";
    }

    if (ret1 == 0) {
        std::cout << "out1 shape: w=" << out1.w
                  << ", h=" << out1.h
                  << ", c=" << out1.c
                  << ", dims=" << out1.dims << std::endl;
    } else {
        std::cerr << "Failed to extract out1\n";
    }

    return 0;
}
