#include <net.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Object {
    cv::Rect2f box;                 // box in letterboxed input scale
    int label;
    float score;
    std::vector<float> mask_coeff;  // 32 mask coefficients
};

struct LetterBoxInfo {
    float scale;
    int pad_x;
    int pad_y;
    int new_w;
    int new_h;
};

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static cv::Mat make_letterbox(const cv::Mat& src, int target_size, LetterBoxInfo& info) {
    int src_w = src.cols;
    int src_h = src.rows;

    info.scale = std::min((float)target_size / src_w, (float)target_size / src_h);
    info.new_w = (int)std::round(src_w * info.scale);
    info.new_h = (int)std::round(src_h * info.scale);

    info.pad_x = (target_size - info.new_w) / 2;
    info.pad_y = (target_size - info.new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(info.new_w, info.new_h));

    cv::Mat out(target_size, target_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(info.pad_x, info.pad_y, info.new_w, info.new_h)));

    return out;
}

static cv::Rect box_to_original(const cv::Rect2f& box, const LetterBoxInfo& lb, const cv::Size& orig_size) {
    float x0 = (box.x - lb.pad_x) / lb.scale;
    float y0 = (box.y - lb.pad_y) / lb.scale;
    float x1 = (box.x + box.width  - lb.pad_x) / lb.scale;
    float y1 = (box.y + box.height - lb.pad_y) / lb.scale;

    x0 = std::max(0.0f, std::min((float)orig_size.width  - 1.0f, x0));
    y0 = std::max(0.0f, std::min((float)orig_size.height - 1.0f, y0));
    x1 = std::max(0.0f, std::min((float)orig_size.width  - 1.0f, x1));
    y1 = std::max(0.0f, std::min((float)orig_size.height - 1.0f, y1));

    int ix0 = (int)std::round(x0);
    int iy0 = (int)std::round(y0);
    int ix1 = (int)std::round(x1);
    int iy1 = (int)std::round(y1);

    int w = std::max(0, ix1 - ix0);
    int h = std::max(0, iy1 - iy0);

    return cv::Rect(ix0, iy0, w, h);
}

static cv::Mat mask_to_original(const cv::Mat& mask_target, const LetterBoxInfo& lb, const cv::Size& orig_size) {
    cv::Rect valid(lb.pad_x, lb.pad_y, lb.new_w, lb.new_h);

    valid.x = std::max(0, valid.x);
    valid.y = std::max(0, valid.y);
    valid.width = std::min(mask_target.cols - valid.x, valid.width);
    valid.height = std::min(mask_target.rows - valid.y, valid.height);

    cv::Mat mask_valid = mask_target(valid).clone();

    cv::Mat mask_orig;
    cv::resize(mask_valid, mask_orig, orig_size);

    return mask_orig;
}

static float rect_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;

    float uni = a.area() + b.area() - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

static std::vector<int> nms(const std::vector<Object>& objects, float iou_thres) {
    std::vector<int> order(objects.size());
    for (int i = 0; i < (int)objects.size(); ++i) order[i] = i;

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return objects[a].score > objects[b].score;
    });

    std::vector<int> keep;
    std::vector<bool> removed(objects.size(), false);

    for (int _i = 0; _i < (int)order.size(); ++_i) {
        int i = order[_i];
        if (removed[i]) continue;

        keep.push_back(i);

        for (int _j = _i + 1; _j < (int)order.size(); ++_j) {
            int j = order[_j];
            if (removed[j]) continue;

            if (rect_iou(objects[i].box, objects[j].box) > iou_thres) {
                removed[j] = true;
            }
        }
    }

    return keep;
}

static std::string class_name_from_id(int label, int num_classes) {
    // Nếu model export là 1-class thì label 0 chính là crop-row.
    if (num_classes == 1 && label == 0) return "crop-row";

    // Nếu model vẫn là COCO + crop-row thì thường crop-row ở class 80.
    if (label == 80) return "crop-row";

    return "cls=" + std::to_string(label);
}

int main() {
    system("mkdir -p output");

    const char* param_path = "model/model.ncnn.param";
    const char* bin_path   = "model/model.ncnn.bin";
    const char* image_path = "input/test.png";

    // Anh export NCNN imgsz=640 thì để 640.
    // Nếu export lại imgsz=320 thì đổi thành 320.
    const int TARGET_SIZE = 640;

    const float CONF_THRES = 0.25f;
    const float NMS_THRES = 0.45f;

    // Với YOLOv8-seg, mask coeff thường là 32.
    const int NUM_MASKS = 32;

    // -1 = tự chọn class có score cao nhất.
    // Nếu muốn ép crop-row:
    //   model 1-class: FORCE_CLASS_ID = 0
    //   model COCO+crop-row: FORCE_CLASS_ID = 80
    const int FORCE_CLASS_ID = -1;

    // Ngưỡng mask. Crop-row có thể mềm hơn person, nên dùng 0.35 dễ thấy hơn.
    const float MASK_THRES = 0.35f;

    ncnn::Net net;

    if (net.load_param(param_path) != 0) {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return -1;
    }

    if (net.load_model(bin_path) != 0) {
        std::cerr << "Failed to load model: " << bin_path << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Cannot read image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

    LetterBoxInfo lb;
    cv::Mat input_img = make_letterbox(img, TARGET_SIZE, lb);

    std::cout << "Letterbox: scale=" << lb.scale
              << ", new_w=" << lb.new_w
              << ", new_h=" << lb.new_h
              << ", pad_x=" << lb.pad_x
              << ", pad_y=" << lb.pad_y << std::endl;

    ncnn::Mat in = ncnn::Mat::from_pixels(
        input_img.data,
        ncnn::Mat::PIXEL_BGR2RGB,
        TARGET_SIZE,
        TARGET_SIZE
    );

    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    if (ex.input("in0", in) != 0) {
        std::cerr << "Failed to set input blob: in0" << std::endl;
        return -1;
    }

    ncnn::Mat out0;
    ncnn::Mat out1;

    if (ex.extract("out0", out0) != 0) {
        std::cerr << "Failed to extract out0" << std::endl;
        return -1;
    }

    if (ex.extract("out1", out1) != 0) {
        std::cerr << "Failed to extract out1" << std::endl;
        return -1;
    }

    std::cout << "out0 shape: w=" << out0.w
              << ", h=" << out0.h
              << ", c=" << out0.c
              << ", dims=" << out0.dims << std::endl;

    std::cout << "out1 shape: w=" << out1.w
              << ", h=" << out1.h
              << ", c=" << out1.c
              << ", dims=" << out1.dims << std::endl;

    int num_preds = out0.w;
    int num_attrs = out0.h;
    int num_classes = num_attrs - 4 - NUM_MASKS;

    std::cout << "num_preds=" << num_preds
              << ", num_attrs=" << num_attrs
              << ", num_classes=" << num_classes
              << ", num_masks=" << NUM_MASKS << std::endl;

    if (num_classes <= 0) {
        std::cerr << "Invalid num_classes. Check out0 shape." << std::endl;
        return -1;
    }

    std::vector<Object> proposals;

    for (int i = 0; i < num_preds; ++i) {
        float cx = out0.row(0)[i];
        float cy = out0.row(1)[i];
        float bw = out0.row(2)[i];
        float bh = out0.row(3)[i];

        int best_class = -1;
        float best_score = 0.0f;

        if (FORCE_CLASS_ID >= 0 && FORCE_CLASS_ID < num_classes) {
            best_class = FORCE_CLASS_ID;
            best_score = out0.row(4 + FORCE_CLASS_ID)[i];
        } else {
            for (int c = 0; c < num_classes; ++c) {
                float score = out0.row(4 + c)[i];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }
        }

        if (best_score < CONF_THRES) continue;

        float x0 = cx - bw * 0.5f;
        float y0 = cy - bh * 0.5f;
        float x1 = cx + bw * 0.5f;
        float y1 = cy + bh * 0.5f;

        x0 = std::max(0.0f, std::min((float)TARGET_SIZE, x0));
        y0 = std::max(0.0f, std::min((float)TARGET_SIZE, y0));
        x1 = std::max(0.0f, std::min((float)TARGET_SIZE, x1));
        y1 = std::max(0.0f, std::min((float)TARGET_SIZE, y1));

        if (x1 <= x0 || y1 <= y0) continue;

        Object obj;
        obj.box = cv::Rect2f(x0, y0, x1 - x0, y1 - y0);
        obj.label = best_class;
        obj.score = best_score;

        obj.mask_coeff.resize(NUM_MASKS);
        for (int m = 0; m < NUM_MASKS; ++m) {
            obj.mask_coeff[m] = out0.row(4 + num_classes + m)[i];
        }

        proposals.push_back(obj);
    }

    std::cout << "Proposals before NMS: " << proposals.size() << std::endl;

    std::vector<int> keep = nms(proposals, NMS_THRES);
    std::cout << "Objects after NMS: " << keep.size() << std::endl;

    cv::Mat overlay = img.clone();

    for (int idx = 0; idx < (int)keep.size(); ++idx) {
        const Object& obj = proposals[keep[idx]];

        cv::Mat mask_proto(out1.h, out1.w, CV_32FC1, cv::Scalar(0));

        for (int y = 0; y < out1.h; ++y) {
            for (int x = 0; x < out1.w; ++x) {
                float v = 0.0f;

                for (int m = 0; m < NUM_MASKS; ++m) {
                    const float* proto = out1.channel(m);
                    v += obj.mask_coeff[m] * proto[y * out1.w + x];
                }

                mask_proto.at<float>(y, x) = sigmoid(v);
            }
        }

        double min_val = 0.0;
        double max_val = 0.0;
        cv::minMaxLoc(mask_proto, &min_val, &max_val);

        cv::Mat mask_target;
        cv::resize(mask_proto, mask_target, cv::Size(TARGET_SIZE, TARGET_SIZE));

        cv::Mat mask_orig_float = mask_to_original(mask_target, lb, img.size());

        cv::Mat mask_full;
        cv::threshold(mask_orig_float, mask_full, MASK_THRES, 255, cv::THRESH_BINARY);
        mask_full.convertTo(mask_full, CV_8UC1);

        cv::Rect box_orig = box_to_original(obj.box, lb, img.size());

        cv::Mat mask_cropped = cv::Mat::zeros(mask_full.size(), CV_8UC1);
        if (box_orig.width > 0 && box_orig.height > 0) {
            mask_full(box_orig).copyTo(mask_cropped(box_orig));
        }

        int full_nonzero = cv::countNonZero(mask_full);
        int cropped_nonzero = cv::countNonZero(mask_cropped);

        std::cout << "Object " << idx
                  << ": label=" << obj.label
                  << ", score=" << obj.score
                  << ", proto_min=" << min_val
                  << ", proto_max=" << max_val
                  << ", full_mask_nonzero=" << full_nonzero
                  << ", cropped_mask_nonzero=" << cropped_nonzero
                  << std::endl;

        // Để chắc chắn thấy mảng, overlay dùng full mask.
        // Nếu muốn chỉ trong bbox, đổi mask_for_overlay = mask_cropped.
        cv::Mat mask_for_overlay = mask_full;

        cv::Scalar color(255, 0, 0);  // Blue in BGR

        cv::Mat color_layer(img.size(), CV_8UC3, color);

        cv::Mat blended;
        cv::addWeighted(overlay, 0.7, color_layer, 0.3, 0, blended);

        blended.copyTo(overlay, mask_for_overlay);

        cv::rectangle(overlay, box_orig, color, 2);

        std::string label_text = class_name_from_id(obj.label, num_classes) +
                                 " " + std::to_string(obj.score).substr(0, 4);

        cv::putText(
            overlay,
            label_text,
            cv::Point(box_orig.x, std::max(25, box_orig.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        );

        std::string full_mask_png = "output/mask_full_" + std::to_string(idx) + ".png";
        std::string cropped_mask_png = "output/mask_cropped_" + std::to_string(idx) + ".png";
        std::string mask_bin_path = "output/mask_full_" + std::to_string(idx) + ".bin";
        std::string info_path = "output/mask_" + std::to_string(idx) + "_info.txt";

        cv::imwrite(full_mask_png, mask_full);
        cv::imwrite(cropped_mask_png, mask_cropped);

        cv::Mat continuous_mask = mask_full.isContinuous() ? mask_full : mask_full.clone();
        std::ofstream fout(mask_bin_path, std::ios::binary);
        fout.write((char*)continuous_mask.data, continuous_mask.total() * continuous_mask.elemSize());
        fout.close();

        std::ofstream info(info_path);
        info << "label=" << obj.label << "\n";
        info << "name=" << class_name_from_id(obj.label, num_classes) << "\n";
        info << "score=" << obj.score << "\n";
        info << "box_x=" << box_orig.x << "\n";
        info << "box_y=" << box_orig.y << "\n";
        info << "box_w=" << box_orig.width << "\n";
        info << "box_h=" << box_orig.height << "\n";
        info << "mask_w=" << mask_full.cols << "\n";
        info << "mask_h=" << mask_full.rows << "\n";
        info << "full_mask_nonzero=" << full_nonzero << "\n";
        info << "cropped_mask_nonzero=" << cropped_nonzero << "\n";
        info << "proto_min=" << min_val << "\n";
        info << "proto_max=" << max_val << "\n";
        info.close();

        std::cout << "Saved " << full_mask_png << std::endl;
    }

    cv::imwrite("output/result_overlay.jpg", overlay);
    std::cout << "Saved output/result_overlay.jpg" << std::endl;

    return 0;
}
