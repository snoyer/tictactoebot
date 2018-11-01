#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "tictactoe.hpp"


typedef struct timespec chrono_time;

chrono_time chrono_gettime() {
    chrono_time t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

double chrono_diff(const chrono_time t0, const chrono_time t1) {
    const long ns = (t1.tv_sec - t0.tv_sec) * 1000000000l + t1.tv_nsec - t0.tv_nsec;
    return ns/1000000000.;
}






cv::Scalar hsv_to_bgr(double h, double s, double v){
    cv::Mat3f hsv(cv::Vec3f(h*360, s, v));
    cv::Mat3f bgr;
    cvtColor(hsv, bgr, CV_HSV2BGR);
    return cv::Scalar(*bgr[0]*255);
}

cv::Scalar random_color(size_t i, double s=1, double v=1){
    return hsv_to_bgr(fmod(i*0.618034, 1), s, v);
}


cv::Mat base_output_image(const cv::Mat& img_gray, int min=200, int max=225){
    double min0, max0;
    cv::minMaxLoc(img_gray, &min0, &max0);
    cv::Mat img;
    cv::cvtColor(((img_gray - min0) / (max0 - min0)) * (max-min) + min, img, cv::COLOR_GRAY2BGR);
    return img;
}



void demo(const cv::Mat& img, double lsd_scale, double dist_tolerance, double angle_tolerance){

    const chrono_time t1 = chrono_gettime();

    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE, lsd_scale);
    std::vector<cv::Vec4f> lsd_lines;
    lsd->detect(img_gray, lsd_lines);

    const chrono_time t2 = chrono_gettime();

    auto lines = tictactoecv::postprocess_lsd_segments(lsd_lines, dist_tolerance, angle_tolerance);

    const chrono_time t3 = chrono_gettime();

    auto line_clusters = tictactoecv::cluster_intersecting_segments(lines, dist_tolerance);

    const chrono_time t4 = chrono_gettime();

    std::vector<tictactoecv::DrawnGrid> grids;
    for(const auto& lines : line_clusters){
        auto grid = tictactoecv::detect_grid(lines, dist_tolerance, 4);
        if(grid.cols>2 && grid.rows>2)
            grids.push_back(grid);
    }

    const chrono_time t5 = chrono_gettime();

    std::vector<std::string> grids_symbols;
    for(const auto& grid : grids)
        grids_symbols.push_back(tictactoecv::read_grid_symbols(grid, img_gray));

    const chrono_time t6 = chrono_gettime();

    printf("lsd        : %fs (%ld lines)\n", chrono_diff(t1,t2), lsd_lines.size());
    printf("postprocess: %fs\n", chrono_diff(t2,t3));
    printf("clustering : %fs (%ld clusters)\n", chrono_diff(t3,t4), line_clusters.size());
    printf("grids      : %fs (%ld grids)\n", chrono_diff(t4,t5), grids.size());
    printf("symbols    : %fs\n", chrono_diff(t5,t6));
    printf("             %fs\n", chrono_diff(t1,t6));


    cv::imwrite("1-input.jpg", img_gray);


    {
        cv::Mat img_dbg = base_output_image(img_gray);

        size_t i = 0;
        for(auto& l:lsd_lines){
            cv::Scalar color = random_color(i++);
            cv::arrowedLine(img_dbg, cv::Point(l[0],l[1]), cv::Point(l[2],l[3]), color, 1, CV_AA);
        }
        cv::imwrite("2-lsd.jpg", img_dbg);
    }


    {
        cv::Mat img_dbg = base_output_image(img_gray);

        size_t i = 0;
        for(auto& l:lines){
            cv::Scalar color = random_color(i++);
            cv::arrowedLine(img_dbg, l.first, l.second, color, 1, CV_AA);
        }
        cv::imwrite("3a-lsd-clean.jpg", img_dbg);
    }


    {
        cv::Mat img_dbg = base_output_image(img_gray);

        size_t i = 0;
        for(auto& lines:line_clusters){
            if(lines.size()>2){
                cv::Scalar color = random_color(i++);
                for(auto& l:lines)
                    cv::arrowedLine(img_dbg, l.first, l.second, color, 1, CV_AA);
            }
            ++i;
        }
        cv::imwrite("3b-clusters.jpg", img_dbg);
    }


    {
        cv::Mat img_dbg = base_output_image(img_gray);

        size_t i = 0;

        for(auto grid : grids){
            cv::Scalar color = random_color(i);
            const std::string symbols = grids_symbols[i++];

            if(grid.cols!=3 || grid.rows!=3 || grid.irregularity()>.2 || std::min(grid.width(),grid.height()) < 10)
                color = cv::Scalar(100,100,100);

            tictactoecv::drawGrid(img_dbg, grid, color, 1, CV_AA);

            const size_t nu = grid.cols;
            const size_t nv = grid.rows;
            for(size_t j=0; j<nu*nv; ++j){
                char symbol = symbols[j];

                auto corners = grid.cellCorners(j);
                auto center = (corners[0]+corners[1]+corners[2]+corners[3])/4;
                double s = std::min((double)grid.width()/grid.cols, (double)grid.height()/grid.rows) / 1.4142 * .4 / 8;
                tictactoecv::Point shift(-s*8,s*8);
                {
                    std::stringstream ss;
                    ss << symbol;
                    cv::putText(img_dbg, ss.str(), center+shift, cv::FONT_HERSHEY_SIMPLEX, s, color, 3, CV_AA);
                }
            }
        }
        cv::imwrite("4-grids.jpg", img_dbg);
    }




}

int main(int argc, char** argv){

    if(argc == 2){
        cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        if(!img.data){
            printf("No image data \n");
            return -1;
        }
        double dist_tolerance = 8;
        double angle_tolerance = (12/360.)*2*M_PI;
        double lsd_scale = 1;

        demo(img, lsd_scale, dist_tolerance, angle_tolerance);

        return 0;
    }else{
        printf("usage: %s IMG\n", argv[0]);
        return 1;
    }
}
