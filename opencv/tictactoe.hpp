#include <math.h>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

namespace tictactoecv {

using namespace tictactoecv;


std::vector<Segment> postprocess_lsd_segments(const std::vector<cv::Vec4f>& lines, double dist_tolerance, double angle_tolerance){
    const double sq_dist_tolerance = dist_tolerance * dist_tolerance;
    const double sq_2_dist_tolerance = (dist_tolerance*2) * (dist_tolerance*2);

    const auto preidcate = [angle_tolerance, dist_tolerance, sq_dist_tolerance, sq_2_dist_tolerance]
                           (const cv::Vec4f& a, const cv::Vec4f& b){

        /* ignore segments shorter than `dist_tolerance` */
        if(sq_dist(a[0],a[1], a[2],a[3]) < sq_dist_tolerance)
            return false;

        /* do AABB precheck within `dist_tolerance` */
        const double precheck_dist = dist_tolerance;
        if(std::min(b[0], b[2]) - precheck_dist > std::max(a[0], a[2]) + precheck_dist 
        || std::max(b[0], b[2]) + precheck_dist < std::min(a[0], a[2]) - precheck_dist
        || std::min(b[1], b[3]) - precheck_dist > std::max(a[1], a[3]) + precheck_dist
        || std::max(b[1], b[3]) + precheck_dist < std::min(a[1], a[3]) - precheck_dist)
            return false;
        
        /* compute angle between segments */
        const double angle = abs(angle_between(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1]));
        
        /* if angle is within tolerance and segments are pointing the same way
           we want one's start to be close to the other's end */
        if(angle < angle_tolerance && (sq_dist(a[2],a[3], b[0],b[1]) < sq_2_dist_tolerance
                                    || sq_dist(a[0],a[1], b[2],b[3]) < sq_2_dist_tolerance))
            return true;

        /* if angle is within tolerance and segments are pointing the opposite way
           we want one of the extremity to be close to the other segment */
        if(angle > M_PI-angle_tolerance && sq_dist(Point(b[0], b[1]), projection(b[0], b[1], a)) < sq_dist_tolerance)
            return true;

        return false;
    };

    /* partition segments according to above predicate and merge groups by fitting as a single segment */
    return cluster<cv::Vec4f, Segment>(lines, preidcate, [](const std::vector<cv::Vec4f>& segments){return fit_segment(segments);});
}


std::vector<std::vector<Segment>> cluster_intersecting_segments(const std::vector<Segment>& lines, double margin=0){
    return cluster<Segment, std::vector<Segment>>(lines, 
        [margin](const Segment& s1, const Segment& s2){return segments_intersect(extend_segment(s1, margin, margin),
                                                                                 extend_segment(s2, margin, margin));},
        [](const std::vector<Segment>& s){return s;});
}




std::pair<std::vector<Segment>,std::vector<Segment>> find_parallel_sets_pair(std::vector<Segment> lines, double margin=0){
    const size_t N = lines.size();

    std::vector<Segment> linesX;
    std::vector<Segment> linesY;

    if(N >= 2){

        /* sort longest first */
        std::sort(lines.begin(), lines.end(), 
            [](const Segment& a, const Segment& b){ return sq_norm(a) > sq_norm(b); });

        /* check all intersections */
        std::vector<std::set<size_t>> intersects(N);
        for(size_t i=0; i<N; ++i)
            for(size_t j=i+1; j<N; ++j){
                const Segment li = extend_segment(lines[i], margin, margin);
                const Segment lj = extend_segment(lines[j], margin, margin);
                if(segments_intersect(li, lj)){
                    intersects[i].insert(j);
                    intersects[j].insert(i);
                }
            }

        /* we want to find 2 sets of lines where:
           - the members of a set don't intersect each other
           - each item of a set intersects all the items of the other
           we test the whole set of n lines first, then the first n-1 lines, and so on */
        for(size_t n=N; n>0; --n){

            /* collect unique intersections sets */
            std::set<std::set<size_t>> sets;
            for(size_t i=0; i<n; ++i)
                sets.insert(intersects[i]);

            /* we expect to collect exactly 2 sets containing all 0..n lines between them */
            if(sets.size()==2){
                std::set<std::set<size_t>>::iterator it = sets.begin();
                const std::set<size_t>& set1 = *it; ++it;
                const std::set<size_t>& set2 = *it;
                if(set1.size() + set2.size() == n){
                    for(auto i : set1) linesX.push_back(lines[i]);
                    for(auto i : set2) linesY.push_back(lines[i]);
                    return std::make_pair(linesX, linesY);
                }
            }

            /* update precomputed intersections to remove instances of last line for next loop */
            for(auto& set:intersects)
                set.erase(set.find(n-1), set.end());
        }
    }

    return std::make_pair(linesX, linesY);
}




class DrawnGrid {
public:
    const size_t cols, rows;
    const std::vector<Point> nodes;

    DrawnGrid(): cols(0), rows(0), nodes() {}
    DrawnGrid(const std::vector<Segment>& linesX, 
              const std::vector<Segment>& linesY
    ) : cols(linesY.size()-1), 
        rows(linesX.size()-1), 
        nodes(compute_nodes(linesX, linesY))
    {}

    bool empty() const { return rows<1 || cols<1; }

    std::pair<size_t,size_t> shape() const {
        return std::make_pair(cols, rows); 
    }

    double width() const {
        return std::max(dist(node(0,0   ), node(cols,0   )),
                        dist(node(0,rows), node(cols,rows)));
    }
    double height() const {
        return std::max(dist(node(0   ,0), node(0   ,rows)),
                        dist(node(cols,0), node(cols,rows)));
    }
    std::pair<double,double> size() const {
        return std::make_pair(width(), height()); 
    }

    std::vector<Point> corners() const {
        std::vector<Point> corners;
        corners.push_back(node(0   , 0   ));
        corners.push_back(node(cols, 0   ));
        corners.push_back(node(cols, rows));
        corners.push_back(node(0   , rows));
        return corners;
    }

    std::vector<Point> cellCorners(const size_t index) const {
        std::vector<Point> corners;
        if(index < cols*rows){
            const size_t i = index%cols;
            const size_t j = index/cols;
            corners.push_back(node(i  , j  ));
            corners.push_back(node(i+1, j  ));
            corners.push_back(node(i+1, j+1));
            corners.push_back(node(i  , j+1));
        }
        return corners;
    }

    std::vector<double> spacingX() const {
        std::vector<double> spacing;
        for(size_t j=0; j<rows; ++j)
            for(size_t i=1; i<cols; ++i)
                spacing.push_back(sqrt(sq_norm(node(i-1,j)-node(i,j))));
        return spacing;
    }

    std::vector<double> spacingY() const {
        std::vector<double> spacing;
        for(size_t i=0; i<rows; ++i)
            for(size_t j=1; j<cols; ++j)
                spacing.push_back(sqrt(sq_norm(node(i,j-1)-node(i,j))));
        return spacing;
    }

    double irregularity() const {
        return std::max(std_dev(spacingX()) / width(),
                        std_dev(spacingY()) / height());
    }

    inline Point node(const size_t i, const size_t j) const {
        return nodes[i+j*(cols+1)];
    }

private:
    static std::vector<Point> compute_nodes(const std::vector<Segment>& linesX,
                                            const std::vector<Segment>& linesY){
        const size_t ni = linesY.size();
        const size_t nj = linesX.size();
        std::vector<Point> nodes(ni*nj);
        for(size_t i=0; i<ni; ++i)
            for(size_t j=0; j<nj; ++j)
                nodes[i+j*ni] = lines_intersection(linesY[i], linesX[j]);
        return nodes;
    }
};


DrawnGrid detect_grid(const std::vector<Segment>& lines, double intersection_margin=0, size_t min_lines=3){
    auto linesets = find_parallel_sets_pair(lines, intersection_margin);
    const size_t NX = linesets.first .size();
    const size_t NY = linesets.second.size();

    if(NX < 1 || NY < 1 || NX+NY < min_lines)
        return DrawnGrid();

    /* get parallel sets pair and check make sure we get the horizontal one first */
    const Segment& first = linesets.first.front();
    const bool swap = u_angle_between(first.second - first.first, Point(1,0)) > M_PI/4;
    
    std::vector<Segment>& linesX = swap? linesets.second : linesets.first ;
    std::vector<Segment>& linesY = swap? linesets.first  : linesets.second;

    Segment axisX = linesX.front();
    Segment axisY = linesY.front();

    const auto flip = [](Segment& s){
        const Point p = s.first;
        s.first = s.second;
        s.second = p;
    };
    
    /* make sure axes are going left to right and top to bottom */
    if(!same_direction(axisX, Segment(Point(0,0), Point(1,0)))) flip(axisX);
    if(!same_direction(axisY, Segment(Point(0,0), Point(0,1)))) flip(axisY);

    /* make sure all lines are going the same way as axes */
    for(auto& l:linesX) if(!same_direction(l, axisX)) flip(l);
    for(auto& l:linesY) if(!same_direction(l, axisY)) flip(l);
    
    /* sort X lines along Y axis */
    std::sort(linesX.begin(), linesX.end(), 
        [axisY](const Segment& a, const Segment& b){ return projection_t(a.first, axisY.first, axisY.second)
                                                          < projection_t(b.first, axisY.first, axisY.second); });

    /* sort Y lines along X axis */
    std::sort(linesY.begin(), linesY.end(), 
        [axisX](const Segment& a, const Segment& b){ return projection_t(a.first, axisX.first, axisX.second)
                                                          < projection_t(b.first, axisX.first, axisX.second); });

    /* check segments sticking out on each side */
    const Segment L = linesY.front();
    const Segment T = linesX.front();
    const Segment R = linesY.back();
    const Segment B = linesX.back();
    std::vector<Point> ts, ls, bs, rs;
    for(auto l:linesY){
        ts.push_back(l.first  - lines_intersection(l, T));
        bs.push_back(l.second - lines_intersection(l, B));
    }
    for(auto l:linesX){
        ls.push_back(l.first  - lines_intersection(l, L));
        rs.push_back(l.second - lines_intersection(l, R));
    }
    const auto key = [](const Point& v){return sq_norm(v);};
    const Point l = *max_by_key(ls.begin(), ls.end(), key);
    const Point t = *max_by_key(ts.begin(), ts.end(), key);
    const Point r = *max_by_key(rs.begin(), rs.end(), key);
    const Point b = *max_by_key(bs.begin(), bs.end(), key);

    /* check if the stick-out on each side is greater than tolerance */
    const double open_tolerance = 10;
    const double sq_open_tolerance = open_tolerance * open_tolerance;
    const bool open_l = sq_norm(l) > sq_open_tolerance;
    const bool open_t = sq_norm(t) > sq_open_tolerance;
    const bool open_r = sq_norm(r) > sq_open_tolerance;
    const bool open_b = sq_norm(b) > sq_open_tolerance;

    /* add lines if needed */
    if(open_l) linesY.insert(linesY.begin(), Segment(L.first + l, L.second + l));
    if(open_t) linesX.insert(linesX.begin(), Segment(T.first + t, T.second + t));
    if(open_r) linesY.push_back(             Segment(R.first + r, R.second + r));
    if(open_b) linesX.push_back(             Segment(B.first + b, B.second + b));

    return DrawnGrid(linesX, linesY);
}





std::vector<DrawnGrid> detect_grids(const cv::Mat& img_gray, const double dist_tolerance, const double lsd_scale=1, double angle_tolerance=M_PI/16, size_t min_lines=3){
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE, lsd_scale);
    std::vector<cv::Vec4f> lsd_lines;
    lsd->detect(img_gray, lsd_lines);

    std::vector<Segment> lines = postprocess_lsd_segments(lsd_lines, dist_tolerance, angle_tolerance);
    std::vector<std::vector<Segment>> line_clusters = cluster_intersecting_segments(lines, dist_tolerance);

    std::vector<DrawnGrid> grids;
    for(const auto& lines : line_clusters){
        const DrawnGrid grid = detect_grid(lines, dist_tolerance, min_lines);
        if(!grid.empty())
            grids.push_back(grid);
    }
    return grids;
}




cv::Mat cellImage(const DrawnGrid& grid, const int pos, const cv::Mat& img, const int margin=0) {
    const std::vector<Point> corners = grid.cellCorners(pos);
    cv::Mat img_cell;
    if(!corners.empty()){
        const int w = sqrt(sq_dist(corners[3],corners[2]));
        const int h = sqrt(sq_dist(corners[3],corners[0]));
        const int m = margin;

        const Point pts1[] = {corners[0], corners[1], corners[2], corners[3]};
        const Point pts2[] = {Point(-m,-m), Point(w-m,-m), Point(w-m,h-m), Point(-m,h-m)};
        const cv::Matx33d M = cv::getPerspectiveTransform(pts1, pts2);
        cv::warpPerspective(img, img_cell, M, cv::Size(w-m-m, h-m-m));
    }
    return img_cell;
}



char detect_symbol(const cv::Mat& cell_img, int r=0){
    cv::Mat img_bin;
    cv::threshold(cell_img, img_bin, 0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY_INV);
    
    int w = img_bin.cols;
    int h = img_bin.rows;

    /* erase rectangles at corners to break potential lines going around the border */
    cv::rectangle(img_bin, cv::Point(0    , 0    ), cv::Point(r, r), 0, CV_FILLED);
    cv::rectangle(img_bin, cv::Point(w-r-1, 0    ), cv::Point(w, r), 0, CV_FILLED);
    cv::rectangle(img_bin, cv::Point(w-r-1, h-r-1), cv::Point(w, h), 0, CV_FILLED);
    cv::rectangle(img_bin, cv::Point(0    , h-r-1), cv::Point(r, h), 0, CV_FILLED);
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_bin, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    const size_t cell_area = w*h;
    for(size_t ci = 0; ci < contours.size(); ci++){
        if(hierarchy[ci][3] < 0){ // top level
            const cv::Moments mmnt = cv::moments(contours[ci]);
            const Point center(mmnt.m10 / mmnt.m00,
                               mmnt.m01 / mmnt.m00);
            if(center.x > r && center.x < w-r && center.y > r && center.y < h-r){

                const double exterior_area = cv::contourArea(contours[ci]);
                std::vector<cv::Point> convex_hull;
                cv::convexHull(contours[ci], convex_hull);
                const double hull_area = cv::contourArea(convex_hull);
                if(hull_area > cell_area/40){
                    double interior_area = 0;
                    int hole_count = 0;
                    int child = hierarchy[ci][2];
                    while(child >- 1 && hierarchy[child][3] == (signed)ci){
                        interior_area += cv::contourArea(contours[child]);
                        hole_count++;
                        child = hierarchy[child][0];
                    }
                    // const double total_ratio = hull_area/cell_area;

                    const cv::RotatedRect rect = cv::minAreaRect(contours[ci]);
                    double ratio = rect.size.width/rect.size.height;

                    if((ratio>.5 && ratio<1.5) && exterior_area/hull_area < .7 && interior_area == 0)
                        return 'x';
                    
                    if(interior_area/exterior_area > .2)
                        return 'o';
                    
                    return '?';
                }
            }
        }
    }
    return ' ';
}


std::string read_grid_symbols(const DrawnGrid& grid, const cv::Mat img_gray){
    std::stringstream ss;
    for(size_t pos=0; pos<grid.cols*grid.rows; ++pos){
        cv::Mat img_cell = cellImage(grid, pos, img_gray, -2);
        char symbol = detect_symbol(img_cell, 7);
        ss << symbol;
    }
    return ss.str();
}



void drawGrid(cv::Mat& img, const DrawnGrid& grid, const cv::Scalar& color, double width=1, int type=4){
    const size_t r = grid.rows;
    const size_t c = grid.cols;
    for(size_t j=0; j<=r; ++j)
        for(size_t i=0; i<=c; ++i){
            if(i<c) cv::line(img, grid.node(i,j), grid.node(i+1,j  ), color, width, type);
            if(j<r) cv::line(img, grid.node(i,j), grid.node(i  ,j+1), color, width, type);
        }
}


} //namespace
