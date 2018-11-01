#include <vector>
#include <map>
#include <functional>

#include <opencv2/opencv.hpp>


namespace tictactoecv {


typedef cv::Point2f Point;
typedef std::pair<Point, Point> Segment;

const double NaN = std::nan("");
const cv::Point2f NaNPoint = cv::Point2f(NaN,NaN);



inline double dot(const double x1, const double y1, const double x2, const double y2){
    return (x1*x2) + (y1*y2);
}

inline double dot(const Point& u, const Point& v){
    return (u.x*v.x) + (u.y*v.y);
}

inline double sq_dist(const Point& a, const Point& b){
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    return dx*dx + dy*dy;
}

inline double dist(const Point& a, const Point& b){
    return sqrt(sq_dist(a,b));
}


inline double sq_dist(const double x1, const double y1, const double x2, const double y2){
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return dx*dx + dy*dy;
}

inline double sq_norm(const Point& p){
    return p.x*p.x + p.y*p.y;
}

inline double sq_norm(const cv::Vec4f& s){
    const double dx = s[2]-s[0];
    const double dy = s[3]-s[1];
    return dx*dx + dy*dy;
}

inline double sq_norm(const Segment& s){
    const double dx = s.second.x - s.first.x;
    const double dy = s.second.y - s.first.y;
    return dx*dx + dy*dy;
}

inline double sq_norm(const double x, const double y){
    return x*x + y*y;
}
inline double norm(const double x, const double y){
    return sqrt(sq_norm(x,y));
}

inline double _norm(const Point& p){
    return sqrt(p.x*p.x + p.y*p.y);
}



inline double angle_between(const double x1, const double y1, const double x2, const double y2){
    return atan2(x1*y2 - y1*x2,  // determinant
                 x1*x2 + y1*y2); // dot product
}

inline double angle_between(const Point& u, const Point& v){
    return atan2(u.x*v.y - u.y*v.x,  // determinant
                 u.x*v.x + u.y*v.y); // dot product
}

inline double u_angle_between(const Point& u, const Point& v){
    const double angle = abs(angle_between(u, v));
    return angle > M_PI/2 ? angle-M_PI : angle;
}



Point segments_intersection(const Segment& s1, const Segment& s2){
    const Point& p = s1.first;
    const Point& q = s2.first;
    const Point& r = s1.second - s1.first;
    const Point& s = s2.second - s2.first;

    const double rs = r.cross(s);
    if(abs(rs) < 1e-12)
        return NaNPoint;

    const double t = (q-p).cross(s) / rs;

    if(t<0 || t>1)
        return NaNPoint;
    else
        return p + t*r;
}


Point lines_intersection(const Segment& s1, const Segment& s2){
    const Point& p = s1.first;
    const Point& q = s2.first;
    const Point r = s1.second - s1.first;
    const Point s = s2.second - s2.first;

    const double rs = r.cross(s);
    if(abs(rs) < 1e-12)
        return NaNPoint;

    const double t = (q-p).cross(s) / rs;
    return p + t*r;
}


bool segments_intersect(const Segment& s1, const Segment& s2){
    const Point& a = s1.first;
    const Point& b = s1.second;
    const Point& c = s2.first;
    const Point& d = s2.second;
    return (((c.x-a.x)*(b.y-a.y) - (c.y-a.y)*(b.x-a.x)) * ((d.x-a.x)*(b.y-a.y) - (d.y-a.y)*(b.x-a.x)) < 0)
        && (((a.x-c.x)*(d.y-c.y) - (a.y-c.y)*(d.x-c.x)) * ((b.x-c.x)*(d.y-c.y) - (b.y-c.y)*(d.x-c.x)) < 0);
}


Segment fit_segment(const std::vector<Point>& points){
    cv::Vec4f fit;
    cv::fitLine(points, fit, cv::DIST_L2, 0, 0.01, 0.01);

    double mind = NaN;
    double maxd = NaN;
    for(const Point& p:points){
        const double d = dot(p.x-fit[2], p.y-fit[3], fit[0],fit[1]);
        if(!(d>mind)) mind = d;
        if(!(d<maxd)) maxd = d;
    }
    return Segment(Point(fit[2]+fit[0]*mind, fit[3]+fit[1]*mind),
                   Point(fit[2]+fit[0]*maxd, fit[3]+fit[1]*maxd));
}


Segment fit_segment(const std::vector<cv::Vec4f>& segments){
    std::vector<Point> points;
    for(const cv::Vec4f& s:segments){
        points.push_back(Point(s[0],s[1]));
        points.push_back(Point(s[2],s[3]));
    }
    return fit_segment(points);
}


inline double projection_t(const Point& p, const Point& a, const Point& b){
    const Point v1 = p - a;
    const Point v2 = b - a;
    return dot(v2, v1)/sq_norm(v2);
}


inline Point projection(const double x, const double y, const cv::Vec4f& s){
    const double dx = s[2]-s[0];
    const double dy = s[3]-s[1];
    const double f = dot(dx, dy, x-s[0], y-s[1])/sq_norm(s);
    return Point(s[0] + dx*f, s[1] + dy*f);
}


Segment extend_segment(const Segment& s, double l0, double l1){
    if(l0 == 0 && l1 == 0)
        return s;
    const Point v = s.second - s.first;
    const Point e = v/sqrt(sq_norm(v)); 
    return Segment(s.first-e*l0, s.second+e*l1);
};


bool same_direction(const Segment& s1, const Segment& s2){
    return dot(s1.second-s1.first, s2.second-s2.first) > 0;
}


double std_dev(const std::vector<double>& v){
    const double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
    const double m =  sum / v.size();

    double accum = 0.0;
    std::for_each (std::begin(v), std::end(v), [&](const double d) {
        accum += (d - m) * (d - m);
    });

    return sqrt(accum / (v.size()-1));
}


template<typename It, typename F>
It max_by_key(It begin, It end, F key){
    It best_t = begin;
    double best_k = key(*begin);
    for(begin++; begin<end; begin++){
        double k = key(*begin);
        if(k > best_k){
            best_k = k;
            best_t = begin;
        }
    }
    return best_t;
}


template<typename I>
class UnionFind{
    std::vector<I> tree, rank;
public:
    UnionFind(size_t N): tree(N), rank(N){
        for(size_t i=0; i<N; ++i){
            tree[i] = i;
            rank[i] = 0;
        }
    }
    I find(I p){
        I root = p;
        while(root != tree[root])
            root = tree[root];
        while(p != root){
            I newp = tree[p];
            tree[p] = root;
            p = newp;
        }
        return root;
    }

    void unite(const I x, const I y){
        const I i = find(x);
        const I j = find(y);
        if (i == j) return;
        
        if(rank[i] > rank[j]){
            tree[j] = i;
        }else{
            tree[i] = j;
            if(rank[i] == rank[j])
                rank[j]++;
        }
    }

    std::map<I, std::vector<I>> sets(){
        std::map<I, std::vector<I>> groups;
        const size_t n = tree.size();
        for(I i=0; i<n; ++i)
            groups[find(i)].push_back(i);
        return groups;
    }
};


template<typename T, typename C, typename F, typename G>
std::vector<C> cluster(const std::vector<T>& items, const F& predicate, const G& merge) {
// template<typename T, typename C>
// std::vector<C> cluster(const std::vector<T>& items, const std::function<bool(T,T)>& predicate, const std::function<C(std::vector<T>)>& merge) { // slower :(
    
    const size_t n = items.size();
    UnionFind<size_t> uf(n);

    for(size_t i=0; i<n; ++i){
        const T& item1 = items[i];
        for(size_t j=i+1; j<n; ++j){
            const T& item2 = items[j];
            if(predicate(item1, item2))
                uf.unite(i,j);
        }
    }

    std::vector<C> result;

    for(const auto& kv : uf.sets()){
        std::vector<T> cluster_items;
        for(const size_t k:kv.second)
            cluster_items.push_back(items[k]);
        result.push_back(merge(cluster_items));
    }

    return result;
}


} //namespace
