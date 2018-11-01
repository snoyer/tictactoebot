#include <boost/python.hpp>
#include "tictactoe.hpp"
#include "pythonhelpers.h"

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include "pyboostcvconverter.hpp"

using namespace boost::python;
using namespace tictactoecv;


#if (PY_VERSION_HEX >= 0x03000000)
    static void *init_ar(){
#else
    static void init_ar(){
#endif
    Py_Initialize();
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}


BOOST_PYTHON_MODULE(_tictactoe){
    init_ar();
    to_python_converter<cv::Mat, pbcvt::matToNDArrayBoostConverter>();
    pbcvt::matFromNDArrayBoostConverter();

    def("detect_grids", &detect_grids,
        (arg("img_gray"), arg("distance_tolerance")=5., arg("lsd_scale")=1., arg("angle_tolerance")=M_PI/16, arg("min_lines")=3));

    def("read_grid_symbols", &read_grid_symbols);

    def("cell_image", &cellImage);

    def("draw_grid", &drawGrid);

    class_<DrawnGrid>("DrawnGrid")
        .def_readonly("shape", &DrawnGrid::shape)
        .def_readonly("size", &DrawnGrid::size)
        .def_readonly("nodes", &DrawnGrid::nodes)
        .def_readonly("irregularity", &DrawnGrid::irregularity)
        .def_readonly("corners", &DrawnGrid::corners)
        .def("node", &DrawnGrid::node)
        .def("cell_corners", &DrawnGrid::cellCorners);

    class_<Point>("Point")
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);

    py_pair<Point, Point>();
    py_pair<size_t, size_t>();
    py_pair<double, double>();

    VECTOR_SEQ_CONV(DrawnGrid);
    VECTOR_SEQ_CONV(Point);
    VECTOR_SEQ_CONV(Segment);
}
