#ifndef _HELPERS_H_
#define _HELPERS_H_

#include <boost/python.hpp>
#include <boost/mpl/if.hpp>



namespace py = boost::python;

template<typename T1, typename T2>
struct PairToPythonConverter {
    static PyObject* convert(const std::pair<T1, T2>& pair)
    {
        return py::incref(py::make_tuple(pair.first, pair.second).ptr());
    }
};

template<typename T1, typename T2>
struct PythonToPairConverter {
    PythonToPairConverter()
    {
        py::converter::registry::push_back(&convertible, &construct, py::type_id<std::pair<T1, T2> >());
    }
    static void* convertible(PyObject* obj)
    {
        if (!PyTuple_CheckExact(obj)) return 0;
        if (PyTuple_Size(obj) != 2) return 0;
        return obj;
    }
    static void construct(PyObject* obj, py::converter::rvalue_from_python_stage1_data* data)
    {
        py::tuple tuple(py::borrowed(obj));
        void* storage = ((py::converter::rvalue_from_python_storage<std::pair<T1, T2> >*) data)->storage.bytes;
        new (storage) std::pair<T1, T2>(py::extract<T1>(tuple[0]), py::extract<T2>(tuple[1]));
        data->convertible = storage;
    }
};

template<typename T1, typename T2>
struct py_pair {
    py::to_python_converter<std::pair<T1, T2>, PairToPythonConverter<T1, T2> > toPy;
    PythonToPairConverter<T1, T2> fromPy;
};



template<typename T>
struct custom_vector_from_seq{
    custom_vector_from_seq(){ boost::python::converter::registry::push_back(&convertible,&construct,boost::python::type_id<std::vector<T> >()); }
    static void* convertible(PyObject* obj_ptr){
        // the second condition is important, for some reason otherwise there were attempted conversions of Body to list which failed afterwards.
        if(!PySequence_Check(obj_ptr) || !PyObject_HasAttrString(obj_ptr,"__len__")) return 0;
        return obj_ptr;
    }
    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data){
         void* storage=((boost::python::converter::rvalue_from_python_storage<std::vector<T> >*)(data))->storage.bytes;
         new (storage) std::vector<T>();
         std::vector<T>* v=(std::vector<T>*)(storage);
         int l=PySequence_Size(obj_ptr); if(l<0) abort(); /*std::cerr<<"l="<<l<<"; "<<typeid(T).name()<<std::endl;*/ v->reserve(l); for(int i=0; i<l; i++) { v->push_back(boost::python::extract<T>(PySequence_GetItem(obj_ptr,i))); }
         data->convertible=storage;
    }
};

template<typename T>
struct custom_vector_to_list{
    static PyObject* convert(std::vector<T> const& ts){
        boost::python::list list;
        typename std::vector<T>::const_iterator it;
        for(it = ts.begin(); it != ts.end(); ++it)
            list.append(*it);
        return boost::python::incref(list.ptr());
    }
};

#define VECTOR_SEQ_CONV(T) \
        custom_vector_from_seq<T>(); \
        boost::python::to_python_converter<std::vector<T>, custom_vector_to_list<T> >();
    

#endif
