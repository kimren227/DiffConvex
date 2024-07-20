#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "diff_convex.hpp"

namespace py = pybind11;

namespace diff_convex
{
  // return a tuple of array
  // array 1: vertex_plane_indices
  // array 2: intersections
  // array 3: mesh_faces
  // array 4: convex_face_indices
  std::tuple<py::array_t<int>, py::array_t<int>> convex_to_mesh(py::array_t<double> &planes_array, py::array_t<int> &convex_indices_array, py::array_t<double> &feasiable_points_array){

    py::buffer_info planes_buffer = planes_array.request();
    double *planes_ptr = (double *)planes_buffer.ptr;
    int num_planes = planes_buffer.shape[0];

    py::buffer_info convex_indices_buffer = convex_indices_array.request();
    int *convex_indices_ptr = (int *)convex_indices_buffer.ptr;

    py::buffer_info feasiable_point_buffer = feasiable_points_array.request();
    double *feasiable_point_ptr = (double *)feasiable_point_buffer.ptr;
    int num_convexs = feasiable_point_buffer.shape[0];

    std::vector<std::vector<double>> plane_parameters;
    std::vector<int> plane_indices;
    std::vector<std::vector<double>> feasiable_points;
    for(int i=0; i<num_planes; i++){
      plane_parameters.push_back(std::vector<double>{planes_ptr[4*i], planes_ptr[4*i+1], planes_ptr[4*i+2], planes_ptr[4*i+3]});
      plane_indices.push_back(convex_indices_ptr[i]);
    }
    for(int i=0; i<num_convexs; i++){
      feasiable_points.push_back(std::vector<double>{feasiable_point_ptr[3*i], feasiable_point_ptr[3*i+1], feasiable_point_ptr[3*i+2]});
    }
    std::tuple<std::vector<std::vector<int>>, 
                std::vector<std::vector<int>>> all_info = DIFFCONVEX::get_convexes_mesh(plane_parameters, plane_indices, feasiable_points);

    std::vector<std::vector<int>> vertex_plane_indices = std::get<0>(all_info);
    std::vector<std::vector<int>> mesh_faces = std::get<1>(all_info);
    // flatten vertex_plane_indices
    std::vector<int> vertex_plane_indices_vec;
    for(int i=0; i<vertex_plane_indices.size(); i++){
      for(int j=0; j<vertex_plane_indices[i].size(); j++){
        vertex_plane_indices_vec.push_back(vertex_plane_indices[i][j]);
      }
    }

    // flatten mesh_faces
    std::vector<int> mesh_faces_vec;
    for(int i=0; i<mesh_faces.size(); i++){
      for(int j=0; j<mesh_faces[i].size(); j++){
        mesh_faces_vec.push_back(mesh_faces[i][j]-1);
      }
    }

    return std::make_tuple(py::array_t<int>(vertex_plane_indices_vec.size(), vertex_plane_indices_vec.data()),
                            py::array_t<int>(mesh_faces_vec.size(), mesh_faces_vec.data()));
  }



  // return a tuple of array
  // array 1: vertex_plane_indices
  // array 2: intersections
  // array 3: mesh_faces
  // array 4: convex_face_indices
  std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>, py::array_t<int>> convex_to_mesh_with_face_plane_id(py::array_t<double> &planes_array, py::array_t<int> &convex_indices_array, py::array_t<double> &feasiable_points_array){

    py::buffer_info planes_buffer = planes_array.request();
    double *planes_ptr = (double *)planes_buffer.ptr;
    int num_planes = planes_buffer.shape[0];

    py::buffer_info convex_indices_buffer = convex_indices_array.request();
    int *convex_indices_ptr = (int *)convex_indices_buffer.ptr;

    py::buffer_info feasiable_point_buffer = feasiable_points_array.request();
    double *feasiable_point_ptr = (double *)feasiable_point_buffer.ptr;

    std::vector<std::vector<double>> plane_parameters;
    std::vector<int> plane_indices;
    std::vector<std::vector<double>> feasiable_points;
    for(int i=0; i<num_planes; i++){
      plane_parameters.push_back(std::vector<double>{planes_ptr[4*i], planes_ptr[4*i+1], planes_ptr[4*i+2], planes_ptr[4*i+3]});
      plane_indices.push_back(convex_indices_ptr[i]);
    }
    int num_convexs = feasiable_point_buffer.shape[0];
    for(int i=0; i<num_convexs; i++){
      feasiable_points.push_back(std::vector<double>{feasiable_point_ptr[3*i], feasiable_point_ptr[3*i+1], feasiable_point_ptr[3*i+2]});
    }
    std::tuple<std::vector<std::vector<int>>, 
                std::vector<std::vector<int>>,
                std::vector<int>,
                std::vector<int>> all_info = DIFFCONVEX::get_convexes_mesh_with_face_plane_map(plane_parameters, plane_indices, feasiable_points);

    std::vector<std::vector<int>> vertex_plane_indices = std::get<0>(all_info);
    std::vector<std::vector<int>> mesh_faces = std::get<1>(all_info);
    std::vector<int> face_plane_id = std::get<2>(all_info);
    std::vector<int> convex_face_indices = std::get<3>(all_info);
    // flatten vertex_plane_indices
    std::vector<int> vertex_plane_indices_vec;
    for(int i=0; i<vertex_plane_indices.size(); i++){
      for(int j=0; j<vertex_plane_indices[i].size(); j++){
        vertex_plane_indices_vec.push_back(vertex_plane_indices[i][j]);
      }
    }

    // flatten mesh_faces
    std::vector<int> mesh_faces_vec;
    for(int i=0; i<mesh_faces.size(); i++){
      for(int j=0; j<mesh_faces[i].size(); j++){
        mesh_faces_vec.push_back(mesh_faces[i][j]-1);
      }
    }

    return std::make_tuple(py::array_t<int>(vertex_plane_indices_vec.size(), vertex_plane_indices_vec.data()),
                            py::array_t<int>(mesh_faces_vec.size(), mesh_faces_vec.data()),
                            py::array_t<int>(face_plane_id.size(), face_plane_id.data()),
                            py::array_t<int>(convex_face_indices.size(), convex_face_indices.data()));
  }


  // return a tuple of array
  // array 1: vertex_plane_indices
  // array 2: intersections
  // array 3: mesh_faces
  // array 4: convex_face_indices
  std::tuple<py::array_t<double>, py::array_t<int>, py::array_t<double>> densify_convexes(py::array_t<double> &planes_array, py::array_t<int> &convex_indices_array, py::array_t<double> &feasiable_points_array){

    py::buffer_info planes_buffer = planes_array.request();
    double *planes_ptr = (double *)planes_buffer.ptr;
    int num_planes = planes_buffer.shape[0];

    py::buffer_info convex_indices_buffer = convex_indices_array.request();
    int *convex_indices_ptr = (int *)convex_indices_buffer.ptr;

    py::buffer_info feasiable_point_buffer = feasiable_points_array.request();
    double *feasiable_point_ptr = (double *)feasiable_point_buffer.ptr;

    std::vector<std::vector<double>> plane_parameters;
    std::vector<int> plane_indices;
    std::vector<std::vector<double>> feasiable_points;
    for(int i=0; i<num_planes; i++){
      plane_parameters.push_back(std::vector<double>{planes_ptr[4*i], planes_ptr[4*i+1], planes_ptr[4*i+2], planes_ptr[4*i+3]});
      plane_indices.push_back(convex_indices_ptr[i]);
    }
    int num_convexs = feasiable_point_buffer.shape[0];
    for(int i=0; i<num_convexs; i++){
      feasiable_points.push_back(std::vector<double>{feasiable_point_ptr[3*i], feasiable_point_ptr[3*i+1], feasiable_point_ptr[3*i+2]});
    }
    std::tuple<std::vector<std::vector<double>>, 
                std::vector<int>,
                std::vector<double>> all_info = DIFFCONVEX::densify_convexes(plane_parameters, plane_indices, feasiable_points);

    std::vector<std::vector<double>> densified_plane_parameters = std::get<0>(all_info);
    std::vector<int> densified_plane_convex_indices = std::get<1>(all_info);
    std::vector<double> translation_offsets = std::get<2>(all_info);

    // flatten vertex_plane_indices
    std::vector<double> densified_plane_parameters_vec;
    for(int i=0; i<densified_plane_parameters.size(); i++){
      for(int j=0; j<densified_plane_parameters[i].size(); j++){
        densified_plane_parameters_vec.push_back(densified_plane_parameters[i][j]);
      }
    }

    return std::make_tuple(py::array_t<double>(densified_plane_parameters_vec.size(), densified_plane_parameters_vec.data()),
                            py::array_t<int>(densified_plane_convex_indices.size(), densified_plane_convex_indices.data()),
                            py::array_t<double>(translation_offsets.size(), translation_offsets.data()));
  }

  py::array_t<double> get_hull_equations(py::array_t<double> &vertices, py::array_t<double>& translation){
    py::buffer_info vertices_buffer = vertices.request();
    double *vertices_ptr = (double *)vertices_buffer.ptr;
    int num_vertices = vertices_buffer.shape[0];

    py::buffer_info translation_buffer = translation.request();
    double *translation_ptr = (double *)translation_buffer.ptr;

    std::vector<std::vector<double>> vertices_data;
    std::vector<double> translation_data;

    for(int i=0; i<num_vertices; i++){
      vertices_data.push_back(std::vector<double>{vertices_ptr[3*i], vertices_ptr[3*i+1], vertices_ptr[3*i+2]});
    }
    std::vector<std::vector<double>> new_plane_equations = DIFFCONVEX::get_convex_hull(vertices_data, translation_data);
    std::vector<double> new_plane_data;
    for(int i=0; i<new_plane_equations.size(); i++){
      new_plane_data.push_back(new_plane_equations[i][0]);
      new_plane_data.push_back(new_plane_equations[i][1]);
      new_plane_data.push_back(new_plane_equations[i][2]);
      new_plane_data.push_back(new_plane_equations[i][3]);
    }
    return py::array_t<double>(new_plane_data.size(), new_plane_data.data());
  }


  PYBIND11_MODULE(diff_convex, m)
  {
    m.def("convex_to_mesh", &convex_to_mesh,
          "convert convex polyhedron to mesh");
    m.def("convex_to_mesh_with_face_plane_id", &convex_to_mesh_with_face_plane_id,
          "convert convex polyhedron to mesh");
    m.def("densify_convexes", &densify_convexes,
          "densify convex polyhedron");
    m.def("get_hull_equations", &get_hull_equations,
          "get convex hull equations from vertices");
          
  }

}
