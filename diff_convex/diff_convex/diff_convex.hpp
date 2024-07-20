
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <math.h>
#include <random>
#include <algorithm>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Convex_hull_3/dual/halfspace_intersection_3.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/Polyhedron_3.h>
#include <list>
typedef CGAL::Exact_predicates_inexact_constructions_kernel   K;
typedef K::Plane_3                                            Plane;
typedef K::Point_3                                            Point;
typedef CGAL::Surface_mesh<Point>                             Surface_mesh;
typedef CGAL::Polyhedron_3<K> Polyhedron; 

namespace DIFFCONVEX
{
    // input: 
    // array 1: transformed_plane_parameters [P, 4]
    // array 2: plane_convex_indices [C]
    // array 3: feasiable_points [C, 3]
    // return:
    // array 1: vertex_plane_indices [V, 3] 
    // array 2: mesh_faces [F, 3]
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> get_convexes_mesh(const std::vector<std::vector<double>>& plane_parameters, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points);
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> get_convexes_mesh_with_face_plane_map(const std::vector<std::vector<double>>& plane_parameters, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points);
    std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<double>> densify_convexes(const std::vector<std::vector<double>>& plane_parameters, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points);
    std::vector<std::vector<double>> get_convex_hull(const std::vector<std::vector<double>>& vertices, const std::vector<double>& translation);

}