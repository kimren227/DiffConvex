#include "diff_convex.hpp"
#include <tbb/tbb.h>
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullVertexSet.h"
#include "libqhullcpp/QhullPoint.h"
#include "libqhullcpp/QhullPoints.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/convex_hull_3.h>

#include <igl/loop.h>

#define ASSERT_WITH_MSG(condition, msg) do { \
    if (!(condition)) { \
        std::cerr << "Assertion failed: (" << #condition << "), function " << __FUNCTION__ \
                  << ", file " << __FILE__ << ", line " << __LINE__ << ".\n" \
                  << "Message: " << msg << std::endl; \
        assert(condition); \
    } \
} while (false)

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                Point_3;
typedef K::Vector_3                                Vector_3;

typedef CGAL::Surface_mesh<Point_3>               Mesh;
typedef CGAL::Polyhedron_3<K>                     Polyhedron;

namespace DIFFCONVEX
{
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> get_convexes_mesh(const std::vector<std::vector<double>>& plane_parameters, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points){
        // plane_parameters is a list of plane parameters with shape (num_planes, 4)
        // plane_convex_indices is a list of indices of planes to be intersected with shape (num_planes)
        // return [vertex_plane_indices, faces]
        // vertex_plane_indices is a list of 3 of indices of planes that intersect at each vertex with shape (num_vertices, 3)
        // faces is a list of 3 of indices of vertices that form each face with shape (num_faces, 3)
        std::set<int> convex_index_set(plane_convex_indices.begin(), plane_convex_indices.end());
        int num_vertices = 0;
        std::vector<std::vector<int>>  vertex_plane_indices;
        std::vector<std::vector<int>> mesh_faces;
        for(int convex_index: convex_index_set){
            std::vector<double> convex_plane_parameters;
            int num_points = 0;
            std::map<int, int> plane_index_map;
            std::vector<Point> hull_points;
            for (int i=0; i< plane_convex_indices.size(); i++){
                if (plane_convex_indices[i] == convex_index){
                    convex_plane_parameters.push_back(plane_parameters[i][0]);
                    convex_plane_parameters.push_back(plane_parameters[i][1]);
                    convex_plane_parameters.push_back(plane_parameters[i][2]);
                    convex_plane_parameters.push_back(plane_parameters[i][3]);
                    plane_index_map[num_points] = i;
                    num_points++;
                }
            }
            // compute halfspace intersection
            orgQhull::Qhull qhull;
            std::string fpStr = "H";
            fpStr += std::to_string(feasiable_points[convex_index][0]);
            fpStr += "," + std::to_string(feasiable_points[convex_index][1]);
            fpStr += "," + std::to_string(feasiable_points[convex_index][2]);
            qhull.runQhull("", 4, num_points, convex_plane_parameters.data(), fpStr.c_str());
            // find dual facets
            std::vector<double> points;
            std::vector<orgQhull::QhullFacet> facets = qhull.facetList().toStdVector();
            for(orgQhull::QhullFacet facet: facets){
                auto vertices = facet.vertices();
                std::vector<int> vertex_indices;
                int counter = 0;
                for(auto vertex: vertices){
                    if(counter == 3){
                        std::cout << "Warning: more than 3 vertices in a facet" << std::endl;
                        break;
                    }
                    vertex_indices.push_back(plane_index_map[vertex.point().id()]);
                    counter++;
                }
                vertex_plane_indices.push_back(vertex_indices);
                double offset = facet.getFacetT()->offset;
                double x = *(facet.getFacetT()->normal) / offset;
                double y = *(facet.getFacetT()->normal + 1) / offset;
                double z = *(facet.getFacetT()->normal + 2) / offset;
                points.push_back(x);
                points.push_back(y);
                points.push_back(z);
                hull_points.push_back(Point(x, y, z));
            }
            // compute convex hull
            orgQhull::Qhull convex_qhull;

            // Run QHull to compute the convex hull
            convex_qhull.runQhull("", 3, points.size()/3, points.data(), "Qt"); // Qt for triangulation

            for (auto f = convex_qhull.facetList().begin(); f != convex_qhull.facetList().end(); ++f) {
                if (!f->isGood()) continue; // Skip non-good facets if any
                // Get the vertices of each facet (triangle in the mesh)
                orgQhull::QhullVertexSet vertices = f->vertices();
                int size = vertices.size(); // Number of vertices, always 3 for a triangle
                ASSERT_WITH_MSG(size == 3, "A facet is not a triangle");
                std::vector<int> face_indices;
                for (auto v = vertices.begin(); v != vertices.end(); ++v) {
                    orgQhull::QhullPoint p = (*v).point();
                    face_indices.push_back(p.id());
                }

                Vector_3 triangle_normal = CGAL::cross_product(
                        hull_points[face_indices[1]]-hull_points[face_indices[0]], 
                        hull_points[face_indices[2]]-hull_points[face_indices[0]]
                        );
                if(CGAL::scalar_product(triangle_normal,
                                        hull_points[face_indices[1]] - Point(0, 0, 0)) 
                                        > 0)
                {
                    std::reverse(face_indices.begin(), face_indices.end());
                }
                std::vector<int> final_face_indices;
                for (int i=0; i<face_indices.size(); i++){
                    final_face_indices.push_back(face_indices[i] + 1 + num_vertices);
                }
                mesh_faces.push_back(final_face_indices);
            }
            num_vertices += hull_points.size();
        }
        return std::make_tuple(vertex_plane_indices, mesh_faces);
    }

    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> get_convexes_mesh_with_face_plane_map(const std::vector<std::vector<double>>& plane_parameters, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points){
        // plane_parameters is a list of plane parameters with shape (num_planes, 4)
        // plane_convex_indices is a list of indices of planes to be intersected with shape (num_planes)
        // return [vertex_plane_indices, faces]
        // vertex_plane_indices is a list of 3 of indices of planes that intersect at each vertex with shape (num_vertices, 3)
        // faces is a list of 3 of indices of vertices that form each face with shape (num_faces, 3)
        std::set<int> convex_index_set(plane_convex_indices.begin(), plane_convex_indices.end());
        int num_vertices = 0;
        std::vector<std::vector<int>>  vertex_plane_indices;
        std::vector<std::vector<int>> mesh_faces;
        std::vector<int> mesh_face_plane_ids;
        std::vector<int> convex_face_start_indices;


        for(int convex_index: convex_index_set){
            std::vector<double> convex_plane_parameters;
            int num_points = 0;
            std::map<int, int> plane_index_map;
            std::vector<Point> hull_points;
            for (int i=0; i< plane_convex_indices.size(); i++){
                if (plane_convex_indices[i] == convex_index){
                    convex_plane_parameters.push_back(plane_parameters[i][0]);
                    convex_plane_parameters.push_back(plane_parameters[i][1]);
                    convex_plane_parameters.push_back(plane_parameters[i][2]);
                    convex_plane_parameters.push_back(plane_parameters[i][3]);
                    plane_index_map[num_points] = i;
                    num_points++;
                }
            }
            // compute halfspace intersection
            orgQhull::Qhull qhull;
            std::string fpStr = "H";
            fpStr += std::to_string(feasiable_points[convex_index][0]);
            fpStr += "," + std::to_string(feasiable_points[convex_index][1]);
            fpStr += "," + std::to_string(feasiable_points[convex_index][2]);

            qhull.runQhull("", 4, num_points, convex_plane_parameters.data(), fpStr.c_str());
            // find dual facets
            std::vector<double> points;
            std::vector<orgQhull::QhullFacet> facets = qhull.facetList().toStdVector();
            for(orgQhull::QhullFacet facet: facets){
                auto vertices = facet.vertices();
                std::vector<int> vertex_indices;
                int counter = 0;
                for(auto vertex: vertices){
                    if(counter == 3){
                        std::cout << "Warning: more than 3 vertices in a facet" << std::endl;
                        break;
                    }
                    vertex_indices.push_back(plane_index_map[vertex.point().id()]);
                    counter++;
                }
                vertex_plane_indices.push_back(vertex_indices);
                double offset = facet.getFacetT()->offset;
                double x = *(facet.getFacetT()->normal) / offset;
                double y = *(facet.getFacetT()->normal + 1) / offset;
                double z = *(facet.getFacetT()->normal + 2) / offset;
                points.push_back(x);
                points.push_back(y);
                points.push_back(z);
                hull_points.push_back(Point(x, y, z));
            }
            // compute convex hull
            orgQhull::Qhull convex_qhull;

            // Run QHull to compute the convex hull
            convex_qhull.runQhull("", 3, points.size()/3, points.data(), "Qt"); // Qt for triangulation

            for (auto f = convex_qhull.facetList().begin(); f != convex_qhull.facetList().end(); ++f) {
                if (!f->isGood()) continue; // Skip non-good facets if any
                // Get the vertices of each facet (triangle in the mesh)
                orgQhull::QhullVertexSet vertices = f->vertices();
                int size = vertices.size(); // Number of vertices, always 3 for a triangle
                ASSERT_WITH_MSG(size == 3, "A facet is not a triangle");
                std::vector<int> face_indices;
                std::vector<int> plane_ids;
                std::map<int, int> plane_id_count;
                for (auto v = vertices.begin(); v != vertices.end(); ++v) {
                    orgQhull::QhullPoint p = (*v).point();
                    face_indices.push_back(p.id());
                    for (int i=0; i<vertex_plane_indices[p.id() + num_vertices].size(); i++){
                        if(plane_id_count.find(vertex_plane_indices[p.id() + num_vertices][i]) != plane_id_count.end()){
                            plane_id_count[vertex_plane_indices[p.id() + num_vertices][i]]++;
                        }
                        else{
                            plane_id_count[vertex_plane_indices[p.id()+ num_vertices][i]] = 1;
                        }

                    }
                }

                // count which 
                for(auto it = plane_id_count.begin(); it != plane_id_count.end(); it++){
                    if(it->second == 3){
                        mesh_face_plane_ids.push_back(it->first);
                        break;
                    }
                }     

                Vector_3 triangle_normal = CGAL::cross_product(
                        hull_points[face_indices[1]]-hull_points[face_indices[0]], 
                        hull_points[face_indices[2]]-hull_points[face_indices[0]]
                        );
                if(CGAL::scalar_product(triangle_normal,
                                        hull_points[face_indices[1]] - Point(0, 0, 0)) 
                                        > 0)
                {
                    std::reverse(face_indices.begin(), face_indices.end());
                }
                std::vector<int> final_face_indices;
                for (int i=0; i<face_indices.size(); i++){
                    final_face_indices.push_back(face_indices[i] + 1 + num_vertices);
                }
                mesh_faces.push_back(final_face_indices);
            }
            num_vertices += hull_points.size();
            convex_face_start_indices.push_back(mesh_faces.size());

        }
        return std::make_tuple(vertex_plane_indices, mesh_faces, mesh_face_plane_ids, convex_face_start_indices);
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<double>> densify_convexes(const std::vector<std::vector<double>>& plane_parameters, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points){
        std::set<int> convex_index_set(plane_convex_indices.begin(), plane_convex_indices.end());
        std::vector<std::vector<int>>  vertex_plane_indices;

        std::vector<std::vector<double>> densified_plane_parameters;
        std::vector<int> densified_plane_convex_indices;
        std::vector<double> translation_offsets;

        for(int convex_index: convex_index_set){
            std::vector<std::vector<int>> mesh_faces;

            std::vector<double> convex_plane_parameters;
            int num_points = 0;
            std::map<int, int> plane_index_map;
            std::vector<Point> hull_points;
            for (int i=0; i< plane_convex_indices.size(); i++){
                if (plane_convex_indices[i] == convex_index){
                    convex_plane_parameters.push_back(plane_parameters[i][0]);
                    convex_plane_parameters.push_back(plane_parameters[i][1]);
                    convex_plane_parameters.push_back(plane_parameters[i][2]);
                    convex_plane_parameters.push_back(plane_parameters[i][3]);
                    plane_index_map[num_points] = i;
                    num_points++;
                }
            }
            // compute halfspace intersection
            orgQhull::Qhull qhull;
            std::string fpStr = "H";
            fpStr += std::to_string(feasiable_points[convex_index][0]);
            fpStr += "," + std::to_string(feasiable_points[convex_index][1]);
            fpStr += "," + std::to_string(feasiable_points[convex_index][2]);
            qhull.runQhull("", 4, num_points, convex_plane_parameters.data(), fpStr.c_str());
            // find dual facets
            std::vector<double> points;
            std::vector<orgQhull::QhullFacet> facets = qhull.facetList().toStdVector();
            for(orgQhull::QhullFacet facet: facets){
                auto vertices = facet.vertices();
                std::vector<int> vertex_indices;
                int counter = 0;
                for(auto vertex: vertices){
                    if(counter == 3){
                        std::cout << "Warning: more than 3 vertices in a facet" << std::endl;
                        break;
                    }
                    vertex_indices.push_back(plane_index_map[vertex.point().id()]);
                    counter++;
                }
                vertex_plane_indices.push_back(vertex_indices);
                double offset = facet.getFacetT()->offset;
                double x = *(facet.getFacetT()->normal) / offset;
                double y = *(facet.getFacetT()->normal + 1) / offset;
                double z = *(facet.getFacetT()->normal + 2) / offset;
                points.push_back(x);
                points.push_back(y);
                points.push_back(z);
                hull_points.push_back(Point(x, y, z));
            }
        
            // compute convex hull
            orgQhull::Qhull convex_qhull;

            // Run QHull to compute the convex hull
            convex_qhull.runQhull("", 3, points.size()/3, points.data(), "Qt"); // Qt for triangulation

            for (auto f = convex_qhull.facetList().begin(); f != convex_qhull.facetList().end(); ++f) {
                if (!f->isGood()) continue; // Skip non-good facets if any
                // Get the vertices of each facet (triangle in the mesh)
                orgQhull::QhullVertexSet vertices = f->vertices();
                int size = vertices.size(); // Number of vertices, always 3 for a triangle
                ASSERT_WITH_MSG(size == 3, "A facet is not a triangle");
                std::vector<int> face_indices;
                for (auto v = vertices.begin(); v != vertices.end(); ++v) {
                    orgQhull::QhullPoint p = (*v).point();
                    face_indices.push_back(p.id());
                }

                Vector_3 triangle_normal = CGAL::cross_product(
                        hull_points[face_indices[1]]-hull_points[face_indices[0]], 
                        hull_points[face_indices[2]]-hull_points[face_indices[0]]
                        );
                if(CGAL::scalar_product(triangle_normal,
                                        hull_points[face_indices[1]] - Point(0, 0, 0)) 
                                        > 0)
                {
                    std::reverse(face_indices.begin(), face_indices.end());
                }
                std::vector<int> final_face_indices;
                for (int i=0; i<face_indices.size(); i++){
                    final_face_indices.push_back(face_indices[i] + 1);
                }
                mesh_faces.push_back(final_face_indices);
            }

            // create a libigl mesh and do loop subdivision
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;

            V.resize(hull_points.size(), 3);
            F.resize(mesh_faces.size(), 3);
            for(int i=0; i<hull_points.size(); i++){
                V.row(i) << hull_points[i].x(), hull_points[i].y(), hull_points[i].z();
            }
            for(int i=0; i<mesh_faces.size(); i++){
                F.row(i) << mesh_faces[i][0]-1, mesh_faces[i][1]-1, mesh_faces[i][2]-1;
            }
            Eigen::MatrixXd Vout;
            Eigen::MatrixXi Fout;
            // create mesh from eigen
            if (V.rows() >= F.rows() && F.rows() > 0){
                std::cout << "Some thing went wrong..." << std::endl;
                Vout = V;
                Fout = F;
            }
            else{
                igl::loop(V, F, Vout, Fout, 1);
            }
            
            // convert eigen matrix back to std vector
            std::vector<double> densified_points;
            // compute vertex mean
            Eigen::MatrixXd Vmean = Vout.colwise().mean();
            translation_offsets.push_back(double(convex_index));
            translation_offsets.push_back(Vmean(0, 0));
            translation_offsets.push_back(Vmean(0, 1));
            translation_offsets.push_back(Vmean(0, 2));


            for(int i=0; i<Vout.rows(); i++){
                densified_points.push_back(Vout(i, 0) - Vmean(0, 0));
                densified_points.push_back(Vout(i, 1) - Vmean(0, 1));
                densified_points.push_back(Vout(i, 2) - Vmean(0, 2));
            }
            orgQhull::Qhull densified_convex_qhull;

            densified_convex_qhull.runQhull("", 3, densified_points.size()/3, densified_points.data(), "n");
            // get all plane equations from densified_convex_qhull

            std::vector<orgQhull::QhullFacet> densified_facets = densified_convex_qhull.facetList().toStdVector();
            for(orgQhull::QhullFacet facet: densified_facets){
                double offset = facet.getFacetT()->offset;
                double x = *(facet.getFacetT()->normal);
                double y = *(facet.getFacetT()->normal + 1);
                double z = *(facet.getFacetT()->normal + 2);
                std::vector<double> equation;
                equation.push_back(x);
                equation.push_back(y);
                equation.push_back(z);
                equation.push_back(offset);
                densified_plane_parameters.push_back(equation);
                densified_plane_convex_indices.push_back(convex_index);
            }
        }

        return std::make_tuple(densified_plane_parameters, densified_plane_convex_indices, translation_offsets);
    }


    std::vector<std::vector<double>> get_convex_hull(const std::vector<std::vector<double>>& vertices, const std::vector<double>& translation){

        std::vector<std::vector<double>> densified_plane_parameters;
        orgQhull::Qhull densified_convex_qhull;
        std::vector<double> vertices_data;
        for(std::vector<double> v: vertices){
            vertices_data.push_back(v[0]);
            vertices_data.push_back(v[1]);
            vertices_data.push_back(v[2]);
        }
        densified_convex_qhull.runQhull("", 3, vertices_data.size()/3, vertices_data.data(), "n");
        // get all plane equations from densified_convex_qhull
        std::vector<orgQhull::QhullFacet> densified_facets = densified_convex_qhull.facetList().toStdVector();
        for(orgQhull::QhullFacet facet: densified_facets){
            double offset = facet.getFacetT()->offset;
            double x = *(facet.getFacetT()->normal);
            double y = *(facet.getFacetT()->normal + 1);
            double z = *(facet.getFacetT()->normal + 2);
            std::vector<double> equation;
            equation.push_back(x);
            equation.push_back(y);
            equation.push_back(z);
            equation.push_back(offset);
            densified_plane_parameters.push_back(equation);
        }
        return densified_plane_parameters;
    }


    // std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<double>> densify_convexes_vertices(const std::vector<std::vector<double>>& vertices, const std::vector<int>& plane_convex_indices, const std::vector<std::vector<double>>& feasiable_points){
    //     std::set<int> convex_index_set(plane_convex_indices.begin(), plane_convex_indices.end());
    //     std::vector<std::vector<int>>  vertex_plane_indices;

    //     std::vector<std::vector<double>> densified_plane_parameters;
    //     std::vector<int> densified_plane_convex_indices;
    //     std::vector<double> translation_offsets;

    //     for(int convex_index: convex_index_set){
    //         std::vector<std::vector<int>> mesh_faces;

    //         std::vector<double> convex_plane_parameters;
    //         int num_points = 0;
    //         std::map<int, int> plane_index_map;
    //         std::vector<Point> hull_points;
    //         for (int i=0; i< plane_convex_indices.size(); i++){
    //             if (plane_convex_indices[i] == convex_index){
    //                 convex_plane_parameters.push_back(plane_parameters[i][0]);
    //                 convex_plane_parameters.push_back(plane_parameters[i][1]);
    //                 convex_plane_parameters.push_back(plane_parameters[i][2]);
    //                 convex_plane_parameters.push_back(plane_parameters[i][3]);
    //                 plane_index_map[num_points] = i;
    //                 num_points++;
    //             }
    //         }
    //         // compute halfspace intersection
    //         orgQhull::Qhull qhull;
    //         std::string fpStr = "H";
    //         fpStr += std::to_string(feasiable_points[convex_index][0]);
    //         fpStr += "," + std::to_string(feasiable_points[convex_index][1]);
    //         fpStr += "," + std::to_string(feasiable_points[convex_index][2]);
    //         qhull.runQhull("", 4, num_points, convex_plane_parameters.data(), fpStr.c_str());
    //         // find dual facets
    //         std::vector<double> points;
    //         std::vector<orgQhull::QhullFacet> facets = qhull.facetList().toStdVector();
    //         for(orgQhull::QhullFacet facet: facets){
    //             auto vertices = facet.vertices();
    //             std::vector<int> vertex_indices;
    //             int counter = 0;
    //             for(auto vertex: vertices){
    //                 if(counter == 3){
    //                     std::cout << "Warning: more than 3 vertices in a facet" << std::endl;
    //                     break;
    //                 }
    //                 vertex_indices.push_back(plane_index_map[vertex.point().id()]);
    //                 counter++;
    //             }
    //             vertex_plane_indices.push_back(vertex_indices);
    //             double offset = facet.getFacetT()->offset;
    //             double x = *(facet.getFacetT()->normal) / offset;
    //             double y = *(facet.getFacetT()->normal + 1) / offset;
    //             double z = *(facet.getFacetT()->normal + 2) / offset;
    //             points.push_back(x);
    //             points.push_back(y);
    //             points.push_back(z);
    //             hull_points.push_back(Point(x, y, z));
    //         }
        
    //         // compute convex hull
    //         orgQhull::Qhull convex_qhull;

    //         // Run QHull to compute the convex hull
    //         convex_qhull.runQhull("", 3, points.size()/3, points.data(), "Qt"); // Qt for triangulation

    //         for (auto f = convex_qhull.facetList().begin(); f != convex_qhull.facetList().end(); ++f) {
    //             if (!f->isGood()) continue; // Skip non-good facets if any
    //             // Get the vertices of each facet (triangle in the mesh)
    //             orgQhull::QhullVertexSet vertices = f->vertices();
    //             int size = vertices.size(); // Number of vertices, always 3 for a triangle
    //             ASSERT_WITH_MSG(size == 3, "A facet is not a triangle");
    //             std::vector<int> face_indices;
    //             for (auto v = vertices.begin(); v != vertices.end(); ++v) {
    //                 orgQhull::QhullPoint p = (*v).point();
    //                 face_indices.push_back(p.id());
    //             }

    //             Vector_3 triangle_normal = CGAL::cross_product(
    //                     hull_points[face_indices[1]]-hull_points[face_indices[0]], 
    //                     hull_points[face_indices[2]]-hull_points[face_indices[0]]
    //                     );
    //             if(CGAL::scalar_product(triangle_normal,
    //                                     hull_points[face_indices[1]] - Point(0, 0, 0)) 
    //                                     > 0)
    //             {
    //                 std::reverse(face_indices.begin(), face_indices.end());
    //             }
    //             std::vector<int> final_face_indices;
    //             for (int i=0; i<face_indices.size(); i++){
    //                 final_face_indices.push_back(face_indices[i] + 1);
    //             }
    //             mesh_faces.push_back(final_face_indices);
    //         }

    //         // create a libigl mesh and do loop subdivision
    //         Eigen::MatrixXd V;
    //         Eigen::MatrixXi F;

    //         V.resize(hull_points.size(), 3);
    //         F.resize(mesh_faces.size(), 3);
    //         for(int i=0; i<hull_points.size(); i++){
    //             V.row(i) << hull_points[i].x(), hull_points[i].y(), hull_points[i].z();
    //         }
    //         for(int i=0; i<mesh_faces.size(); i++){
    //             F.row(i) << mesh_faces[i][0]-1, mesh_faces[i][1]-1, mesh_faces[i][2]-1;
    //         }
    //         Eigen::MatrixXd Vout;
    //         Eigen::MatrixXi Fout;
    //         // create mesh from eigen
    //         if (V.rows() >= F.rows() && F.rows() > 0){
    //             std::cout << "Some thing went wrong..." << std::endl;
    //             Vout = V;
    //             Fout = F;
    //         }
    //         else{
    //             igl::loop(V, F, Vout, Fout, 1);
    //         }
            
    //         // convert eigen matrix back to std vector
    //         std::vector<double> densified_points;
    //         // compute vertex mean
    //         Eigen::MatrixXd Vmean = Vout.colwise().mean();
    //         translation_offsets.push_back(double(convex_index));
    //         translation_offsets.push_back(Vmean(0, 0));
    //         translation_offsets.push_back(Vmean(0, 1));
    //         translation_offsets.push_back(Vmean(0, 2));


    //         for(int i=0; i<Vout.rows(); i++){
    //             densified_points.push_back(Vout(i, 0) - Vmean(0, 0));
    //             densified_points.push_back(Vout(i, 1) - Vmean(0, 1));
    //             densified_points.push_back(Vout(i, 2) - Vmean(0, 2));
    //         }
    //         orgQhull::Qhull densified_convex_qhull;

    //         densified_convex_qhull.runQhull("", 3, densified_points.size()/3, densified_points.data(), "n");
    //         // get all plane equations from densified_convex_qhull

    //         std::vector<orgQhull::QhullFacet> densified_facets = densified_convex_qhull.facetList().toStdVector();
    //         for(orgQhull::QhullFacet facet: densified_facets){
    //             double offset = facet.getFacetT()->offset;
    //             double x = *(facet.getFacetT()->normal);
    //             double y = *(facet.getFacetT()->normal + 1);
    //             double z = *(facet.getFacetT()->normal + 2);
    //             std::vector<double> equation;
    //             equation.push_back(x);
    //             equation.push_back(y);
    //             equation.push_back(z);
    //             equation.push_back(offset);
    //             densified_plane_parameters.push_back(equation);
    //             densified_plane_convex_indices.push_back(convex_index);
    //         }
    //     }

    //     return std::make_tuple(densified_plane_parameters, densified_plane_convex_indices, translation_offsets);
    // }


}
