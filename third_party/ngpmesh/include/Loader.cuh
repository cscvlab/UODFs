#pragma once
#ifndef LOADER_CUH
#define LOADER_CUH

#include<iostream>
#include<vector>
#include<Triangle.cuh>

inline std::vector<Triangle> loadTriangles(std::vector<Eigen::Matrix3f> &triangles_matrix){
    std::vector<Triangle> triangles;
    for(auto t : triangles_matrix){
        Triangle triangle;
        triangle.a = t.row(0);
        triangle.b = t.row(1);
        triangle.c = t.row(2);
        triangles.push_back(triangle);
    }
    std::cout << "Loaded Triangles: " << triangles.size() << std::endl;
    return triangles;
}

inline std::vector<Triangle> loadTriangles(std::vector<Eigen::Vector3f> &vertices, std::vector<Eigen::Vector3i> &faces){
    std::vector<Triangle> triangles;
    for(auto f : faces){
        Triangle triangle;
        triangle.a = vertices[f.x()];
        triangle.b = vertices[f.y()];
        triangle.c = vertices[f.z()];
        triangles.push_back(triangle);
    }
    std::cout << "Loaded Triangles: " << triangles.size() << std::endl;
    return triangles;
}

#endif