#pragma once
#include <vector>

const float RAY_EPSILON = 0.0000001f;

struct Model
{
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texcoords;
    std::vector<unsigned int> indices;
    // float *vertices;
    // float *normals;
    // float *texcoords;
    // int *indices;
    // int num_vertices;
    // int num_normals;
    // int num_texcoords;
    // int num_indices;
};

struct Ray
{
    float origin[3];
    float direction[3];
    float tmin;
    float tmax;
};

struct Intersection
{
    float t;
    float p[3];
    float n[3];
    float texcoord[2];
    int hit;
};