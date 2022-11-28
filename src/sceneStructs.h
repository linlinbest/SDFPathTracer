#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define USE_BVH_FOR_INTERSECTION 1
//Just for testing
#define SAMPLE_COUNT 100

//Hemisphere Harmonic Coefficient
//reference: 
//(0,0)
#define HSH_COEFFICIENT_0 0.398942280f
//(-1,1)
#define HSH_COEFFICIENT_1 0.488602512f
//(0,1)
#define HSH_COEFFICIENT_2 0.690988299f
//(1,1)
#define HSH_COEFFICIENT_3 0.488602512f
//(-2,2)
#define HSH_COEFFICIENT_4 0.182091405f
//(-1,2)
#define HSH_COEFFICIENT_5 0.364182810f
//(0,2)
#define HSH_COEFFICIENT_6 0.892062058f
//(1,2)
#define HSH_COEFFICIENT_7 0.364182810f
//(2,2)
#define HSH_COEFFICIENT_8 0.182091405f

#define PI 3.141592653

static const float HemisphereHarmonicCoefficient[9] = { HSH_COEFFICIENT_0 ,HSH_COEFFICIENT_1 ,HSH_COEFFICIENT_2 ,
HSH_COEFFICIENT_3 ,HSH_COEFFICIENT_4 ,HSH_COEFFICIENT_5 ,HSH_COEFFICIENT_6,HSH_COEFFICIENT_7,HSH_COEFFICIENT_8 };

void getCorrespondHemisphereHarmonicBasisFunc(int num)
{

}

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE,
    MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int faceStartIdx; // use with array of Triangle
    int faceNum;
};

struct Triangle {
    glm::vec3 point1;
    glm::vec3 point2;
    glm::vec3 point3;
    glm::vec3 normal1;
    glm::vec3 normal2;
    glm::vec3 normal3;
#if USE_BVH_FOR_INTERSECTION
    int geomId;
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
    glm::vec3 centroid;
    __host__ __device__ void computeLocalBoundingBox()
    {
        minCorner = glm::min(point1, glm::min(point2, point3));
        maxCorner = glm::max(point1, glm::max(point2, point3));
        centroid = (minCorner + maxCorner) * 0.5f;
    }
    __host__ __device__ void computeGlobalBoundingBox(const Geom& geom)
    {
        glm::vec3 globalPoint1 = glm::vec3(geom.transform * glm::vec4(point1, 1.0f));
        glm::vec3 globalPoint2 = glm::vec3(geom.transform * glm::vec4(point2, 1.0f));
        glm::vec3 globalPoint3 = glm::vec3(geom.transform * glm::vec4(point3, 1.0f));
        minCorner = glm::min(globalPoint1, glm::min(globalPoint2, globalPoint3));
        maxCorner = glm::max(globalPoint1, glm::max(globalPoint2, globalPoint3));
        centroid = (minCorner + maxCorner) * 0.5f;
    }

    __host__ __device__ void localToWorld(const Geom& geom)
    {
        point1 = glm::vec3(geom.transform * glm::vec4(point1, 1.0f));
        point2 = glm::vec3(geom.transform * glm::vec4(point2, 1.0f));
        point3 = glm::vec3(geom.transform * glm::vec4(point3, 1.0f));
        normal1 = glm::vec3(geom.transform * glm::vec4(normal1, 0.f));
        normal2 = glm::vec3(geom.transform * glm::vec4(normal2, 0.f));
        normal3 = glm::vec3(geom.transform * glm::vec4(normal3, 0.f));
    }
#endif
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    //map for diffuse for test
    glm::vec3 diffuseColorRadianceCache[9];
    //glossiness need higher level
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};

// CHECKITOUT - a simple struct for storing scene geometry information per-pixel.
// What information might be helpful for guiding a denoising filter?
struct GBufferPixel {
    float t;
    glm::vec3 pos;
    glm::vec3 nor;
};

struct SDF
{
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
    glm::ivec3 resolution;
    glm::vec3 gridExtent;
};

struct SDFGrid {
    float dist;
    int geomId;
};
