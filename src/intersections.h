#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "bvhTree.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(const Ray &r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(const glm::mat4 &m, const glm::vec4 &v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(const Geom &box, const Ray &r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(const Geom& sphere, const Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


// CHECKITOUT
/**
 * Test intersection between a ray and a triangle.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float triangleIntersectionTest(const Triangle &triangle, const Ray &r,
    glm::vec3& intersectionPoint)
{
    glm::vec3 baryPos;
    bool hasIntersect = glm::intersectRayTriangle(r.origin, r.direction, triangle.point1, triangle.point2, triangle.point3, baryPos);
    if (!hasIntersect) return -1.f;

    intersectionPoint = (1.f - baryPos.x - baryPos.y) * triangle.point1 + baryPos.x * triangle.point2 + baryPos.y * triangle.point3;
    
    return glm::length(r.origin - intersectionPoint);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(const Geom &mesh, const Triangle* faces, const Ray &r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    // get ray in untransformed space
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float closestDist = FLT_MAX;

    glm::vec3 currIntersectPoint;
    glm::vec3 currNormal;
    bool currOutside;

    const Triangle* closestTri = nullptr;
    for (int i = mesh.faceStartIdx; i < mesh.faceStartIdx + mesh.faceNum; i++)
    {
        float t = triangleIntersectionTest(faces[i], rt, currIntersectPoint);
        if (t == -1.f) continue;
        
        if (t < closestDist)
        {
            closestDist = t;
            intersectionPoint = currIntersectPoint;
            closestTri = faces + i;
        }
    }

    if (closestDist != FLT_MAX)
    {
        // calculate local space normal of the closest triangle
        float S = 0.5f * glm::length(glm::cross(closestTri->point1 - closestTri->point2, closestTri->point1 - closestTri->point3));
        float s1 = 0.5f * glm::length(glm::cross(intersectionPoint - closestTri->point2, intersectionPoint - closestTri->point3)) / S;
        float s2 = 0.5f * glm::length(glm::cross(intersectionPoint - closestTri->point3, intersectionPoint - closestTri->point1)) / S;
        float s3 = 0.5f * glm::length(glm::cross(intersectionPoint - closestTri->point1, intersectionPoint - closestTri->point2)) / S;
        normal = closestTri->normal1 * s1 + closestTri->normal2 * s2 + closestTri->normal3 * s3;

        if (glm::dot(rt.direction, normal) <= 0.f)
        {
            outside = true;
        }
        else
        {
            outside = false;
            normal = -normal;
        }

        // Turn intersection point and normal into global space
        intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.f));
        normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal, 0.f)));

        closestDist = glm::length(r.origin - intersectionPoint);
        return closestDist;
    }
    return -1.f;
}

#if USE_BVH_FOR_INTERSECTION
__host__ __device__ float boxIntersectionTest(const glm::vec3& minCorner, const glm::vec3& maxCorner, const Ray& r)
{
    // global space intersection test
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = r.direction[xyz];
        if (glm::abs(qdxyz) > 0.00001f) {
            float t1 = (minCorner[xyz] - r.origin[xyz]) / qdxyz;
            float t2 = (maxCorner[xyz] - r.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        return tmin;
    }
    return -1;
}

__host__ __device__ float triangleIntersectionTest(const Triangle& triangle, const Geom& mesh,
    const Ray& r, glm::vec3& intersectionPoint)
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    glm::vec3 baryPos;
    bool hasIntersect = glm::intersectRayTriangle(rt.origin, rt.direction, triangle.point1, triangle.point2, triangle.point3, baryPos);
    if (!hasIntersect) return -1.f;

    intersectionPoint = (1.f - baryPos.x - baryPos.y) * triangle.point1 + baryPos.x * triangle.point2 + baryPos.y * triangle.point3;

    glm::vec3 globalIntersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.f));

    return glm::length(r.origin - globalIntersectionPoint);
}

__host__ __device__ float bvhIntersectionTest(const Geom* geoms, const BVHNode* bvhNodes, const int bvhNodes_size,
    const Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, int* hitGeomId)
{    
    int currIdx = 0;

    int idxToVisit[64];
    int toVisitOffset = 0;

    idxToVisit[toVisitOffset++] = 0;

    const Triangle* closestTri = nullptr;
    float closestDist = FLT_MAX;

    glm::vec3 invDir(1.f / r.direction.x, 1.f / r.direction.y, 1.f / r.direction.z);
    bool dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
    while (toVisitOffset > 0)
    {
        currIdx = idxToVisit[--toVisitOffset];
        //if (currIdx >= bvhNodes_size) continue;

        if (!bvhNodes[currIdx].isLeaf)
        {
            float rayLength = boxIntersectionTest(bvhNodes[currIdx].minCorner, bvhNodes[currIdx].maxCorner, r);
            if (rayLength == -1.f || rayLength > closestDist) continue;
            int leftIdx = currIdx * 2 + 1;
            int rightIdx = leftIdx + 1;

            /*idxToVisit[toVisitOffset++] = rightIdx;
            idxToVisit[toVisitOffset++] = leftIdx;*/

            // Advance to the near node first
            if (dirIsNeg[bvhNodes[currIdx].axis]) {
                idxToVisit[toVisitOffset++] = leftIdx;
                idxToVisit[toVisitOffset++] = rightIdx;
            }
            else {
                idxToVisit[toVisitOffset++] = rightIdx;
                idxToVisit[toVisitOffset++] = leftIdx;
            }
        }
        else
        {
            if (!bvhNodes[currIdx].hasFace) continue;
            const Triangle* face = &bvhNodes[currIdx].face;
            glm::vec3 localIntersectPos;
            float t = triangleIntersectionTest(*face, geoms[face->geomId], r, localIntersectPos);

            if (t != -1.f && closestDist > t)
            {
                closestTri = &bvhNodes[currIdx].face;
                closestDist = t;
                intersectionPoint = localIntersectPos;
            }
        }
    }

    if (closestDist != FLT_MAX)
    {
        *hitGeomId = closestTri->geomId;

        // calculate local space normal of the closest triangle
        float S = 0.5f * glm::length(glm::cross(closestTri->point1 - closestTri->point2, closestTri->point1 - closestTri->point3));
        float s1 = 0.5f * glm::length(glm::cross(intersectionPoint - closestTri->point2, intersectionPoint - closestTri->point3)) / S;
        float s2 = 0.5f * glm::length(glm::cross(intersectionPoint - closestTri->point3, intersectionPoint - closestTri->point1)) / S;
        float s3 = 0.5f * glm::length(glm::cross(intersectionPoint - closestTri->point1, intersectionPoint - closestTri->point2)) / S;
        normal = closestTri->normal1 * s1 + closestTri->normal2 * s2 + closestTri->normal3 * s3;

        const Geom* mesh = geoms + closestTri->geomId;

        // Turn intersection point and normal into global space
        intersectionPoint = getPointOnRay(r, closestDist);
        //intersectionPoint = multiplyMV(mesh->transform, glm::vec4(intersectionPoint, 1.f));
        normal = glm::normalize(multiplyMV(mesh->invTranspose, glm::vec4(normal, 0.f)));

        if (glm::dot(r.direction, normal) <= 0.f)
        {
            outside = true;
        }
        else
        {
            outside = false;
            normal = -normal;
        }

        closestDist = glm::length(r.origin - intersectionPoint);
        return closestDist;
    }

    return -1.f;
}

#endif

// USE SDF
#if 1

__host__ __device__
float udfTriangle(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c)
{
    glm::vec3 ba = b - a; glm::vec3 pa = p - a;
    glm::vec3 cb = c - b; glm::vec3 pb = p - b;
    glm::vec3 ac = a - c; glm::vec3 pc = p - c;
    glm::vec3 nor = glm::cross(ba, ac);

    glm::vec3 ba_pa = ba * glm::clamp(glm::dot(ba, pa) / glm::dot(ba, ba), 0.f, 1.f) - pa;
    glm::vec3 cb_pb = cb * glm::clamp(glm::dot(cb, pb) / glm::dot(cb, cb), 0.f, 1.f) - pb;
    glm::vec3 ac_pc = ac * glm::clamp(glm::dot(ac, pc) / glm::dot(ac, ac), 0.f, 1.f) - pc;

    return glm::sqrt(
        (glm::sign(glm::dot(glm::cross(ba, nor), pa)) +
         glm::sign(glm::dot(glm::cross(cb, nor), pb)) +
         glm::sign(glm::dot(glm::cross(ac, nor), pc)) < 2.f)
        ?
        glm::min(glm::min(
            glm::dot(ba_pa, ba_pa),
            glm::dot(cb_pb, cb_pb)),
            glm::dot(ac_pc, ac_pc))
        :
        glm::dot(nor, pa) * glm::dot(nor, pa) / glm::dot(nor, nor));
}

//__host__ __device__
//bool intersectSphereBox(glm::vec3 lower, glm::vec3 upper, glm::vec3 p, float radius2)
//{
//    glm::vec3 q = glm::clamp(p, lower, upper);
//    return glm::dot(p - q, p - q) <= radius2;
//}
//
//__host__ __device__
//bool intersectSphereTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 p, float r2)
//{
//    float udf = udfTriangle(p, a, b, c);
//    return udf * udf <= r2;
//}

__host__ __device__
float sdBox(glm::vec3 p, glm::vec3 b)
{
    glm::vec3 q = abs(p) - b;
    return glm::length(glm::max(q, 0.0f)) + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

__host__ __device__
float udfBox(glm::vec3 p, glm::vec3 minCorner, glm::vec3 maxCorner)
{
    glm::vec3 center = (maxCorner + minCorner) / 2.f;
    glm::vec3 halfScale = maxCorner - center;
    return sdBox(p - center, halfScale);
}


__global__ void generateSDFwithBVH(const SDF* sdf, SDFGrid* SDFGrids, const BVHNode* bvhNodes, const int bvhNodes_size, const Geom* geoms)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (x >= sdf->resolution.x || y >= sdf->resolution.y || z >= sdf->resolution.z) return;
    int idx = z * sdf->resolution.x * sdf->resolution.y + y * sdf->resolution.x + x;


    glm::vec3 voxelPos = glm::vec3(sdf->minCorner.x + sdf->gridExtent.x * (float(x) + 0.5f),
                                   sdf->minCorner.y + sdf->gridExtent.y * (float(y) + 0.5f),
                                   sdf->minCorner.z + sdf->gridExtent.z * (float(z) + 0.5f));
    
    // find closest triangle
    int currIdx = 0;

    int idxToVisit[64];
    int toVisitOffset = 0;

    idxToVisit[toVisitOffset++] = 0;

    const Triangle* minTriangle = nullptr;
    float minUdf = FLT_MAX;

    while (toVisitOffset > 0)
    {
        currIdx = idxToVisit[--toVisitOffset];
        //if (currIdx >= bvhNodes_size) continue;

        if (!bvhNodes[currIdx].isLeaf)
        {
            /*float rayLength = boxIntersectionTest(bvhNodes[currIdx].minCorner, bvhNodes[currIdx].maxCorner, r);
            if (rayLength == -1.f || rayLength > minUdf) continue;*/

            float distToBox = udfBox(voxelPos, bvhNodes[currIdx].minCorner, bvhNodes[currIdx].maxCorner);
            if (distToBox > minUdf) continue;

            int leftIdx = currIdx * 2 + 1;
            int rightIdx = leftIdx + 1;

            idxToVisit[toVisitOffset++] = rightIdx;
            idxToVisit[toVisitOffset++] = leftIdx;
        }
        else
        {
            if (!bvhNodes[currIdx].hasFace) continue;
            const Triangle* face = &bvhNodes[currIdx].face;
            float t = udfTriangle(voxelPos, face->point1, face->point2, face->point3);

            if (minUdf > t)
            {
                minTriangle = &bvhNodes[currIdx].face;
                minUdf = t;
            }
        }
    }


    // will crash without this
    if (minTriangle == nullptr) return;
    int geomId = minTriangle->geomId;

    glm::vec3 triCenter = (minTriangle->point1 + minTriangle->point2 + minTriangle->point3) / 3.f;
    glm::vec3 worldNor = glm::normalize(triCenter - voxelPos);
    glm::vec3 localNor = multiplyMV(geoms[geomId].inverseTransform, glm::vec4(worldNor, 0.f));

    glm::vec3 localtriNor = multiplyMV(geoms[geomId].inverseTransform, glm::vec4(minTriangle->normal1, 0.f));
    // if inside
    if (glm::dot(localNor, localtriNor) > 0.f)
    {
        SDFGrids[idx].dist = -minUdf;
    }
    else
    {
        SDFGrids[idx].dist = minUdf;
    }

    SDFGrids[idx].geomId = minTriangle->geomId;
}


// brute force
__global__ void generateSDF(const SDF* sdf, SDFGrid* SDFGrids, const Triangle* triangles, const int triangles_size, const Geom* geoms)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int z = (blockIdx.z * blockDim.z) + threadIdx.z;

    if (x >= sdf->resolution.x || y >= sdf->resolution.y || z >= sdf->resolution.z) return;
    int idx = z * sdf->resolution.x * sdf->resolution.y + y * sdf->resolution.x + x;

    glm::vec3 voxelPos = glm::vec3(sdf->minCorner.x + sdf->gridExtent.x * (float(x) + 0.5f),
                                   sdf->minCorner.y + sdf->gridExtent.y * (float(y) + 0.5f),
                                   sdf->minCorner.z + sdf->gridExtent.z * (float(z) + 0.5f));
    const Triangle* minTriangle = nullptr;

    float minUdf = FLT_MAX;
    for (int i = 0; i < triangles_size; i++)
    {
        float udf = udfTriangle(voxelPos, triangles[i].point1, triangles[i].point2, triangles[i].point3);
        if (udf < minUdf)
        {
            minUdf = udf;
            minTriangle = triangles + i;
        }
    }

    /*if (minUdf < 0.001f)
    {
        printf("%d, %d, %d, %.2f, %.2f, %.2f, %f\n", x, y, z, a, b, c, minUdf);
    }*/

    // will crash without this
    if (minTriangle == nullptr) return;
    int geomId = minTriangle->geomId;

    glm::vec3 triCenter = (minTriangle->point1 + minTriangle->point2 + minTriangle->point3) / 3.f;
    glm::vec3 worldNor = glm::normalize(triCenter - voxelPos);
    glm::vec3 localNor = multiplyMV(geoms[geomId].inverseTransform, glm::vec4(worldNor, 0.f));

    glm::vec3 localtriNor = multiplyMV(geoms[geomId].inverseTransform, glm::vec4(minTriangle->normal1, 0.f));
    // if inside
    if (glm::dot(localNor, localtriNor) > 0.f)
    {
        SDFGrids[idx].dist = -minUdf;
        //printf("%d, %d, %d\n", x, y, z);
    }
    else
    {
        SDFGrids[idx].dist = minUdf;
    }

    ////////////////// ??
    if (minUdf < 0.5f)
    {
        SDFGrids[idx].geomId = minTriangle->geomId;
    }
    else
    {
        SDFGrids[idx].geomId = -1;
    }

    // ?? bug ??
    SDFGrids[idx].geomId = minTriangle->geomId;
    ////////////////

    //printf("%d, %d, %d, %.2f\n", x, y, z, SDFGrids[idx].dist);
    
}



__host__ __device__ const SDFGrid* sceneSDF(glm::vec3 pos, const SDF* sdf, const SDFGrid* SDFGrids)
{
    glm::ivec3 gridCoord = glm::floor((pos - sdf->minCorner) / sdf->gridExtent);
    int idx = gridCoord.z * sdf->resolution.x * sdf->resolution.y + gridCoord.y * sdf->resolution.x + gridCoord.x;
    if (gridCoord.x < 0 || gridCoord.y < 0 || gridCoord.z < 0
        || gridCoord.x >= sdf->resolution.x || gridCoord.y >= sdf->resolution.y || gridCoord.z >= sdf->resolution.z) return nullptr;

    return SDFGrids + idx;
}


__host__ __device__ glm::vec3 estimateNormal(glm::vec3 p, const SDF* sdf, const SDFGrid* SDFGrids)
{
    const SDFGrid* currGrid = sceneSDF(p, sdf, SDFGrids);
    if (currGrid == nullptr) return glm::vec3(0);
    const SDFGrid* dx1Grid = sceneSDF(glm::vec3(p.x + sdf->gridExtent.x, p.y, p.z), sdf, SDFGrids);
    float dx1 = dx1Grid == nullptr ? currGrid->dist : dx1Grid->dist;
    const SDFGrid* dx2Grid = sceneSDF(glm::vec3(p.x - sdf->gridExtent.x, p.y, p.z), sdf, SDFGrids);
    float dx2 = dx2Grid == nullptr ? currGrid->dist : dx2Grid->dist;
    const SDFGrid* dy1Grid = sceneSDF(glm::vec3(p.x, p.y + sdf->gridExtent.y, p.z), sdf, SDFGrids);
    float dy1 = dy1Grid == nullptr ? currGrid->dist : dy1Grid->dist;
    const SDFGrid* dy2Grid = sceneSDF(glm::vec3(p.x, p.y - sdf->gridExtent.y, p.z), sdf, SDFGrids);
    float dy2 = dy2Grid == nullptr ? currGrid->dist : dy2Grid->dist;
    const SDFGrid* dz1Grid = sceneSDF(glm::vec3(p.x, p.y, p.z + sdf->gridExtent.z), sdf, SDFGrids);
    float dz1 = dz1Grid == nullptr ? currGrid->dist : dz1Grid->dist;
    const SDFGrid* dz2Grid = sceneSDF(glm::vec3(p.x, p.y, p.z - sdf->gridExtent.z), sdf, SDFGrids);
    float dz2 = dz2Grid == nullptr ? currGrid->dist : dz2Grid->dist;

    return glm::normalize(glm::vec3(
        dx1 - dx2,
        dy1 - dy2,
        dz1 - dz2
    ));
}


__host__ __device__ float sdfIntersectionTest(const Geom* geoms, const SDF* sdf, const SDFGrid* SDFGrids,
    const Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, int* hitGeomId)
{
    const SDFGrid* startGrid = sceneSDF(r.origin, sdf, SDFGrids);

    //if (startGrid == nullptr) return -1.f;

    if (startGrid == nullptr || startGrid->dist < 0.f) outside = false;
    else outside = true;


    normal = glm::vec3(0.f, 0.f, 0.f);
    //normal = estimateNormal(r.origin, sdf, SDFGrids);

    //float t = startGrid->dist;
    float t = 0.05f;

    int maxMarchSteps = 64;
    glm::vec3 lastRayMarchPos = r.origin;
    int lastGeomId = -1;
    for (int i = 0; i < maxMarchSteps; i++)
    {
        glm::vec3 rayMarchPos = r.origin + r.direction * t;
        const SDFGrid* currSDFGrid = sceneSDF(rayMarchPos, sdf, SDFGrids);
        
        // ??
        if (currSDFGrid == nullptr)
        {
            //return -1.f;
            t += 0.7f;
            continue;
        }

        if (currSDFGrid->dist < 0.01f)
        {
            intersectionPoint = lastRayMarchPos;
            normal = estimateNormal(lastRayMarchPos, sdf, SDFGrids);
            *hitGeomId = lastGeomId;
            
            return t;
        }

        // Move along the view ray
        t += currSDFGrid->dist;
        lastGeomId = currSDFGrid->geomId;
        lastRayMarchPos = rayMarchPos;

        if (t >= 100.f)
        {
            // Gone too far; give up
            return 100.f;
        }
    }


    return -1.f;
}

#endif