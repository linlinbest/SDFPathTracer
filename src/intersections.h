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


//__global__ void generateSDF(const SDF* sdf, SDFGrid* SDFGrids, const BVHNode* bvhNodes, const int bvhNodes_size)
//{
//    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//    int z = (blockIdx.z * blockDim.z) + threadIdx.z;
//
//    int idx = z * sdf->resolution.x * sdf->resolution.y + y * sdf->resolution.x + x;
//
//
//    SDFGrids[idx].material;
//}


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
    float t = 0.2f;

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
            t += 0.5f;
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

#if 1

//For incoming point radiance cache
__host__ __device__
void precomputeRadianceCache(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    RadianceCache& radianceCache,
    Geom* geoms,
    int geom_Size,
    //mesh faces
    Triangle* faces,
    int faces_size,
    Material* materials,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float factor = 2 * PI / SAMPLE_COUNT;

    glm::vec3 lamda[9];
    for (int i = 0; i < 9; i++)
    {
        lamda[i] = glm::vec3(0, 0, 0);
    }

    // generate n sample light
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        //generate random sample ray direction
        //This is to compute radiance cache

        //Turn it into sphere coordinates
        //Get sample ray's intersection lighting results

        float up = sqrt(u01(rng)); // cos(theta)
        float over = sqrt(1 - up * up); // sin(theta)
        float around = u01(rng) * TWO_PI;

        glm::vec3 directionNotNormal;
        if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
            directionNotNormal = glm::vec3(1, 0, 0);
        }
        else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
            directionNotNormal = glm::vec3(0, 1, 0);
        }
        else {
            directionNotNormal = glm::vec3(0, 0, 1);
        }

        // Use not-normal direction to generate two perpendicular directions
        glm::vec3 perpendicularDirection1 =
            glm::normalize(glm::cross(normal, directionNotNormal));
        glm::vec3 perpendicularDirection2 =
            glm::normalize(glm::cross(normal, perpendicularDirection1));

        glm::vec3 rayDir = up * normal
            + cos(around) * over * perpendicularDirection1
            + sin(around) * over * perpendicularDirection2;

        //Get the generated rayDir's intersection BSDF cache
        Ray newRay;
        newRay.direction = rayDir;
        newRay.origin = (intersect)+0.0001f * rayDir;


        PathSegment newPath;
        newPath.ray = newRay;
        ShadeableIntersection newRayIntersection;
        newRayIntersection.t = FLT_MAX;
        newRayIntersection.surfaceNormal = glm::vec3(0, 0, 0);
        newRayIntersection.materialId = -1;
        //Only bounce one time
        //Remember here only want diffuse color
        //begin compute intersection(remember to use new ray)

        float t;
        glm::vec3 intersect_point;
        glm::vec3 new_normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geom_Size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, newRay, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, newRay, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                t = meshIntersectionTest(geom, faces, newRay, tmp_intersect, tmp_normal, outside);
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1 || t_min == FLT_MAX)
        {
            newRayIntersection.t = -1.0f;
        }
        else
        {
            //The ray hits something
            newRayIntersection.t = t_min;
            newRayIntersection.materialId = geoms[hit_geom_index].materialid;
            newRayIntersection.surfaceNormal = normal;
        }

        //launched new ray got diffuse irradiance

        //Get the generated rayDir's intersection BSDF cache
        //First need to convert sample direction it into sphere coordinate
        float r = sqrt(pow(rayDir.x, 2) + pow(rayDir.y, 2) + pow(rayDir.z, 2));
        float theta = acos(rayDir.z / r);
        float phi = acos(rayDir.x / (r * sin(theta)));

        //need to convert raydir into hemisphere theta and phi
        //Compute lamda(m,l)(have 9 in total since second order)
        for (int n = 0; n < 9; n++)
        {
            lamda[n] = newPath.color * getHemisphereHarmonicBasis(n, theta, phi);
        }
    }

    for (int i = 0; i < 9; i++)
    {
        lamda[i] *= factor;
    }

    // Now have lamda, can compute ray radiance 
    //
    float ray_r = sqrt(pow(pathSegment.ray.direction.x, 2) + pow(pathSegment.ray.direction.y, 2) + pow(pathSegment.ray.direction.z, 2));
    float ray_theta = acos(pathSegment.ray.direction.z / ray_r);
    float ray_phi = acos(pathSegment.ray.direction.x / (ray_r * sin(ray_theta)));

    for (int i = 0; i < 9; i++)
    {
        radianceCache.radianceHSH += lamda[i] * getHemisphereHarmonicBasis(i, ray_theta, ray_phi);
    }
    radianceCache.position = intersect;
}

#endif