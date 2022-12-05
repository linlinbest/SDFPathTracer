#pragma once

#include "intersections.h"
#include "sceneStructs.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 reflectDir;
    glm::vec3 refractDir;
    float etaI = 0.f;
    float etaT = 0.f;
    float cosTheta = 0.f;
    if (m.hasRefractive)
    {
        cosTheta = glm::dot(pathSegment.ray.direction, normal);

        bool entering = cosTheta < 0.f;
        etaI = entering ? 1.f : m.indexOfRefraction;
        etaT = entering ? m.indexOfRefraction : 1.f;
        if (!entering) normal = -normal;

        refractDir = glm::refract(pathSegment.ray.direction, normal, etaI / etaT);
    }
    if (m.hasReflective)
    {
        // specular reflection
        reflectDir = glm::reflect(pathSegment.ray.direction, normal);
    }

    if (m.hasRefractive && m.hasReflective)
    {
        float r0 = (etaT - etaI) / (etaT + etaI);
        r0 *= r0;
        float cosThetaPlus1 = 1.f + cosTheta;
        float pReflect = r0 + (1 - r0) * cosThetaPlus1 * cosThetaPlus1 * cosThetaPlus1 * cosThetaPlus1 * cosThetaPlus1;

        thrust::uniform_real_distribution<float> u01(0, 1);
        float probability = u01(rng);
        if (probability < pReflect)
        {
            pathSegment.ray.direction = reflectDir;
            pathSegment.ray.origin = intersect;
        }
        else
        {
            pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
            pathSegment.ray.direction = refractDir;
        }
        pathSegment.color *= m.specular.color;

    }
    else if (m.hasRefractive)
    {
        pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
        pathSegment.ray.direction = refractDir;
        //pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
        pathSegment.color *= m.specular.color;
    }
    else if (m.hasReflective)
    {
        pathSegment.ray.direction = reflectDir;
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect;
    }
    else
    {
        // pure diffuse reflection
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect;
    }

}
//Added by Hanlin
__host__ __device__
float getLegendrePolynomialsValue(int index, float input)
{
    float result = input;
    //This need to be put into Legendre Polynomials

    switch (index)
    {
    case 0:
        //(m=0,l=0)
        //p(0,0)=1
        result = 1;
        break;
    case 1:
        //(m=-1,l=1)
        result = -0.5f * sqrt(pow(input, 2.f) - 1.f);
        break;
    case 2:
        //(m=0,l=1)
        result = input;
        break;
    case  3:
        //(m=1,l=1)
        result = sqrt(pow(input, 2.f) - 1.f);
        break;
    case  4:
        //(m=-2,l=2)
        result = 0.125f * (1.f - pow(input, 2.f));
        break;
    case  5:
        //(m=-1,l=2)
        result = -0.5f * input * sqrt(pow(input, 2.f) - 1.f);
        break;
    case  6:
        //(m=0,l=2)
        result = 0.5f * (3.f * pow(input, 2.f) - 1.f);
        break;
    case  7:
        //(m=1,l=2)
        result = 3.f * input * sqrt(pow(input, 2.f) - 1.f);
        break;
    case  8:
        //(m=2,l=2)
        result = 3.f - 3.f * pow(input, 2.f);
        break;
    }
    return result;
}

//return H(m,l) 
__host__ __device__
float getHemisphereHarmonicBasis(const int index, const float theta,const float phi)
{
    float result = 0;
    float factor = 2.f * cos(theta) - 1.f;
    printf("Result: %f \n", result);
    switch (index)
    {
    case 0:
        //£¨m=0,l=0£©
        result = HemisphereHarmonicCoefficient[index] * getLegendrePolynomialsValue(index, cos(theta));
        break;
    case  1:
        //(m=-1,l=1)
        result =float(sqrt(2.f)) * HemisphereHarmonicCoefficient[index] * sin(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case 2:
        //(m=0,l=1)
        result = HemisphereHarmonicCoefficient[index] * getLegendrePolynomialsValue(index, cos(theta));
        break;
    case  3:
        //(m=1,l=1)
        result = float(sqrt(2.f)) * HemisphereHarmonicCoefficient[index] * cos(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case  4:
        //(m=-2,l=2)
        result = float(sqrt(2.f)) * HemisphereHarmonicCoefficient[index] * sin(2 * phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case 5:
        //(m=-1,l=2)
        result = float(sqrt(2.f)) * HemisphereHarmonicCoefficient[index] * sin(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case  6:
        //(m=0,l=2)
        result = HemisphereHarmonicCoefficient[index] * getLegendrePolynomialsValue(index, cos(theta));
        break;
    case  7:
        //(m=1,l=2)
        result = float(sqrt(2.f)) * HemisphereHarmonicCoefficient[index] * cos(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case  8:
        //(m=2,l=2)
        result = float(sqrt(2.f)) * HemisphereHarmonicCoefficient[index] * cos(2 * phi) * getLegendrePolynomialsValue(index, factor);
        break;
    }

    return result;
}

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

    glm::vec3 derivative_lamda_x[9];
    glm::vec3 derivative_lamda_y[9];
    for (int i = 0; i < 9; i++)
    {
        lamda[i] = glm::vec3(0, 0, 0);
        derivative_lamda_x[i] = glm::vec3(0, 0, 0);
        derivative_lamda_y[i] = glm::vec3(0, 0, 0);
    }
    //printf("Lamda: %f, %f, %f \n", lamda[4].x, lamda[4].y, lamda[4].z);
    // generate n sample light

    //N is SAMPLE_COUNT
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        //generate random sample ray direction
        //This is to compute radiance cache

        //Turn it into sphere coordinates
        //Get sample ray's intersection lighting results

        Ray newRay;
        newRay.direction = calculateRandomDirectionInHemisphere(normal, rng);
        newRay.direction = glm::normalize(newRay.direction);
        //Get the generated rayDir's intersection BSDF cache
        newRay.origin = intersect;
       // printf("newRay origin: %f,%f,%f \n", newRay.origin.x, newRay.origin.y, newRay.origin.z);
     //   printf("newRay.direction: %f, %f, %f\n:", newRay.direction.x,newRay.direction.y,newRay.direction.z);


        PathSegment newPath;
        newPath.ray = newRay;

        ShadeableIntersection newRayIntersection;
        newRayIntersection.t = FLT_MAX;
        newRayIntersection.surfaceNormal = glm::vec3(0, 0, 0);
        newRayIntersection.materialId = 0;
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
        //    printf("geom.materialID: %d\n:", geom.materialid);
            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, newRay, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
              //  printf("Hit test 2\n");
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
                //printf("Intersect with geom \n");
                t_min = t;
                hit_geom_index = i;
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
        Material intersectMat = materials[newRayIntersection.materialId];

        glm::vec3 q_k_normal = newRayIntersection.surfaceNormal;

        //launched new ray got diffuse irradiance
        //printf(" intersectMat Col %f, %f %f \n:", intersectMat.color.r, intersectMat.color.g, intersectMat.color.b);
        //Get the generated rayDir's intersection BSDF cache
        //First need to convert sample direction it into sphere coordinate

        float r_k = sqrt(pow(newRay.direction.x, 2) + pow(newRay.direction.y, 2) + pow(newRay.direction.z, 2));
        float theta_k = acos(newRay.direction.z / r_k);
        float phi_k = acos(newRay.direction.x / (r_k * sin(theta_k)));

        //check again
        glm::vec3 q_k = getPointOnRay(newRay, t);
        float test_r_k = glm::length(q_k - newRay.origin);
        glm::vec3 direction = q_k - newRay.origin;
        float test_r = glm::length(q_k - newRay.origin);
        float test_theta_k = acos(direction.z / test_r);
        float test_phi_k = acos(direction.x / (test_r * sin(test_theta_k)));

        //compute derivative x
        float derivative_theta_k_x = -cos(theta_k) * cos(phi_k) / r_k;
        float derivative_phi_k_x = sin(phi_k) / (r_k * sin(theta_k));

        glm::vec3 normalized_q_k_normal = glm::normalize(q_k_normal);

       glm::vec3 normalized_direction = glm::normalize(direction);
        float cos_ep_k = -glm::dot(normalized_q_k_normal, normalized_direction);

        float derivative_omega_k_x = ((2 * PI) / SAMPLE_COUNT) * ((test_r_k * q_k_normal.x + 3 * direction.x * cos_ep_k) / (pow(test_r_k, 2.f) * cos_ep_k));


     //compute derivative y
        float derivative_theta_k_y = -(cos(theta_k) * sin(phi_k)) / r_k;
        float derivative_phi_k_y = -cos(phi_k) / (r_k * sin(theta_k));
     // 
       // printf(" test sphere %f, %f %f \n:", r, theta, phi);
        //need to convert raydir into hemisphere theta and phi
        //Compute lamda(m,l)(have 9 in total since second order)
        for (int n = 0; n < 9; n++)
        {
            lamda[n] = intersectMat.color * getHemisphereHarmonicBasis(n, theta_k, phi_k);
            float derivative_H_x = derivative_theta_k_x * getDerivativeH_Theta(n, theta_k, phi_k) + derivative_phi_k_x * getDerivativeH_Phi(n, theta_k, phi_k);
            derivative_lamda_x[n] = intersectMat.color * (derivative_omega_k_x * getHemisphereHarmonicBasis(n, theta_k, phi_k) + (2 * PI) / SAMPLE_COUNT * derivative_H_x);
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
        //Seems to have problem here
        radianceCache.radianceHSH += lamda[i] * getHemisphereHarmonicBasis(i, ray_theta, ray_phi);
    }
    //if (radianceCache.radianceHSH != glm::vec3(0, 0, 0))
    //{
    //    printf(" radianceCache.radianceHSH %f, %f %f \n:", radianceCache.radianceHSH.x, radianceCache.radianceHSH.y, radianceCache.radianceHSH.z);
    //}
   // printf(" radianceCache.radianceHSH %f, %f %f \n:", radianceCache.radianceHSH.x, radianceCache.radianceHSH.y, radianceCache.radianceHSH.z);
    radianceCache.position = intersect;
}
__host__ __device__
float getDerivativeH_Theta(int index,int theta,int phi)
{

}

__host__ __device__
float getDerivativeH_Phi(int index, int theta, int phi)
{

}