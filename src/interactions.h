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
        result = -0.5f * sqrt(pow(input, 2) - 1);
        break;
    case 2:
        //(m=0,l=1)
        result = input;
        break;
    case  3:
        //(m=1,l=1)
        result = sqrt(pow(input, 2) - 1);
        break;
    case  4:
        //(m=-2,l=2)
        result = 0.125f * (1 - pow(input, 2));
        break;
    case  5:
        //(m=-1,l=2)
        result = -0.5f * input * sqrt(pow(input, 2) - 1);
        break;
    case  6:
        //(m=0,l=2)
        result = 0.5f * (3 * pow(input, 2) - 1);
        break;
    case  7:
        //(m=1,l=2)
        result = 3.f * input * sqrt(pow(input, 2) - 1);
        break;
    case  8:
        //(m=2,l=2)
        result = 3.f - 3.f * pow(input, 2);
        break;
    }
    return result;
}

//return H(m,l) 
float getHemisphereHarmonicBasis(int index, float theta, float phi)
{
    float result = 0;
    float factor = 2 * cos(theta) - 1;
    switch (index)
    {
    case 0:
        //£¨m=0,l=0£©
        result = HemisphereHarmonicCoefficient[0] * getLegendrePolynomialsValue(index, cos(theta));
        break;
    case  1:
        //(m=-1,l=1)
        result = sqrt(2) * HemisphereHarmonicCoefficient[index] * sin(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case 2:
        //(m=0,l=1)
        result = HemisphereHarmonicCoefficient[index] * getLegendrePolynomialsValue(index, cos(theta));
        break;
    case  3:
        //(m=1,l=1)
        result = sqrt(2) * HemisphereHarmonicCoefficient[index] * cos(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case  4:
        //(m=-2,l=2)
        result = sqrt(2) * HemisphereHarmonicCoefficient[index] * sin(2 * phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case 5:
        //(m=-1,l=2)
        result = sqrt(2) * HemisphereHarmonicCoefficient[index] * sin(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case  6:
        //(m=0,l=2)
        result = HemisphereHarmonicCoefficient[index] * getLegendrePolynomialsValue(index, cos(theta));
        break;
    case  7:
        //(m=1,l=2)
        result = sqrt(2) * HemisphereHarmonicCoefficient[index] * cos(phi) * getLegendrePolynomialsValue(index, factor);
        break;
    case  8:
        //(m=2,l=2)
        result = sqrt(2) * HemisphereHarmonicCoefficient[index] * cos(2 * phi) * getLegendrePolynomialsValue(index, factor);
        break;
    }
    return result;
}


_