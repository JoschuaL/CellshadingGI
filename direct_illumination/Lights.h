//
// Created by kano on 06.06.20.
//

#ifndef VK_RAYTRACING_TUTORIAL_AREALIGHT_H
#define VK_RAYTRACING_TUTORIAL_AREALIGHT_H


#include <nvmath/nvmath.h>
struct AreaLight
{
  nvmath::vec4f color;
  nvmath::vec4f v1;
  nvmath::vec4f v2;
  nvmath::vec3f v3;
  int last = false;

};

struct PointLight
{
  nvmath::vec4f color;
  nvmath::vec4f pos;
};


#endif  //VK_RAYTRACING_TUTORIAL_AREALIGHT_H
