#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT celPayload prd;

layout(push_constant) uniform Constants
{
  vec4 clearColor;
};

void main()
{
  prd.normal = vec3(0);
  prd.depth  = -1;
  prd.object = 0;
  prd.celid  = 0;
}
