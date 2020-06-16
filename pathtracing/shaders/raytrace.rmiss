#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform Constants
{
  vec4 clearColor;
};

void main()
{

  //prd.color = clearColor.xyz;
  prd.done = true;
  prd.weight = vec3(0);
  prd.color = vec3(0);
}
