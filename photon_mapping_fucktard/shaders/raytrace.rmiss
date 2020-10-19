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
  prd.weight   = vec3(0);
  prd.done     = true;
  prd.gnrm     = vec3(0);
  prd.snrm     = vec3(0);
  prd.photons  = false;
  prd.material = -1;
}
