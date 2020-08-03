#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#include "random.glsl"
#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;


layout(location = 0) rayPayloadInEXT celPayload prd;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;


layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;

layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];


// clang-format on

layout(push_constant) uniform Constants
{
  vec4  clearColor;
  vec4  lightColor;
  vec4  lightPosition;
  int   numObjs;
  int   numAreaSamples;
  int   frame;
  int   numSamples;
  float fuzzyAngle;
  float ior;
  int numPointLights;
}
pushC;




void main()
{
   
 

  // Object of this instance
  const uint objId = scnDesc.i[gl_InstanceID].objId;
  //init_rnd(prd.seed);

  // Indices of the triangle
  const ivec3 ind = ivec3(indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 0],   //
                    indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 1],   //
                    indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 2]);  //
  // Vertex of the triangle
  Vertex v0 = vertices[nonuniformEXT(objId)].v[ind.x];
  Vertex v1 = vertices[nonuniformEXT(objId)].v[ind.y];
  Vertex v2 = vertices[nonuniformEXT(objId)].v[ind.z];


  prd.object = (v0.mat & 32) != 0 ? v0.id : 0;
  prd.depth =  (v0.mat & 32) != 0 ? gl_HitTEXT : 0;
  prd.celid = (v0.mat & 32) != 0 ? v0.celid : 0;
  

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Computing the normal at hit position
  vec3 snormal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;

  // Transforming the normal to world space
  snormal       = normalize(vec3(scnDesc.i[gl_InstanceID].transfoIT * vec4(snormal, 0.0)));
  const float ins = dot(gl_WorldRayDirectionEXT, snormal);

  snormal *= -sign(ins);

  prd.normal =  (v0.mat & 32) != 0 ?  (snormal + vec3(1)) / 2 : vec3(0);

  

  
}
