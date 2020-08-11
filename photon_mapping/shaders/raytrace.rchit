#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#include "random.glsl"
#include "raycommon.glsl"
#include "wavefront.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT photonPayload prd;


layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
layout(binding = 3, set = 1) uniform sampler2D textureSamplers[];
layout(binding = 4, set = 1)  buffer MatIndexColorBuffer { int i[]; } matIndex[];
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
  int   numAreaLights;
  int   maxBounces;
  float maxRussian;
  int   numPointLights;
  int   numIds;
  int   celsteps;
  float celramp;
  float r;
  float cut;
  float maxillum;
  int   obid;
  int   pass;
  int   offset;
}
pushC;

layout(location = 0) callableDataEXT materialCall mc;


const float tMin = 0.0001;


void main()
{


  const uint flags =
      gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  // Object of this instance
  const uint objId = scnDesc.i[gl_InstanceID].objId;
  //init_rnd(prd.seed);

  // Indices of the triangle
  const ivec3 ind = ivec3(indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 0],   //
                          indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 1],   //
                          indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 2]);  //
  // Vertex of the triangle
  const Vertex v0 = vertices[nonuniformEXT(objId)].v[ind.x];
  const Vertex v1 = vertices[nonuniformEXT(objId)].v[ind.y];
  const Vertex v2 = vertices[nonuniformEXT(objId)].v[ind.z];

  const int matProb = v0.mat;

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Computing the normal at hit position
  vec3 snormal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  vec3 gnormal = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
  // Transforming the normal to world space
  snormal = normalize(vec3(scnDesc.i[gl_InstanceID].transfoIT * vec4(snormal, 0.0)));
  gnormal = normalize(vec3(scnDesc.i[gl_InstanceID].transfoIT * vec4(gnormal, 0.0)));

  bool entering = dot(gl_WorldRayDirectionEXT, gnormal) < 0;

  mc.entering = entering;
  gnormal     = entering ? gnormal : -gnormal;
  snormal     = entering ? snormal : -snormal;

  const vec3 worldPos = offset_ray(
      vec3(
          scnDesc.i[gl_InstanceID].transfo
          * vec4(v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z, 1.0)),
      gnormal);


  //Material of the object
  const int               matIdx = matIndex[nonuniformEXT(objId)].i[gl_PrimitiveID];
  const WaveFrontMaterial mat    = materials[nonuniformEXT(objId)].m[matIdx];


  mc.objId  = objId;
  mc.pId    = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  mc.normal = snormal;


  bool diffuse  = (matProb & 1) != 0;
  bool specular = (matProb & 12) != 0;
  bool glossy   = (matProb & 2) != 0;


  float ww   = 1.0;
  int   mask = matProb;
  if(specular && diffuse)
  {
    float r = rnd(prd.seed);
    mask    = r > 0.5 ? (mask & 12) : (mask & 3);
    ww *= 2;
  }


  switch(mask)
  {
    case 1: {
      mask = 0;
      break;
    }
    case 2: {
      mask = 1;
      break;
    }
    case 3: {
      mask = rnd(prd.seed) > 0.5 ? 1 : 0;
      ww *= 2;
      break;
    }
    case 4: {
      mask = 2;
      break;
    }
    case 8: {
      mask = 3;
      break;
    }
    case 12: {
      mask = rnd(prd.seed) > 0.5 ? 3 : 2;
      ww *= 2;
      break;
    }
    case 16: {
      mask = 0;
      break;
    }
    case 32: {
      mask = pushC.pass == 0 ? 8 : 0;
      ww   = 1;
      break;
    }
  }

  prd.emplace = mask <= 0 || mask > 3;

  mc.objId  = objId;
  mc.pId    = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  mc.normal = snormal;

  mc.position = worldPos;

  mc.fuzzyAngle = pushC.fuzzyAngle;
  mc.ior        = pushC.ior;
  mc.inDir;


  mc.inDir      = vec3(1, 0, 0);
  mc.eval_color = vec3(0, 0, 0);

  mc.seed = prd.seed;
  executeCallableEXT(mask, 0);

  prd.seed = mc.seed;


  const float p = russian_roulette(prd.color);
  if(rnd(prd.seed) > p)
  {
    prd.done = true;
    return;
  }


  prd.color *= mc.sample_color / (mc.sample_pdf * p);
  //prd.color = prd.weight;


  prd.rayOrigin    = worldPos;
  prd.rayDirection = mc.sample_in;
}