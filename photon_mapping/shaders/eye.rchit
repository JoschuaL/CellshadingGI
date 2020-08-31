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
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
layout(binding = 3, set = 1) uniform sampler2D textureSamplers[];
layout(binding = 4, set = 1)  buffer MatIndexColorBuffer { int i[]; } matIndex[];
layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];
layout(binding = 7, set = 1) buffer AreaLightsBuffer { AreaLight l[]; } lights;
layout(binding = 8, set = 1) buffer PointLightsBuffer { PointLight l[]; } Plights;
layout(binding = 9, set = 1) buffer CelInfo { celinfo i[]; } CelInfos[];

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
layout(location = 1) callableDataEXT emissionCall ec;
layout(location = 2) callableDataEXT directSampleCall dsc;


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
  prd.snrm    = snormal;
  prd.gnrm    = gnormal;
  prd.photons = true;
  return;

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

  /*if(pushC.pass == 0)
  {
    prd.done  = true;
    float c   = ceil(abs(dot(mc.celdir, mc.normal)) * pushC.celsteps) / pushC.celsteps;
    prd.color = c < pushC.celramp ? vec3(1, 0, 0) : vec3(0, 1, 0);

    return;
  }*/


  /* mc.objId  = objId;
  mc.pId    = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  mc.normal        = gn;
  mc.outDir        = gl_WorldRayDirectionEXT;

  mc.emission      = vec3(0, 0, 0);*/


  if((matProb & 16) != 0)
  {

    const int lightType = 1;
    const int call      = 4 + lightType;
    ec.dir              = -gl_WorldRayDirectionEXT;
    ec.li = AreaLight(vec4(mat.emission, 1), vec4(v0.pos, 0), vec4(v1.pos, 0), vec4(v2.pos, 0));
    executeCallableEXT(call, 1);
    if(prd.specular || prd.first)
    {
      prd.color += ec.intensity * prd.weight;
    }
    else
    {
      const vec3  r = gl_WorldRayOriginEXT - worldPos;
      const float pne =
          (ec.pdf_area * dot(r, r))
          / (dot(-gl_WorldRayDirectionEXT, snormal) * (pushC.numAreaLights + pushC.numPointLights));
      const float p_bsdf     = prd.last_bsdf_pdf;
      const float mis_weight = p_bsdf / (p_bsdf + pne);
      prd.color += ec.intensity * mis_weight * prd.weight;
    }

    prd.done    = true;
    prd.photons = false;
    return;
  }


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
      mask = 8;
      break;
    }
  }

  prd.material = mask;

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

  int lightType;
  if(mask == 1)
  {


    const int cl = min(int(rnd(prd.seed) * (pushC.numAreaLights + pushC.numPointLights)),
                       pushC.numAreaLights + pushC.numPointLights - 1);

    if(cl < pushC.numAreaLights)
    {
      const AreaLight li   = lights.l[cl];
      const vec3      e1   = li.v1.xyz - li.v0.xyz;
      const vec3      e2   = li.v2.xyz - li.v0.xyz;
      const vec3      ln   = normalize(cross(e1, e2));
      const float     u    = rnd(prd.seed);
      const float     v    = rnd(prd.seed);
      const vec3      lpos = u + v < 1.0 ? li.v0.xyz + (u * e1) + (v * e2) :
                                      li.v0.xyz + ((1.0 - u) * e1) + ((1.0 - v) * e2);
      const vec3 pos = offset_ray(offset_ray(lpos, ln), ln);

      dsc.pos   = pos;
      dsc.li    = li;
      lightType = 1;
    }
    else
    {

      const PointLight p = Plights.l[cl - pushC.numAreaLights];

      dsc.p   = p;
      dsc.pos = p.pos.xyz;

      lightType = 0;
    }


    dsc.seed = prd.seed;

    dsc.from = worldPos;


    const int call = 6 + lightType;
    executeCallableEXT(call, 2);
    prd.seed = dsc.seed;
  }

  const vec3  dir     = worldPos - dsc.pos;
  const float d2      = dot(dir, dir);
  const float dist    = sqrt(d2);
  const vec3  rayDir  = dir / dist;
  const float cos_hit = dot(rayDir, gnormal);
  mc.inDir            = -rayDir;
  mc.eval_color       = vec3(0, 0, 0);

  mc.seed = prd.seed;
  executeCallableEXT(mask, 0);

  prd.seed = mc.seed;


  isShadowed = true;

  if(mask == 1)
  {
    traceRayEXT(topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                dsc.pos,     // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                dist,        // ray max range
                1            // payload (location = 1)
    );
  }
  if(!isShadowed)
  {
    const float pne = ((dsc.pdf_area * d2) / (dsc.cos_v * pushC.numAreaLights));


    const float p_bsdf = mc.pdf_pdf;

    const float mis_weight = lightType == 1 ? pne / (pne + p_bsdf) : 1.0;


    prd.color +=
        ww * dsc.intensity * mis_weight
        * ((prd.weight * mc.eval_color * dsc.cos_v * (pushC.numAreaLights + pushC.numPointLights))
           / (d2 * dsc.pdf_area));
  }


  const float p = russian_roulette(prd.weight);
  if(rnd(prd.seed) > p || mc.sample_color == vec3(0, 0, 0) || mc.sample_pdf <= 0.0)
  {
    prd.done    = true;
    prd.photons = false;
    return;
  }


  prd.last_bsdf_pdf = mc.sample_pdf;


  prd.weight *= mc.sample_color / (mc.sample_pdf * p);
  //prd.color = prd.weight;


  prd.rayOrigin    = worldPos;
  prd.rayDirection = mc.sample_in;

  if(mask == 0 || mask == 8)
  {
    prd.done    = true;
    prd.photons = true;
  }
}