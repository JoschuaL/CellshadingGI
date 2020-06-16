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
layout(binding = 7, set = 1) buffer AreaLightsBuffer { AreaLight l[]; } lights[];

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
}
pushC;

layout(location = 0) callableDataEXT materialCall mc;


void main()
{
  // Object of this instance
  uint objId = scnDesc.i[gl_InstanceID].objId;
  //init_rnd(prd.seed);

  // Indices of the triangle
  ivec3 ind = ivec3(indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 0],   //
                    indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 1],   //
                    indices[nonuniformEXT(objId)].i[3 * gl_PrimitiveID + 2]);  //
  // Vertex of the triangle
  Vertex v0 = vertices[nonuniformEXT(objId)].v[ind.x];
  Vertex v1 = vertices[nonuniformEXT(objId)].v[ind.y];
  Vertex v2 = vertices[nonuniformEXT(objId)].v[ind.z];

  int matProb = v0.mat;

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Computing the normal at hit position
  vec3 normal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  // Transforming the normal to world space
  normal        = normalize(vec3(scnDesc.i[gl_InstanceID].transfoIT * vec4(normal, 0.0)));
  int lightType = floatBitsToInt(pushC.lightPosition.w);

  // Computing the coordinates of the hit position
  vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  // Transforming the position to world space
  worldPos = vec3(scnDesc.i[gl_InstanceID].transfo * vec4(worldPos, 1.0));

  float inanglecos = dot(gl_WorldRayDirectionEXT, normal);
  vec3  gn;
  if(inanglecos < 0)
  {
    gn = normal;
  }
  else
  {
    gn = -normal;
  }
  worldPos = offset_ray(worldPos, gn);

  // Vector toward the light
  /*vec3  L;
  vec4  lightColor    = pushC.lightColor;
  float lightDistance = 100000.0;
  // Point light
  if(lightType == 0)
  {
    vec3 lDir      = pushC.lightPosition.xyz - worldPos;
    lightDistance  = length(lDir);
    lightColor.xyz = pushC.lightColor.xyz / (lightDistance * lightDistance);
    L              = normalize(lDir);
  }
  else  // Directional light
  {
    L = normalize(pushC.lightPosition.xyz - vec3(0));
  }

  // Material of the object
  int               matIdx = matIndex[nonuniformEXT(objId)].i[gl_PrimitiveID];
  WaveFrontMaterial mat    = materials[nonuniformEXT(objId)].m[matIdx];


  // Diffuse
  vec3 diffuse = computeDiffuse(mat, L, normal);
  if(mat.textureId >= 0)
  {
    uint txtId = mat.textureId + scnDesc.i[gl_InstanceID].txtOffset;
    vec2 texCoord =
        v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
    diffuse *= texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
  }

  vec3  specular    = vec3(0);
  float attenuation = 1;

  // Tracing shadow ray only if the light is visible from the surface
  if(dot(normal, L) > 0)
  {
    float tMin   = 0.01;
    float tMax   = lightDistance;
    vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3  rayDir = L;
    uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT
                 | gl_RayFlagsSkipClosestHitShaderEXT;
    isShadowed = true;
    traceRayEXT(topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                origin,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
    );
    int cell_levels = 10;
    if(isShadowed)
    {
      attenuation = 0.3;
    }
    else
    {
      specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, normal);
    }
  }*/


  mc.objId  = objId;
  mc.pId    = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  ;
  mc.normal        = gn;
  mc.outDir        = gl_WorldRayDirectionEXT;
  mc.reflectance   = vec3(0, 0, 0);
  mc.emission      = vec3(0, 0, 0);
  vec3 lightsColor = vec3(0.0, 0.0, 0.0);

  for(int i = 0; i < pushC.numObjs; i++)
  {
    int j = 0;
    while(true)
    {
      AreaLight li = lights[0].l[j];
      j++;
      if(li.color.x + li.color.y + li.color.z <= 0.0)
      {
        if(floatBitsToInt(li.v2.w) == 1)
        {
          break;
        }
        else
        {
          continue;
        }
      }
      vec3 d1        = li.v1.xyz - li.v0.xyz;
      vec3 d2        = li.v2.xyz - li.v0.xyz;
      vec3 ln        = normalize(cross(d1, d2));
      vec3 tempColor = vec3(0, 0, 0);
      for(int k = 0; k < pushC.numAreaSamples; k++)
      {
        float u = rnd(prd.seed);
        float v = rnd(prd.seed);

        vec3 lpos = u + v < 1.0 ? li.v0.xyz + (u * d1) + (v * d2) :
                                  li.v0.xyz + ((1.0 - u) * d1) + ((1.0 - v) * d2);


        vec3 pos = offset_ray(offset_ray(lpos, ln), ln);

        vec3  ldir = pos - worldPos;
        float dist = length(ldir);
        vec3  L    = normalize(ldir);

        float outanglecos = dot(-L, gn);

        if(inanglecos < 0 && outanglecos > 0 || inanglecos > 0 && outanglecos < 0)
        {
          if(floatBitsToInt(li.v2.w) == 1)
          {
            break;
          }
          else
          {
            continue;
          }
        }


        float tMin   = 0.000;
        float tMax   = dist;
        vec3  rayDir = L;
        uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT
                     | gl_RayFlagsSkipClosestHitShaderEXT;
        isShadowed = true;


        traceRayEXT(topLevelAS,  // acceleration structure
                    flags,       // rayFlags
                    0xFF,        // cullMask
                    0,           // sbtRecordOffset
                    0,           // sbtRecordStride
                    1,           // missIndex
                    worldPos,    // ray origin
                    tMin,        // ray min range
                    rayDir,      // ray direction
                    tMax,        // ray max range
                    1            // payload (location = 1)
        );
        vec3 specular = vec3(0, 0, 0);


        float attenuation = isShadowed ? 0.0 : 0.1;
        mc.inDir          = L;
        mc.reflectance    = vec3(0, 0, 0);
        mc.emission       = vec3(0, 0, 0);
        if((matProb & 1) != 0)
        {
          executeCallableEXT(0, 0);
        }
        if((matProb & 2) != 0)
        {
          executeCallableEXT(1, 0);
        }
        tempColor += mc.emission + mc.reflectance * (li.color.xyz * attenuation) / (dist * dist);
      }


      lightsColor += tempColor / pushC.numAreaSamples;

      if(floatBitsToInt(li.v2.w) == 1)
      {
        break;
      }
    }
  }

  if((matProb & 8) != 0)
  {
    mc.position   = worldPos;
    mc.seed       = prd.seed;
    mc.fuzzyAngle = pushC.ior;
    executeCallableEXT(3, 0);
    prd.seed         = mc.seed;
    prd.attenuation  = mc.reflectance;
    prd.rayOrigin    = worldPos;
    prd.rayDirection = mc.emission;
    prd.done         = false;
  } else if((matProb & 4) != 0)
  {
    mc.position   = worldPos;
    mc.fuzzyAngle = pushC.fuzzyAngle;
    mc.seed       = prd.seed;
    executeCallableEXT(2, 0);
    prd.seed         = mc.seed;
    prd.attenuation  = mc.reflectance;
    prd.rayOrigin    = worldPos;
    prd.rayDirection = mc.emission;
    prd.done         = false;
  }

  


  prd.hitValue = lightsColor;
}