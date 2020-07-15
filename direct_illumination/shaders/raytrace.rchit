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
layout(binding = 8, set = 1) buffer PointLightsBuffer { PointLight l[]; } plights;

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

layout(location = 0) callableDataEXT materialCall mc;


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

  const int matProb = v0.mat;

  prd.depth = (matProb & 32) != 0 ? max(200 - gl_HitTEXT, 0) / 200 : 0;

  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Computing the normal at hit position
  vec3 snormal = v0.nrm * barycentrics.x + v1.nrm * barycentrics.y + v2.nrm * barycentrics.z;
  vec3 gnormal = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
  // Transforming the normal to world space
  snormal       = normalize(vec3(scnDesc.i[gl_InstanceID].transfoIT * vec4(snormal, 0.0)));
  gnormal       = normalize(vec3(scnDesc.i[gl_InstanceID].transfoIT * vec4(gnormal, 0.0)));
  const int lightType = floatBitsToInt(pushC.lightPosition.w);

  // Computing the coordinates of the hit position
  vec3 worldPos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
  // Transforming the position to world space
  worldPos = vec3(scnDesc.i[gl_InstanceID].transfo * vec4(worldPos, 1.0));

  const float inanglecos = dot(gl_WorldRayDirectionEXT, gnormal);
  const float ins = dot(gl_WorldRayDirectionEXT, snormal);

  gnormal *= -sign(inanglecos);
  snormal *= -sign(ins);

  prd.normal = (matProb & 32) != 0 ? (snormal + vec3(1)) / 2 : vec3(-1);
  prd.object = (matProb & 32) != 0 ? v0.id : 0;

  
  worldPos = offset_ray(worldPos, gnormal);

  mc.celcounter = 0;
  mc.objId  = objId;
  mc.pId    = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  ;
  mc.normal        = snormal;
  mc.outDir        = gl_WorldRayDirectionEXT;
  mc.origin = gl_WorldRayOriginEXT;
  mc.inR   = vec3(0, 0, 0);
  mc.outR = vec3(0,0,0);
  mc.emission      = vec3(0, 0, 0);
  mc.celradiance = vec3(0);
  mc.celfaccounter = 1;
  vec3 lightsColor = vec3(0.0, 0.0, 0.0);

  // Vector toward the light
  vec3  L;
  vec3  lightColor    = pushC.lightColor.xyz;
  float lightDistance = 100000.0;
  // Point light
  if(lightType == 0)
  {
    const vec3 lDir      = pushC.lightPosition.xyz - worldPos;
    lightDistance  = length(lDir);
    lightColor = pushC.lightColor.xyz / (lightDistance * lightDistance);
    L              = lDir / lightDistance;
  }
  else  // Directional light
  {
    L = normalize(pushC.lightPosition.xyz);
  }

  // Material of the object
  const int               matIdx = matIndex[nonuniformEXT(objId)].i[gl_PrimitiveID];
  const WaveFrontMaterial mat    = materials[nonuniformEXT(objId)].m[matIdx];


  // Diffuse
  
  
  vec3 lcolor = vec3(0);
   mc.inDir          = L;
        mc.inR    = lightColor.xyz;
        mc.emission       = vec3(0, 0, 0);
    int call = 0;
    float fac = 1.0;
  switch(matProb){
    case 1:{call = 0; break;}
    case 2:{call = 1; break;}
    case 3:{call = rnd(prd.seed) > 0.5 ? 1 : 0; fac *= 2; break;}
    case 32:{call = 4; break;}
  }
       
          executeCallableEXT(call, 0);
 
       
 
    vec3 emission = mc.emission;

  // Tracing shadow ray only if the light is visible from the surface
  if(dot(snormal, L) > 0)
  {
    const float tMin   = 0.00;
    const float tMax   = lightDistance;
    const vec3  rayDir = L;
    const uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT
                 | gl_RayFlagsSkipClosestHitShaderEXT;
    isShadowed = true;
    traceRayEXT(topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                worldPos,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
    );
  
        

           lcolor = mc.outR * fac * float(!isShadowed);
  
    }
   
        
       
    


  for(int i = 0; i < pushC.numPointLights; i++){
   
   PointLight li = plights.l[i];
      
      if(li.color.x + li.color.y + li.color.z <= 0.0)
      {
       
          continue;
        
      }
     

       

        vec3 pos = li.pos.xyz;

        vec3  ldir = pos - worldPos;
        float dist = length(ldir);
        vec3  L    = ldir / dist;

        float outanglecos = dot(L, gnormal);

        if(outanglecos < 0)
        {
          continue;
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
        
        
        if(isShadowed){
            continue;
        }

        
        mc.inDir          = L;
        mc.inR    = li.color.xyz / (dist * dist);
       
       


         int call = 0;
         float fac = 1.0;
  switch(matProb){
    case 1:{call = 0; break;}
    case 2:{call = 1; break;}
    case 3:{executeCallableEXT(0, 0); call = 1;break;}
    case 32:{call = 4; break;}
  }
       
        executeCallableEXT(call, 0);
        
      


       

      


  }
  
  


  for(int i = 0; i < pushC.numObjs; i++)
  {
    int j = 0;
    while(true)
    {
      AreaLight li = lights[i].l[j];
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
      
        float u = rnd(prd.seed);
        float v = rnd(prd.seed);

        vec3 lpos = u + v < 1.0 ? li.v0.xyz + (u * d1) + (v * d2) :
                                  li.v0.xyz + ((1.0 - u) * d1) + ((1.0 - v) * d2);


        vec3 pos = offset_ray(offset_ray(lpos, ln), ln);

        vec3  ldir = pos - worldPos;
        float dist = length(ldir);
        vec3  L    = ldir / dist;

        float outanglecos = dot(L, gnormal);

        if(outanglecos < 0)
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
      if(isShadowed)
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


        
        mc.inDir          = L;
        mc.inR    = li.color.xyz / (dist * dist);
        mc.emission       = vec3(0, 0, 0);
        float fac = 1.0;
         int call = 0;

  switch(matProb){
    case 1:{call = 0; break;}
    case 2:{call = 1; break;}
    case 3:{executeCallableEXT(0,0); call = 1; break;}
    case 32:{call = 4; break;}
  }
       
          executeCallableEXT(call, 0);
        

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
    prd.attenuation  = mc.inR;
    prd.rayOrigin    = worldPos;
    prd.rayDirection = mc.emission;
    prd.done         = false;
  }
  else if((matProb & 4) != 0)
  {
    mc.position   = worldPos;
    mc.fuzzyAngle = pushC.fuzzyAngle;
    mc.seed       = prd.seed;
    executeCallableEXT(2, 0);
    prd.seed         = mc.seed;
    prd.attenuation  = mc.inR;
    prd.rayOrigin    = worldPos;
    prd.rayDirection = mc.emission;
    prd.done         = false;
  }


  prd.hitValue = lcolor + mc.outR + emission;
}
