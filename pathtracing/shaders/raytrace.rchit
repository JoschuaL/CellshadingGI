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
  int   numLights;
  int maxBounces;
  float maxRussian;
}
pushC;

layout(location = 0) callableDataEXT materialCall mc;
layout(location = 2) callableDataEXT emissionCall ec;
layout(location = 3) callableDataEXT directSampleCall dsc;


void main()
{

    uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT
                     | gl_RayFlagsSkipClosestHitShaderEXT;
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

 

  //Material of the object
  int               matIdx = matIndex[nonuniformEXT(objId)].i[gl_PrimitiveID];
  WaveFrontMaterial mat    = materials[nonuniformEXT(objId)].m[matIdx];


  mc.objId  = objId;
  mc.pId    = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  mc.normal        = gn;
  mc.outDir        = gl_WorldRayDirectionEXT;


 


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
    
    int lightType = 1;
    int call = 4 + lightType;
    ec.dir = -gl_WorldRayDirectionEXT;
    ec.li = AreaLight(vec4(mat.emission,1), vec4(v0.pos,0),vec4(v1.pos,0),vec4(v2.pos,0));
    executeCallableEXT(call, 2);
    if(prd.specular)
    {
      prd.color +=  ec.intensity * prd.weight;
      //prd.color = vec3(1,1,1);
    }
    else
    {
      vec3  r   = gl_WorldRayOriginEXT - worldPos;
      float pne = (ec.pdf_area * dot(r, r)) / (dot(-gl_WorldRayDirectionEXT, gn) * pushC.numLights);
      float p_bsdf     = prd.last_bsdf_pdf;
      float mis_weight = p_bsdf / (p_bsdf + pne);
      prd.color     +=  ec.intensity * mis_weight * prd.weight;
      
    }
    prd.done = true;
    return;
  }

  

  prd.specular = (matProb & 12) != 0;
  bool diffuse = (matProb & 3) != 0;




  float ww = 1.0;
  int mask = matProb;
  if(prd.specular && diffuse){
    float r = rnd(prd.seed);
    mask = r > 0.5 ? (mask & 12) : (mask & 3);
    ww *= 2;
  }

  switch(mask){
    case 1:{mask = 0; break;}
    case 2:{mask = 1; break;}
    case 3:{mask = rnd(prd.seed) > 0.5 ? 1 : 0;ww*=2; break;}
    case 4:{mask = 2; break;}
    case 8:{mask = 3; break;}
    case 12:{mask = rnd(prd.seed) > 0.5 ? 3 : 2;ww*=2; break;}
  }

  mc.objId = objId;
  mc.pId = gl_PrimitiveID;
  mc.instID = gl_InstanceID;
  mc.texCoord =
      v0.texCoord * barycentrics.x + v1.texCoord * barycentrics.y + v2.texCoord * barycentrics.z;
  mc.normal = gn;

  mc.position   = worldPos;
  mc.seed       = prd.seed;
  mc.fuzzyAngle = pushC.ior;
  mc.inDir = vec3(1,0,0);

  executeCallableEXT(0,0);

  prd.seed         = mc.seed;

   
   
  
  if(diffuse)
  {
    float mis_weight;
    int cl = min(int(rnd(prd.seed) * pushC.numLights), pushC.numLights - 1);
    AreaLight li = lights.l[cl];
    vec3 e1        = li.v1.xyz - li.v0.xyz;
    vec3 e2        = li.v2.xyz - li.v0.xyz;
    vec3 ln        = normalize(cross(e1, e2));
    float u = rnd(prd.seed);
    float v = rnd(prd.seed);
    vec3 lpos = u + v < 1.0 ? li.v0.xyz + (u * e1) + (v * e2) :
                                  li.v0.xyz + ((1.0 - u) * e1) + ((1.0 - v) * e2);
    vec3 pos = offset_ray(offset_ray(lpos, ln), ln);
    
    dsc.seed = prd.seed;
    dsc.li = li;
    dsc.from = worldPos;
    dsc.pos = pos;
    int lightType = 1;
    int call = 6 + lightType;
    executeCallableEXT(call, 3);
    prd.seed = dsc.seed;

    vec3 dir = worldPos - dsc.pos;
    float d2 = dot(dir,dir);
    float dist = sqrt(d2);
    vec3 rayDir = dir / dist;
    float cos_hit = dot(rayDir, gn);
    if(cos_hit < 0){
      isShadowed = true;


      traceRayEXT(topLevelAS,  // acceleration structure
                  flags,       // rayFlags
                  0xFF,        // cullMask
                  0,           // sbtRecordOffset
                  0,           // sbtRecordStride
                  1,           // missIndex
                  dsc.pos,    // ray origin
                  0.0,         // ray min range
                  rayDir,      // ray direction
                  dist,        // ray max range
                  1            // payload (location = 1)
      );

      if(!isShadowed)
      {
        mc.inDir       = -rayDir;
        mc.eval_color = vec3(0, 0, 0);
     


        float pne = ((dsc.pdf_area * d2) / (dsc.cos_v * pushC.numLights));
        float ww = 1.0;
        int mask = matProb;
        if((mask & 3) == 3){
            mask = rnd(prd.seed) > 0.5 ? 2 : 1;
            ww *= 2;
        }
        mask -= 1;
        executeCallableEXT(mask,0);

        float p_bsdf = mc.pdf_pdf;
        mis_weight   = pne / (pne + p_bsdf);
        prd.color +=  ww * 
                      li.color.xyz * 
                      mis_weight * 
                      (
                        (
                            prd.weight * 
                            mc.eval_color * 
                            dsc.cos_v * 
                            pushC.numLights * 
                            abs(cos_hit)
                        )
                        / 
                        (
                            d2 * 
                            dsc.pdf_area
                        )
                      );
       
        
      }
        
        
    }
  }



  float p = russian_roulette(prd.weight, pushC.maxRussian);
  if(rnd(prd.seed) > p)
  {
    prd.done = true;
    return;
  }

  


  



  

  prd.last_bsdf_pdf = mc.sample_pdf;
  if(mc.sample_color == vec3(0, 0, 0) || mc.sample_pdf <= 0.0)
  {
    prd.done = true;
    return;
  }

 

  prd.weight *= (mc.sample_color * abs(dot(mc.sample_in, gn))) / (mc.sample_pdf * p);

  
  
  prd.rayOrigin    = worldPos;
  prd.rayDirection = mc.sample_in;
 



  


 


  
  
}
