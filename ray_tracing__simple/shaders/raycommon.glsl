const float M_PI = 3.1415926535897932384626433832795;

struct hitPayload
{
  vec3 hitValue;
  vec3 attenuation;
  vec3 rayOrigin;
  vec3 rayDirection;
  uint seed;
  bool done;
};

struct AreaLight
{
  vec4 color;
  vec4 v0;
  vec4 v1;
  vec4 v2;
};

struct materialCall
{
  uint  objId;
  uint  seed;
  int   pId;
  int   instID;
  vec2  texCoord;
  vec3  normal;
  vec3  inDir;
  vec3  outDir;
  vec3 position;
  vec3  reflectance;
  vec3  emission;
  float fuzzyAngle;
};


vec3 offsetRay(in vec3 p, in vec3 n)
{
  const float intScale   = 256.0f;
  const float floatScale = 1.0f / 65536.0f;
  const float origin     = 1.0f / 32.0f;

  ivec3 of_i = ivec3(intScale * n.x, intScale * n.y, intScale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + floatScale * n.x : p_i.x,  //
              abs(p.y) < origin ? p.y + floatScale * n.y : p_i.y,  //
              abs(p.z) < origin ? p.z + floatScale * n.z : p_i.z);
}