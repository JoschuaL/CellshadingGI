const float M_PI = 3.1415926535897932384626433832795;

struct hitPayload
{
  vec3  hitValue;
  vec3  attenuation;
  vec3  rayOrigin;
  vec3  rayDirection;
  uint  seed;
  bool  done;
  float depth;
  vec3  normal;
  int   object;
  int   celid;
};

struct celValues
{
  float cel1;
  float cel2;
  float cel3;
  float  celoutR;
};

struct celPayload
{
  float depth;
  vec3  normal;
  int   object;
  uint  celid;
};

struct AreaLight
{
  vec4 color;
  vec4 v0;
  vec4 v1;
  vec4 v2;
};

struct PointLight
{
  vec4 color;
  vec4 pos;
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
  vec3  position;
  vec3  origin;
  vec3  inR;
  vec3  outR;
  vec3  emission;
  float fuzzyAngle;
  float celfaccounter;
  int   celcounter;
  vec3  celradiance;
  int lighttype;
  float celtotal;
};

const float origin      = 1.0 / 32.0;
const float float_scale = 1.0 / 65536.0;
const float int_scale   = 256.0;

vec3 offset_ray(vec3 p, vec3 n)
{
  ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

  vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                  intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                  intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
              abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
              abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}
