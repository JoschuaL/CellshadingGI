const float M_PI = 3.1415926535897932384626433832795;

struct hitPayload
{
  vec3  color;
  vec3  weight;
  vec3  rayOrigin;
  vec3  rayDirection;
  uint  seed;
  bool  done;
  float last_bsdf_pdf;
  bool  specular;
  bool  first;
  vec3  gnrm;
  vec3  snrm;
  bool  photons;
  int   material;
};

struct photonPayload
{
  vec3 color;
  vec3 rayOrigin;
  vec3 rayDirection;
  uint seed;
  bool done;
  vec3 gnrm;
  vec3 snrm;
  bool emplace;
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
  float fuzzyAngle;
  float ior;
  vec3  eval_color;
  vec3  sample_color;
  float sample_pdf;
  vec3  sample_in;
  float pdf_pdf;
  bool  entering;
  vec3  celdir;
};


struct emissionCall
{
  vec3      dir;
  vec3      intensity;
  float     pdf_area;
  float     pdf_dir;
  AreaLight li;
};

struct directSampleCall
{
  vec3       pos;
  vec3       intensity;
  float      pdf_area;
  float      pdf_dir;
  float      cos_v;
  AreaLight  li;
  PointLight p;
  uint       seed;
  vec3       from;
};

struct Photon
{
  vec4 pos;
  vec4 gnrm;
  vec4 snrm;
  vec4 inDir;
  vec4 color;
};

struct HitInfo
{
  vec4 pos;
  vec4 gnrm;
  vec4 snrm;
  vec4 inDir;
  vec4 outDir;
  vec4 color;
  vec4 weight;
  int  material;
};

struct LocalCoords
{
  vec3 n;   ///< Normal
  vec3 t;   ///< Tangent
  vec3 bt;  ///< Bitangent
};


struct celinfo
{
  vec4 max;
  vec4 min;
  vec4 avg;
};
const float origin      = 1.0 / 32.0;
const float float_scale = 1.0 / 65536.0;
const float int_scale   = 256.0;

vec3 offset_ray(vec3 p, vec3 n)
{
  const ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

  const vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                        intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                        intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

  return vec3(abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
              abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
              abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z);
}

void make_direct_sample(vec3                   pos,
                        vec3                   intensity,
                        float                  pdf_area,
                        float                  pdf_dir,
                        float                  cos_v,
                        inout directSampleCall dsc)
{
  if(pdf_area > 0 && pdf_dir > 0 && cos_v > 0)
  {
    dsc.pos       = pos;
    dsc.intensity = intensity;
    dsc.pdf_area  = pdf_area;
    dsc.pdf_dir   = pdf_dir;
    dsc.cos_v     = cos_v;
  }
  else
  {
    dsc.pos       = pos;
    dsc.intensity = vec3(0, 0, 0);
    dsc.pdf_area  = 1.0;
    dsc.pdf_dir   = 1.0;
    dsc.cos_v     = 1.0;
  }
}

void make_sample(vec3 dir, float pdf, vec3 color, vec3 n, inout materialCall mc)
{
  float sign = dot(dir, n);
  if(pdf > 0 && sign > 0)
  {
    mc.sample_in    = dir;
    mc.sample_pdf   = pdf;
    mc.sample_color = color;
  }
  else
  {
    mc.sample_in    = dir;
    mc.sample_pdf   = 1.0;
    mc.sample_color = vec3(0, 0, 0);
  }
}


const float PiOver4 = 0.78539816339744830961;
const float PiOver2 = 1.57079632679489661923;


void sample_cosine_hemisphere(LocalCoords coords, float u, float v, inout materialCall mc)
{
  // TODO: Sample a direction on the hemisphere using a pdf proportional to
  // cos(theta). The hemisphere is defined by the coordinate system "coords".
  // "u" and "v" are random numbers between [0, 1].

  /*
  const vec2 s = vec2(u, v) * 2 - vec2(1, 1);
  vec2       d;
  float      r, theta;
  if(s == vec2(0, 0))
  {
    d = s;
  }
  else
  {
    if(abs(s.x) > abs(s.y))
    {
      r     = s.x;
      theta = PiOver4 * (s.y / s.x);
    }
    else
    {
      r     = s.y;
      theta = PiOver2 - PiOver4 * (s.x / s.y);
    }
    d = s == vec2(0, 0) ? s : r * vec2(cos(theta), sin(theta));
  }
  const float z = sqrt(max(0, 1 - d.x * d.x - d.y * d.y));
  mc.sample_in  = d.x * coords.t + d.y * coords.bt + z * coords.n;
  mc.sample_pdf = abs(z) / M_PI;*/

  float phi       = 2.0 * M_PI * u;
  float cos_theta = sqrt(v);
  float theta     = acos(cos_theta);

  vec3 du       = normalize(cos(phi) * coords.t + sin(phi) * coords.bt);
  vec3 dir      = normalize(cos_theta * coords.n + sin(theta) * du);
  mc.sample_in  = dir;
  mc.sample_pdf = cos_theta / M_PI;
}

struct DirSample
{
  vec3  dir;
  float pdf;
};

DirSample sample_cosine_hemisphere_direct(LocalCoords coords, float u, float v)
{


  float phi       = 2.0 * M_PI * u;
  float cos_theta = sqrt(v);
  float theta     = acos(cos_theta);

  vec3 du = normalize(cos(phi) * coords.t + sin(phi) * coords.bt);
  return DirSample(normalize(cos_theta * coords.n + sin(theta) * du), cos_theta / M_PI);
}

float cosine_power_hemisphere_pdf(float c, float k)
{
  return pow(c, k) * ((k + 1.0) / (2.0 * M_PI));
}


void sample_cosine_power_hemisphere(LocalCoords        coords,
                                    float              k,
                                    float              u,
                                    float              v,
                                    inout materialCall mc)
{
  float phi       = 2.0 * M_PI * u;
  float cos_theta = pow(v, 1.0 / (k + 1.0f));
  float theta     = acos(cos_theta);
  float p         = cosine_power_hemisphere_pdf(cos_theta, k);
  vec3  vec       = normalize(cos(phi) * coords.t + sin(phi) * coords.bt);
  vec3  dir       = normalize(cos_theta * coords.n + sin(theta) * vec);
  mc.sample_in    = dir;
  mc.sample_pdf   = p;
}

LocalCoords gen_local_coords(vec3 n)
{
  // From "Building an Orthonormal Basis, Revisited", Duff et al.
  const float sign = n.z < 0.0 ? -1.0 : 1.0;
  const float a    = -1.0f / (sign + n.z);
  const float b    = n.x * n.y * a;
  const vec3  t    = normalize(vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x));
  const vec3  bt   = normalize(vec3(b, sign + n.y * n.y * a, -n.y));
  return LocalCoords(n, t, bt);
}