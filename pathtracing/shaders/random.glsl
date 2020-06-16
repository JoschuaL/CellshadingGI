
const vec3 luminance = vec3(0.2126f, 0.7152f, 0.0722f);


uint tea(uint val0, uint val1)
{
  uint v0 = val0;
  uint v1 = val1;
  uint s0 = 0;

  for(uint n = 0; n < 16; n++)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
    v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
  }

  return v0;
}

// Generate random unsigned int in [0, 2^24)
uint lcg(inout uint prev)
{
  uint LCG_A = 1664525u;
  uint LCG_C = 1013904223u;
  prev       = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

uint lcg2(inout uint prev)
{
  prev = (prev * 8121 + 28411) % 134456;
  return prev;
}

// Generate random float in [0, 1)
float rnd(inout uint prev)
{
  return (float(lcg(prev)) / float(0x01000000));
}


vec2 rnd2(inout uint prev)
{
  return vec2(rnd(prev), rnd(prev));
}

float russian_roulette(vec3 c, float mx)
{
  return min(mx, dot(c, luminance) * 2.0);
}

