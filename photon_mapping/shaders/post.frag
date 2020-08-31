#version 450
#extension GL_KHR_vulkan_glsl : enable
layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;
layout(set = 0, binding = 1) uniform sampler2D normalTxt;
layout(set = 0, binding = 2) uniform sampler2D depthTxt;
layout(set = 0, binding = 3) uniform sampler2D idTxt;
layout(binding = 4, set = 0, rgba32f) uniform image2D save;

layout(push_constant) uniform shaderInformation
{
  float aspectRatio;
  int   width;
  int   height;
  float threshold;
  int   useSobel;
  int   blurRange;
}
pushc;

const float gaussian_k[25] = {0.024, 0.034, 0.038, 0.034, 0.024, 0.034, 0.049, 0.055, 0.049,
                              0.034, 0.038, 0.055, 0.063, 0.055, 0.038, 0.034, 0.049, 0.055,
                              0.049, 0.034, 0.024, 0.034, 0.038, 0.034, 0.024};


const float sobel_x[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

const float sobel_y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

const float just_scalingx[9] = {0, 0, 0, -1, 0, 1, 0, 0, 0};

const float just_scalingy[9] = {0, -1, 0, 0, 0, 0, 0, 1, 0};


vec3 blury(sampler2D txt, vec2 uv)
{

  int w = pushc.width;
  int h = pushc.height;
  //auto img = static_cast<float *>(malloc(sizeof(float) * w * h));


  /**
         * Blur image with the given Gaussian kernel (see gaussian_kernel.h).
         * Write the blurred image to the file out_blur.pgm
         */

  int w_k = min(w - (1 - (w % 2)), 5);
  int h_k = min(h - (1 - (h % 2)), 5);

  vec3 blur = vec3(0);

  //auto result = static_cast<float *>(malloc(sizeof(float) * w * h));
  {

    int a = ((w_k + 1) / 2) - 1;
    int b = ((h_k + 1) / 2) - 1;


    for(int j = 0; j < h_k; j++)
    {
      for(int i = 0; i < w_k; i++)
      {
        float kernelpix = gaussian_k[(j * w_k) + i];
        blur += kernelpix * texture(txt, vec2(uv.x + i - a, uv.y + j - b)).rgb;
      }
    }
  }

  return blur;
}


float sobel(sampler2D Txt, vec2 uv, float t)
{
  float offsetx    = 1.0 / pushc.width;
  float offsety    = 1.0 / pushc.height;
  vec3  blurred[9] = vec3[9](
      texture(Txt, uv + vec2(-offsetx, -offsety)).rgb, texture(Txt, uv + vec2(0, -offsety)).rgb,
      texture(Txt, uv + vec2(offsetx, -offsety)).rgb, texture(Txt, uv + vec2(-offsetx, 0)).rgb,
      texture(Txt, uv).rgb, texture(Txt, uv + vec2(offsetx, 0)).rgb,
      texture(Txt, uv + vec2(-offsetx, offsety)).rgb, texture(Txt, uv + vec2(0, offsety)).rgb,
      texture(Txt, uv + vec2(offsetx, offsety)).rgb

      /*vec3 blurred[9] = vec3[9](
    blury(Txt, uv + vec2(-1,-1)).rgb,
    blury(Txt, uv + vec2(0,-1)).rgb,
    blury(Txt, uv + vec2(1,-1)).rgb,
    blury(Txt, uv + vec2(-1,0)).rgb,
    blury(Txt, uv).rgb,
    blury(Txt, uv + vec2(1,0)).rgb,
    blury(Txt, uv + vec2(-1,1)).rgb,
    blury(Txt, uv + vec2(0,1)).rgb,
    blury(Txt, uv + vec2(1,1)).rgb*/
  );

  vec3 devx = vec3(0);
  vec3 devy = vec3(0);

  {
    int w_k = 3;
    int h_k = 3;
    int a   = ((w_k + 1) / 2) - 1;
    int b   = ((h_k + 1) / 2) - 1;


    for(int j = 0; j < h_k; j++)
    {
      for(int i = 0; i < w_k; i++)
      {
        float kernelpixx = sobel_x[(j * w_k) + i];
        float kernelpixy = sobel_y[(j * w_k) + i];
        devx += kernelpixx * blurred[(j * w_k) + i];
        devy += kernelpixy * blurred[(j * w_k) + i];
      }
    }
  }


  vec3  mag = sqrt(devx * devx + devy * devy);
  float d   = texture(depthTxt, uv).r;
  mag       = step(t, mag * d * d);

  return (1 - mag.r) * (1 - mag.g) * (1 - mag.b);
}


void main()
{
  vec3  acc   = vec3(1);
  vec2  uv    = outUV;
  float gamma = 1. / 2.2;
  //fragColor   = pow(vec4(blury(uv), 1.f), vec4(gamma));


  float outlines = pushc.useSobel > 0 ?
                       sobel(idTxt, uv, 0.001) * sobel(normalTxt, uv, pushc.threshold)
                           * sobel(depthTxt, uv, pushc.threshold) :
                       1;
  vec4 c = pow(vec4(texture(noisyTxt, uv).rgb * outlines, 1.0), vec4(gamma));

  //imageStore(save, ivec2(uv * vec2(pushc.width, pushc.height)), c);

  fragColor = imageLoad(save, ivec2(uv * vec2(pushc.width, pushc.height)));
}