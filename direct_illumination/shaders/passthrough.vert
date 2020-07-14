#version 450
layout (location = 0) out vec2 outUV;
layout(set = 0, binding = 0) uniform sampler2D noisyTxt;
layout(set = 0, binding = 1) uniform sampler2D normalTxt;
layout(set = 0, binding = 2) uniform sampler2D depthTxt;

layout(push_constant) uniform shaderInformation
{
  float aspectRatio;
  int width;
  int height;
}
pushc;

out gl_PerVertex
{
  vec4 gl_Position;
};

const float gaussian_k[25] = {
            0.024, 0.034, 0.038, 0.034, 0.024,
            0.034, 0.049, 0.055, 0.049, 0.034,
            0.038, 0.055, 0.063, 0.055, 0.038,
            0.034, 0.049, 0.055, 0.049, 0.034,
            0.024, 0.034, 0.038, 0.034, 0.024
    };


     const float sobel_x[9] = {
            1, 0, -1,
            2, 0, -2,
            1, 0, -1
    };

    const float sobel_y[9] = {
            1, 2, 1,
            0, 0, 0,
            -1, -2, -1
    };

    const float just_scalingx[9] = {
            0, 0, 0,
            -1, 0, 1,
            0, 0, 0
    };

    const float just_scalingy[9] = {
            0, -1, 0,
            0, 0, 0,
            0, 1, 0
    };




    


    


    vec3 blury(vec2 uv) {

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


       
        for(int j = 0; j < h_k; j++){
            for(int i = 0; i < w_k; i++){
                float kernelpix = gaussian_k[(j * w_k) + i];
                blur += kernelpix * texture(noisyTxt, vec2(uv.x + i - a, uv.y + j - b)).rgb;
            }
        }
       

        }

        return blur;
       

    }





void main()
{
  outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

  gl_Position = vec4(outUV * 2.0f - 1.0f, 1.0f, 1.0f);
}
