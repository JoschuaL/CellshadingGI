/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include "nvvk/allocator_vk.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

// #VKRay
#include "Lights.h"
#include "nvvk/raytraceKHR_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//

enum LightType{
  Point = 0,
  Infinite,
  Spot,
  Area
};


class HelloVulkan : public nvvk::AppBase
{
public:
  void setup(const vk::Instance&       instance,
             const vk::Device&         device,
             const vk::PhysicalDevice& physicalDevice,
             uint32_t                  queueFamily) override;
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
  void updateDescriptorSet();
  void createUniformBuffer();
  void createSceneDescriptionBuffer();
  void createTextureImages(const vk::CommandBuffer&        cmdBuf,
                           const std::vector<std::string>& textures);
  void updateUniformBuffer();
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();
  void rasterize(const vk::CommandBuffer& cmdBuff);

  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
  };

  // Instance of the OBJ
  struct ObjInstance
  {
    uint32_t      objIndex{0};     // Reference to the `m_objModel`
    uint32_t      txtOffset{0};    // Offset in `m_textures`
    nvmath::mat4f transform{1};    // Position of the instance
    nvmath::mat4f transformIT{1};  // Inverse transpose
  };

  // Information pushed at each draw call
  struct ObjPushConstant
  {
    nvmath::vec3f lightPosition{0.f, 2.f, 0.f};
    int           instanceId{0};  // To retrieve the transformation matrix
    nvmath::vec4f lightColor{3.f,3.f,3.f, 1.f};
    LightType     lightType{Point};  // 0: point, 1: infinite
  };
  ObjPushConstant m_pushConstant;

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;
  std::vector<ObjInstance> m_objInstance;

  // Graphic pipeline
  vk::PipelineLayout          m_pipelineLayout;
  vk::Pipeline                m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  vk::DescriptorPool          m_descPool;
  vk::DescriptorSetLayout     m_descSetLayout;
  vk::DescriptorSet           m_descSet;

  nvvk::Buffer               m_cameraMat;  // Device-Host of the camera matrices
  nvvk::Buffer               m_sceneDesc;  // Device buffer of the OBJ instances
  std::vector<nvvk::Texture> m_textures;   // vector of all textures of the scene


  nvvk::AllocatorDedicated m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;  // Utility to name objects

  // #Post
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(vk::CommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  vk::DescriptorPool          m_postDescPool;
  vk::DescriptorSetLayout     m_postDescSetLayout;
  vk::DescriptorSet           m_postDescSet;
  vk::Pipeline                m_postPipeline;
  vk::PipelineLayout          m_postPipelineLayout;
  vk::RenderPass              m_offscreenRenderPass;
  vk::Framebuffer             m_offscreenFramebuffer;

  nvvk::Image                 m_offscreenColorImage;
  nvvk::Texture               m_offscreenColor;
  vk::Format                  m_offscreenColorFormat{vk::Format::eR32G32B32A32Sfloat};

  nvvk::Image                 m_offscreenDepthImage;
  nvvk::Texture               m_offscreenDepth;
  vk::Format                  m_offscreenDepthFormat{vk::Format::eD32Sfloat};

  nvvk::Image                 m_offscreenDepthImageRT;
  nvvk::Texture               m_offscreenDepthRT;
  vk::Format                  m_offscreenDepthFormatRT{vk::Format::eR32G32B32A32Sfloat};

  nvvk::Image                 m_offscreenNormalImage;
  nvvk::Texture m_offscreenNormal;
  vk::Format m_offscreenNormalFormat{vk::Format::eR32G32B32A32Sfloat};

nvvk::Image   m_offscreenIdImage;
nvvk::Texture m_offscreenId;
vk::Format    m_offscreenIdFormat{vk::Format::eR32G32B32A32Sfloat};

	
nvvk::Image   m_saveImage;
nvvk::Texture m_save;
vk::Format    m_saveFormat{vk::Format::eR32G32B32A32Sfloat};

	uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties);
	

  // #VKRay
  void                             initRayTracing();
  nvvk::RaytracingBuilderKHR::Blas objectToVkGeometryKHR(const ObjModel& model);
  void                             createBottomLevelAS();
  void                             createTopLevelAS();
  void                             createRtDescriptorSet();
  void                             updateRtDescriptorSet();
  void                             createRtPipeline();
  void                             createRtShaderBindingTable();
  void raytrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);
  void resetFrame();
  void updateFrame();
  void addPointLight(PointLight p);
  void saveImage();
  void postModelSetup();


  vk::PhysicalDeviceRayTracingPropertiesKHR           m_rtProperties;
  nvvk::RaytracingBuilderKHR                          m_rtBuilder;
  nvvk::DescriptorSetBindings                         m_rtDescSetLayoutBind;
  vk::DescriptorPool                                  m_rtDescPool;
  vk::DescriptorSetLayout                             m_rtDescSetLayout;
  vk::DescriptorSet                                   m_rtDescSet;
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  vk::PipelineLayout                                  m_rtPipelineLayout;
  vk::Pipeline                                        m_rtPipeline;
  nvvk::Buffer                                        m_rtSBTBuffer;
  float m_fuzzyAngle = 0.1f;
  std::vector<AreaLight>     m_AreaLightsPerObject = {};
  int m_numAreaSamples = 1;
  int m_numSamples = 1;
  int m_FrameCount = 0;
  float m_IOR = 0.0f;
	LightType m_LightType = LightType::Infinite;
  nvmath::vec3f                                       m_LightPosition       = {0, 0, 1};

  nvvk::Image saveImageData;

	std::vector<PointLight> m_PointLights = {};
  nvvk::Buffer            m_pointLightBuffer;
		nvvk::Buffer m_areaLightBuffer;
	int m_modelNumber = 0;

  struct RtPushConstant
  {
    nvmath::vec4f clearColor;
    nvmath::vec4f lightColor;
    nvmath::vec3f lightPosition = {0,0,1};
    LightType     lightType = LightType::Infinite;
    int           numObjs;
    int numAreaSamples = 1;
    int frame = 0;
    int numSamples = 1;
    float fuzzyAngle = 0.1f;
    float ior = 0.0f;
    int           numPointLights;
  	int numAreaLights;
    float         celramp = 0.9;
    int           celsteps = 10;
    bool          celatten = false;
  	int numids;
  	float r = 0.005;
  	float cut = 0.7;
  
  	
  } m_rtPushConstants;

	struct PostPushConstant
	{
    float aspectRatio;
    int   width;
    int   height;
    float threshold = 1.5;
    int useSobel = 1;
    int blurRange = 1;
	} m_postPushConstants;
};
