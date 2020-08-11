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

#include <sstream>
#include <vulkan/vulkan.hpp>

extern std::vector<std::string> defaultSearchPaths;

#define STB_IMAGE_IMPLEMENTATION
#include "fileformats/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "fileformats/stb_image_write.h"
#include "obj_loader.h"

#include "VulkanInitializers.hpp"
#include "VulkanTools.hpp"
#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"


// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const vk::Instance&       instance,
                        const vk::Device&         device,
                        const vk::PhysicalDevice& physicalDevice,
                        uint32_t                  queueFamily)
{
  AppBase::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(device, physicalDevice);
  m_debug.setup(m_device);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer()
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);

  CameraMatrices ubo = {};
  ubo.view           = CameraManip.getMatrix();
  ubo.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // ubo.proj[1][1] *= -1;  // Inverting Y for Vulkan
  ubo.viewInverse = nvmath::invert(ubo.view);
  // #VKRay
  ubo.projInverse = nvmath::invert(ubo.proj);

  void* data = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(ubo));
  memcpy(data, &ubo, sizeof(ubo));
  m_device.unmapMemory(m_cameraMat.allocation);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());
  uint32_t nbObj = static_cast<uint32_t>(m_objModel.size());

  // Camera matrices (binding = 0)
  m_descSetLayoutBind.addBinding(
      vkDS(0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenKHR));
  // Materials (binding = 1)
  m_descSetLayoutBind.addBinding(
      vkDS(1, vkDT::eStorageBuffer, nbObj,
           vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eCallableKHR));
  // Scene description (binding = 2)
  m_descSetLayoutBind.addBinding(  //
      vkDS(2, vkDT::eStorageBuffer, 1,
           vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eCallableKHR));
  // Textures (binding = 3)
  m_descSetLayoutBind.addBinding(vkDS(3, vkDT::eCombinedImageSampler, nbTxt,
                                      vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eCallableKHR));
  // Materials (binding = 4)
  m_descSetLayoutBind.addBinding(vkDS(4, vkDT::eStorageBuffer, nbObj,
                                      vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eCallableKHR));
  // Storing vertices (binding = 5)
  m_descSetLayoutBind.addBinding(  //
      vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR | vkSS::eRaygenKHR));
  // Storing indices (binding = 6)
  m_descSetLayoutBind.addBinding(  //
      vkDS(6, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitKHR | vkSS::eRaygenKHR));
  // AreaLights (binding = 7)
  m_descSetLayoutBind.addBinding(
      vkDS(7, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eRaygenKHR));
  // Point Lights (binding = 8)
  m_descSetLayoutBind.addBinding(
      vkDS(8, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eRaygenKHR));
  // Photon storage (binding = 9)
  m_descSetLayoutBind.addBinding(
      vkDS(9, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR));

  // HitPoint storage (binding = 10)
  m_descSetLayoutBind.addBinding(
      vkDS(10, vkDT::eStorageBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR)
  );


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 0, &dbiUnif));
  vk::DescriptorBufferInfo dbiSceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 2, &dbiSceneDesc));

  // All material buffers, 1 buffer per OBJ
  std::vector<vk::DescriptorBufferInfo> dbiMat;
  std::vector<vk::DescriptorBufferInfo> dbiMatIdx;
  std::vector<vk::DescriptorBufferInfo> dbiVert;
  std::vector<vk::DescriptorBufferInfo> dbiIdx;
  for(size_t i = 0; i < m_objModel.size(); ++i)
  {
    dbiMat.push_back({m_objModel[i].matColorBuffer.buffer, 0, VK_WHOLE_SIZE});
    dbiMatIdx.push_back({m_objModel[i].matIndexBuffer.buffer, 0, VK_WHOLE_SIZE});
    dbiVert.push_back({m_objModel[i].vertexBuffer.buffer, 0, VK_WHOLE_SIZE});
    dbiIdx.push_back({m_objModel[i].indexBuffer.buffer, 0, VK_WHOLE_SIZE});
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 1, dbiMat.data()));
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 4, dbiMatIdx.data()));
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 5, dbiVert.data()));
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 6, dbiIdx.data()));

  vk::DescriptorBufferInfo lights{m_areaLightsBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 7, &lights));

  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.push_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, 3, diit.data()));

  vk::DescriptorBufferInfo dbiPLights{m_pointLightBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 8, &dbiPLights));

  vk::DescriptorBufferInfo dbiPhotons{m_photonBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 9, &dbiPhotons));

  vk::DescriptorBufferInfo dbiHitPoints{m_hitBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, 10, &dbiHitPoints));


  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  using vkSS = vk::ShaderStageFlagBits;

  vk::PushConstantRange pushConstantRanges = {vkSS::eVertex | vkSS::eFragment, 0,
                                              sizeof(ObjPushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::DescriptorSetLayout      descSetLayout(m_descSetLayout);
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths), vkSS::eVertex);
  gpb.addShader(nvh::loadFile("shaders/frag_shader.frag.spv", true, paths), vkSS::eFragment);
  gpb.addBindingDescription({0, sizeof(VertexObj)});
  gpb.addAttributeDescriptions(std::vector<vk::VertexInputAttributeDescription>{
      {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, pos)},
      {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, nrm)},
      {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, color)},
      {3, 0, vk::Format::eR32G32Sfloat, offsetof(VertexObj, texCoord)}});

  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}


//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, nvmath::mat4f transform)
{
  using vkBU = vk::BufferUsageFlagBits;

  ObjLoader loader;
  loader.loadModel(filename, 1 << m_modelId++, 0);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);
  }


  ObjInstance instance;
  instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
  instance.transform   = transform;
  instance.transformIT = nvmath::transpose(nvmath::invert(transform));
  instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  for(int i = 0; i < loader.m_matIndx.size(); i++)
  {
    int                id  = loader.m_matIndx[i];
    const MaterialObj& mat = loader.m_materials[id];
    if(mat.emission.x > 0.f || mat.emission.y > 0.f || mat.emission.z > 0.f)
    {
      const AreaLight light = {mat.emission, loader.m_vertices[loader.m_indices[3 * i + 0]].pos,
                               loader.m_vertices[loader.m_indices[3 * i + 1]].pos,
                               loader.m_vertices[loader.m_indices[3 * i + 2]].pos};
      m_AreaLightsPerObject.push_back(light);
      std::cout << mat.emission.x << ',' << mat.emission.y << ',' << mat.emission.z << std::endl;
    }
  }


  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
  model.vertexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_vertices,
                           vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
  model.indexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_indices,
                           vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, vkBU::eStorageBuffer);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, vkBU::eStorageBuffer);


  // Creates all textures found
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(instance.objIndex);
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb).c_str()));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb).c_str()));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb).c_str()));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb).c_str()));


  m_objModel.emplace_back(model);
  m_objInstance.emplace_back(instance);
}

void HelloVulkan::postModelSetup()
{
  using vkBU = vk::BufferUsageFlagBits;
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();
  if(m_AreaLightsPerObject.size() > 0)
  {


    m_areaLightsBuffer = m_alloc.createBuffer(cmdBuf, m_AreaLightsPerObject, vkBU::eStorageBuffer);
  }
  else
  {

    m_areaLightsBuffer = m_alloc.createBuffer(cmdBuf, dummy, vkBU::eStorageBuffer);
  }

  std::vector<Photon> i(
      m_size.width * 32,
      {{0, 0, 0}, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  m_photonBuffer = m_alloc.createBuffer(cmdBuf, i, vkBU::eStorageBuffer | vkBU::eTransferSrc | vkBU::eTransferDst);

  std::vector<HitInfo> h(m_size.width * m_size.height, {
                                                           {0, 0, 0},
                                                           0,
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0, 0, 0, 0},
                                                           {0,0,0,0},
                                                           {0,0,0,0},
                                                       0});
  m_hitBuffer = m_alloc.createBuffer(cmdBuf, h, vkBU::eStorageBuffer | vkBU::eTransferSrc | vkBU::eTransferDst);


  cmdBufGet.submitAndWait(cmdBuf);

  m_alloc.finalizeAndReleaseStaging();

  m_debug.setObjectName(m_areaLightsBuffer.buffer, (std::string("Arealights").c_str()));
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer,
                                     vkMP::eHostVisible | vkMP::eHostCoherent);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createSceneDescriptionBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_sceneDesc = m_alloc.createBuffer(cmdBuf, m_objInstance, vkBU::eStorageBuffer);
  if(m_PointLights.size() > 0)
  {
    m_pointLightBuffer = m_alloc.createBuffer(cmdBuf, m_PointLights, vkBU::eStorageBuffer);
  }
  else
  {
    m_alloc.createBuffer(cmdBuf, dummy, vkBU::eStorageBuffer);
  }
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_sceneDesc.buffer, "sceneDesc");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer&        cmdBuf,
                                      const std::vector<std::string>& textures)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    vk::DeviceSize         bufferSize      = sizeof(color);
    auto                   imgSize         = vk::Extent2D(1, 1);
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the dummy texure
    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                        = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eShaderReadOnlyOptimal);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths);

      stbi_uc* pixels =
          stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      // Handle failure
      if(!pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        std::array<uint8_t, 4> color{255u, 0u, 255u, 255u};
        pixels = reinterpret_cast<stbi_uc*>(color.data());
      }

      vk::DeviceSize bufferSize = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto           imgSize    = vk::Extent2D(texWidth, texHeight);
      auto imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

      {
        nvvk::ImageDedicated image =
            m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        vk::ImageViewCreateInfo ivInfo =
            nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);
  m_alloc.destroy(m_sceneDesc);
  m_alloc.destroy(m_areaLightsBuffer);
  m_alloc.destroy(m_pointLightBuffer);
  m_alloc.destroy(m_photonBuffer);
  m_alloc.destroy(m_hitBuffer);


  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);
  m_device.destroy(m_postDescPool);
  m_device.destroy(m_postDescSetLayout);
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  m_alloc.destroy(m_offscreenDepthRT);
  m_alloc.destroy(m_offscreenNormal);
  m_alloc.destroy(m_offscreenId);
  m_alloc.destroy(m_save);
  m_device.destroy(m_offscreenRenderPass);
  m_device.destroy(m_offscreenFramebuffer);

  // #VKRay
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc.destroy(m_rtSBTBuffer);
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const vk::CommandBuffer& cmdBuf)
{
  using vkPBP = vk::PipelineBindPoint;
  using vkSS  = vk::ShaderStageFlagBits;
  vk::DeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  // Drawing all triangles
  cmdBuf.bindPipeline(vkPBP::eGraphics, m_graphicsPipeline);
  cmdBuf.bindDescriptorSets(vkPBP::eGraphics, m_pipelineLayout, 0, {m_descSet}, {});
  for(int i = 0; i < m_objInstance.size(); ++i)
  {
    auto& inst                = m_objInstance[i];
    auto& model               = m_objModel[inst.objIndex];
    m_pushConstant.instanceId = i;  // Telling which instance is drawn
    cmdBuf.pushConstants<ObjPushConstant>(m_pipelineLayout, vkSS::eVertex | vkSS::eFragment, 0,
                                          m_pushConstant);

    cmdBuf.bindVertexBuffers(0, {model.vertexBuffer.buffer}, {offset});
    cmdBuf.bindIndexBuffer(model.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    cmdBuf.drawIndexed(model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  m_alloc.destroy(m_offscreenNormal);
  m_alloc.destroy(m_offscreenId);
  m_alloc.destroy(m_offscreenDepthRT);
  m_alloc.destroy(m_save);

  // Creating the color image
  vk::SamplerCreateInfo sampler =
      vk::SamplerCreateInfo({}, vk::Filter::eLinear, vk::Filter::eLinear,
                            vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eMirroredRepeat,
                            vk::SamplerAddressMode::eMirroredRepeat,
                            vk::SamplerAddressMode::eMirroredRepeat);
  sampler.setUnnormalizedCoordinates(false);
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment
                                                       | vk::ImageUsageFlagBits::eSampled
                                                       | vk::ImageUsageFlagBits::eStorage
                                                       | vk::ImageUsageFlagBits::eTransferSrc);


    m_offscreenColorImage = m_alloc.createImage(colorCreateInfo);
    vk::ImageViewCreateInfo ivInfo =
        nvvk::makeImageViewCreateInfo(m_offscreenColorImage.image, colorCreateInfo);
    m_offscreenColor = m_alloc.createTexture(m_offscreenColorImage, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  {
    auto normalCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenNormalFormat,
                                                        vk::ImageUsageFlagBits::eColorAttachment
                                                        | vk::ImageUsageFlagBits::eSampled
                                                        | vk::ImageUsageFlagBits::eStorage);


    m_offscreenNormalImage = m_alloc.createImage(normalCreateInfo);
    vk::ImageViewCreateInfo ivInfo =
        nvvk::makeImageViewCreateInfo(m_offscreenNormalImage.image, normalCreateInfo);
    m_offscreenNormal = m_alloc.createTexture(m_offscreenNormalImage, ivInfo, sampler);
    m_offscreenNormal.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  {
    auto depthCreateInfoRT = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormatRT,
                                                         vk::ImageUsageFlagBits::eColorAttachment
                                                         | vk::ImageUsageFlagBits::eSampled
                                                         | vk::ImageUsageFlagBits::eStorage);


    m_offscreenDepthImageRT = m_alloc.createImage(depthCreateInfoRT);
    vk::ImageViewCreateInfo ivInfo =
        nvvk::makeImageViewCreateInfo(m_offscreenDepthImageRT.image, depthCreateInfoRT);
    m_offscreenDepthRT = m_alloc.createTexture(m_offscreenDepthImageRT, ivInfo, sampler);
    m_offscreenDepthRT.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  {
    auto idCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenIdFormat,
                                                    vk::ImageUsageFlagBits::eColorAttachment
                                                    | vk::ImageUsageFlagBits::eSampled
                                                    | vk::ImageUsageFlagBits::eStorage);


    m_offscreenIdImage = m_alloc.createImage(idCreateInfo);
    vk::ImageViewCreateInfo ivInfo =
        nvvk::makeImageViewCreateInfo(m_offscreenIdImage.image, idCreateInfo);
    m_offscreenId = m_alloc.createTexture(m_offscreenIdImage, ivInfo, sampler);
    m_offscreenId.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  {
    auto idCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_saveFormat,
                                                    vk::ImageUsageFlagBits::eColorAttachment
                                                    | vk::ImageUsageFlagBits::eSampled
                                                    | vk::ImageUsageFlagBits::eStorage
                                                    | vk::ImageUsageFlagBits::eTransferSrc);


    m_saveImage                    = m_alloc.createImage(idCreateInfo);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(m_saveImage.image, idCreateInfo);
    m_save                         = m_alloc.createTexture(m_saveImage, ivInfo, sampler);
    m_save.descriptor.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
  }


  // Creating the depth buffer
  auto depthCreateInfo =
      nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat,
                                  vk::ImageUsageFlagBits::eDepthStencilAttachment);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_offscreenDepthFormat);
    depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
    depthStencilView.setImage(image.image);

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenNormal.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepthRT.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenId.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_save.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);

    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::ImageAspectFlagBits::eDepth);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass =
        nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                               true, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral);
  }

  // Creating the frame buffer for offscreen
  std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                            m_offscreenDepth.descriptor.imageView};

  m_device.destroy(m_offscreenFramebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_offscreenRenderPass);
  info.setAttachmentCount(2);
  info.setPAttachments(attachments.data());
  info.setWidth(m_size.width);
  info.setHeight(m_size.height);
  info.setLayers(1);
  m_offscreenFramebuffer = m_device.createFramebuffer(info);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0,
                                              sizeof(PostPushConstant)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<std::string> paths = defaultSearchPaths;

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout,
                                                            m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/post.frag.spv", true, paths),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_postPipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_postDescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayoutBind.addBinding(vkDS(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayoutBind.addBinding(vkDS(2, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayoutBind.addBinding(vkDS(3, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayoutBind.addBinding(vkDS(4, vkDT::eStorageImage, 1, vkSS::eFragment));
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  {
    vk::WriteDescriptorSet writeDescriptorSets =
        m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
  {
    vk::WriteDescriptorSet writeDescriptorSets =
        m_postDescSetLayoutBind.makeWrite(m_postDescSet, 1, &m_offscreenNormal.descriptor);
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
  {
    vk::WriteDescriptorSet writeDescriptorSets =
        m_postDescSetLayoutBind.makeWrite(m_postDescSet, 2, &m_offscreenDepthRT.descriptor);
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
  {
    vk::WriteDescriptorSet writeDescriptorSets =
        m_postDescSetLayoutBind.makeWrite(m_postDescSet, 3, &m_offscreenId.descriptor);
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
  {
    vk::WriteDescriptorSet writeDescriptorSets =
        m_postDescSetLayoutBind.makeWrite(m_postDescSet, 4, &m_save.descriptor);
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(vk::CommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  m_postPushConstants.aspectRatio = aspectRatio;
  m_postPushConstants.width       = m_size.width;
  m_postPushConstants.height      = m_size.height;
  cmdBuf.pushConstants<PostPushConstant>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment,
                                         0, m_postPushConstants);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0,
                            m_postDescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  auto properties = m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
      vk::PhysicalDeviceRayTracingPropertiesKHR>();
  m_rtProperties  = properties.get<vk::PhysicalDeviceRayTracingPropertiesKHR>();
  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a OBJ primitive to the ray tracing geometry used for the BLAS
//
nvvk::RaytracingBuilderKHR::Blas HelloVulkan::objectToVkGeometryKHR(const ObjModel& model)
{
  // Setting up the creation info of acceleration structure
  vk::AccelerationStructureCreateGeometryTypeInfoKHR asCreate;
  asCreate.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asCreate.setIndexType(vk::IndexType::eUint32);
  asCreate.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  asCreate.setMaxPrimitiveCount(model.nbIndices / 3);  // Nb triangles
  asCreate.setMaxVertexCount(model.nbVertices);
  asCreate.setAllowsTransforms(VK_FALSE);  // No adding transformation matrices

  // Building part
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({model.vertexBuffer.buffer});
  vk::DeviceAddress indexAddress  = m_device.getBufferAddress({model.indexBuffer.buffer});

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(asCreate.vertexFormat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(VertexObj));
  triangles.setIndexType(asCreate.indexType);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(asCreate.geometryType);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
  asGeom.geometry.setTriangles(triangles);

  // The primitive itself
  vk::AccelerationStructureBuildOffsetInfoKHR offset;
  offset.setFirstVertex(0);
  offset.setPrimitiveCount(asCreate.maxPrimitiveCount);
  offset.setPrimitiveOffset(0);
  offset.setTransformOffset(0);

  // Our blas is only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::Blas blas;
  blas.asGeometry.emplace_back(asGeom);
  blas.asCreateGeometryInfo.emplace_back(asCreate);
  blas.asBuildOffsetInfo.emplace_back(offset);

  return blas;
}

//--------------------------------------------------------------------------------------------------
//
//
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::Blas> allBlas;
  allBlas.reserve(m_objModel.size());
  for(const auto& obj : m_objModel)
  {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    allBlas.emplace_back(blas);
  }
  m_rtBuilder.buildBlas(allBlas, 0,
                        vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace
                        | vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction);
}

void HelloVulkan::createTopLevelAS()
{
  std::vector<nvvk::RaytracingBuilderKHR::Instance> tlas;
  tlas.reserve(m_objInstance.size());
  for(int i = 0; i < static_cast<int>(m_objInstance.size()); i++)
  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform  = m_objInstance[i].transform;  // Position of the instance
    rayInst.instanceId = i;                           // gl_InstanceID
    rayInst.blasId     = m_objInstance[i].objIndex;
    rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
    rayInst.flags      = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask       = 1 << i;
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, 0, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.addBinding(vkDSLB(0, vkDT::eAccelerationStructureKHR, 1,
                                          vkSS::eRaygenKHR | vkSS::eClosestHitKHR));  // TLAS
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));  // Output image

  m_rtDescSetLayoutBind.addBinding(vkDSLB(2, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));

  m_rtDescSetLayoutBind.addBinding(vkDSLB(3, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));

  m_rtDescSetLayoutBind.addBinding(vkDSLB(4, vkDT::eStorageImage, 1, vkSS::eRaygenKHR));
  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure(0);
  vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&tlas);
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};

  vk::DescriptorImageInfo normalInfo{
      {}, m_offscreenNormal.descriptor.imageView, vk::ImageLayout::eGeneral};

  vk::DescriptorImageInfo depthInfo{
      {}, m_offscreenDepthRT.descriptor.imageView, vk::ImageLayout::eGeneral};

  vk::DescriptorImageInfo idInfo{{}, m_offscreenId.descriptor.imageView, vk::ImageLayout::eGeneral};
  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 0, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 1, &imageInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 2, &normalInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 3, &depthInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 4, &idInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
  m_device.updateDescriptorSets(wds, nullptr);
  vk::DescriptorImageInfo normalInfo{
      {}, m_offscreenNormal.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet nds{m_rtDescSet, 2, 0, 1, vkDT::eStorageImage, &normalInfo};
  m_device.updateDescriptorSets(nds, nullptr);

  vk::DescriptorImageInfo depthInfo{
      {}, m_offscreenDepthRT.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet dds{m_rtDescSet, 3, 0, 1, vkDT::eStorageImage, &depthInfo};
  m_device.updateDescriptorSets(dds, nullptr);

  vk::DescriptorImageInfo idInfo{{}, m_offscreenId.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet  ids{m_rtDescSet, 4, 0, 1, vkDT::eStorageImage, &idInfo};
  m_device.updateDescriptorSets(ids, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;

  vk::ShaderModule raygenSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rgen.spv", true, paths));

  vk::ShaderModule prepassRaygenSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/prepass.rgen.spv", true, paths));


  vk::ShaderModule missSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rmiss.spv", true, paths));

  // The second miss shader is invoked when a shadow ray misses the geometry. It
  // simply indicates that no occlusion has been found
  vk::ShaderModule shadowmissSM =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/raytraceShadow.rmiss.spv", true, paths));

  vk::ShaderModule celmissSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/cel.rmiss.spv", true, paths));

  vk::ShaderModule photonmissSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/prepass.rmiss.spv", true, paths));


  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoKHR rg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, raygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(rg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, prepassRaygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(rg);
  // Miss
  vk::RayTracingShaderGroupCreateInfoKHR mg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, missSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, celmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, photonmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);

  // Hit Group - Closest Hit + AnyHit
  vk::ShaderModule chitSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rchit.spv", true, paths));

  vk::ShaderModule celchitSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/cel.rchit.spv", true, paths));

  vk::ShaderModule eyechitSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/eye.rchit.spv", true, paths));

  vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, celchitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, eyechitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);

  vk::RayTracingShaderGroupCreateInfoKHR cg{vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};


  vk::ShaderModule lambertSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/lambert.rcall.spv", true, paths));

  vk::ShaderModule blinnSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/blinn.rcall.spv", true, paths));

  vk::ShaderModule mirrorSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/mirror.rcall.spv", true, paths));

  vk::ShaderModule glassSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/glass.rcall.spv", true, paths));

  vk::ShaderModule pointEmittSM =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/PointEmission.rcall.spv", true, paths));

  vk::ShaderModule areaEmittSM =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/AreaEmission.rcall.spv", true, paths));


  vk::ShaderModule pointDirectSampleSM =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/PointSampleDirect.rcall.spv", true, paths));
  vk::ShaderModule areaDirectSampleSM =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/AreaSampleDirect.rcall.spv", true, paths));

  vk::ShaderModule celSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/cel.rcall.spv", true, paths));


  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, lambertSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, blinnSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, mirrorSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, glassSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, pointEmittSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, areaEmittSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, pointDirectSampleSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, areaDirectSampleSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableKHR, celSM, "main"});
  cg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(cg);


  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR
                                     | vk::ShaderStageFlagBits::eClosestHitKHR
                                     | vk::ShaderStageFlagBits::eMissKHR
                                     | vk::ShaderStageFlagBits::eCallableKHR,
                                     0, sizeof(RtPushConstant)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline

  vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
                                    m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

  rayPipelineInfo.setMaxRecursionDepth(2);  // Ray depth
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = m_device.createRayTracingPipelineKHR({}, rayPipelineInfo).value;

  m_device.destroy(raygenSM);
  m_device.destroy(prepassRaygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(celmissSM);
  m_device.destroy(chitSM);
  m_device.destroy(celchitSM);
  m_device.destroy(lambertSM);
  m_device.destroy(blinnSM);
  m_device.destroy(mirrorSM);
  m_device.destroy(glassSM);
  m_device.destroy(pointDirectSampleSM);
  m_device.destroy(areaDirectSampleSM);
  m_device.destroy(pointEmittSM);
  m_device.destroy(areaEmittSM);
  m_device.destroy(celSM);
  m_device.destroy(eyechitSM);
  m_device.destroy(photonmissSM);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void HelloVulkan::createRtShaderBindingTable()
{
  auto groupCount =
      static_cast<uint32_t>(m_rtShaderGroups.size());               // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  uint32_t baseAlignment   = m_rtProperties.shaderGroupBaseAlignment;  // Size of shader alignment

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * baseAlignment;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesKHR(m_rtPipeline, 0, groupCount, sbtSize,
                                              shaderHandleStorage.data());
  // Write the handles in the SBT
  m_rtSBTBuffer = m_alloc.createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
                                       vk::MemoryPropertyFlagBits::eHostVisible
                                       | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc.map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += baseAlignment;
  }
  m_alloc.unmap(m_rtSBTBuffer);


  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const vk::CommandBuffer& cmdBuf,
                           const nvmath::vec4f&     clearColor,
                           int                      pass)
{
  if(pass == 0)
  {
    updateFrame();
  }
  else
  {
    m_FrameCount++;
  }

  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_rtPushConstants.clearColor     = clearColor;
  m_rtPushConstants.lightPosition  = m_pushConstant.lightPosition;
  m_rtPushConstants.lightColor     = m_pushConstant.lightColor;
  m_rtPushConstants.lightType      = m_pushConstant.lightType;
  m_rtPushConstants.numObjs        = m_AreaLightsPerObject.size();
  m_rtPushConstants.numAreaSamples = m_numAreaSamples;
  m_rtPushConstants.frame          = m_FrameCount;
  m_rtPushConstants.numSamples     = m_numSamples;
  m_rtPushConstants.fuzzyAngle     = m_fuzzyAngle;
  m_rtPushConstants.ior            = m_IOR;
  m_rtPushConstants.numLights      = m_AreaLightsPerObject.size();
  m_rtPushConstants.max_russian    = m_maxRussian;
  m_rtPushConstants.numPointLights = m_PointLights.size();
  m_rtPushConstants.numIds         = ObjLoader::id_counter;
  m_rtPushConstants.pass           = pass;
  m_rtPushConstants.width          = m_size.width;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0,
                            {m_rtDescSet, m_descSet}, {});
  cmdBuf.pushConstants<RtPushConstant>(m_rtPipelineLayout,
                                       vk::ShaderStageFlagBits::eRaygenKHR
                                       | vk::ShaderStageFlagBits::eClosestHitKHR
                                       | vk::ShaderStageFlagBits::eMissKHR
                                       | vk::ShaderStageFlagBits::eCallableKHR,
                                       0, m_rtPushConstants);

  vk::DeviceSize progSize =
      m_rtProperties.shaderGroupBaseAlignment;           // Size of a program identifier
  vk::DeviceSize rayGenOffset        = pass * progSize;  // Start at the beginning of m_sbtBuffer
  vk::DeviceSize missOffset          = 2u * progSize;    // Jump over raygen
  vk::DeviceSize hitGroupOffset      = 6u * progSize;    // Jump over the previous shaders
  vk::DeviceSize callableGroupOffset = 9u * progSize;

  vk::DeviceSize sbtSize = progSize * (vk::DeviceSize)m_rtShaderGroups.size();

  const vk::StridedBufferRegionKHR raygenShaderBindingTable = {m_rtSBTBuffer.buffer, rayGenOffset,
                                                               progSize, sbtSize};
  const vk::StridedBufferRegionKHR missShaderBindingTable   = {m_rtSBTBuffer.buffer, missOffset,
                                                               progSize, sbtSize};
  const vk::StridedBufferRegionKHR hitShaderBindingTable    = {m_rtSBTBuffer.buffer, hitGroupOffset,
                                                               progSize, sbtSize};
  const vk::StridedBufferRegionKHR callableShaderBindingTable = {
      m_rtSBTBuffer.buffer, callableGroupOffset, progSize, sbtSize};
  if(pass == 1)
  {
    cmdBuf.traceRaysKHR(&raygenShaderBindingTable, &missShaderBindingTable, &hitShaderBindingTable,
                        &callableShaderBindingTable,  //
                        m_size.width, 1, 1);          //
  }
  else
  {
    cmdBuf.traceRaysKHR(&raygenShaderBindingTable, &missShaderBindingTable, &hitShaderBindingTable,
                        &callableShaderBindingTable,  //
                        m_size.width, m_size.height, 1);
  }


  m_debug.endLabel(cmdBuf);
}

void HelloVulkan::resetFrame()
{
  m_FrameCount = -1;
}

void HelloVulkan::updateFrame()
{
  static nvmath::mat4f refCamMatrix;

  auto& m = CameraManip.getMatrix();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0)
  {
    resetFrame();
    refCamMatrix = m;
  }
  m_FrameCount++;
}

void HelloVulkan::addPointLight(PointLight p)
{
  m_PointLights.push_back(p);
}

uint32_t HelloVulkan::getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
  vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &deviceMemoryProperties);
  for(uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
  {
    if((typeBits & 1) == 1)
    {
      if((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
      {
        return i;
      }
    }
    typeBits >>= 1;
  }
  return 0;
}

void HelloVulkan::saveImage()
{
  float* imagedata;
  {
    // Create the linear tiled destination image to copy to and to read the memory from
    VkImageCreateInfo imgCreateInfo(vks::initializers::imageCreateInfo());
    imgCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgCreateInfo.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
    imgCreateInfo.extent.width  = m_size.width;
    imgCreateInfo.extent.height = m_size.height;
    imgCreateInfo.extent.depth  = 1;
    imgCreateInfo.arrayLayers   = 1;
    imgCreateInfo.mipLevels     = 1;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCreateInfo.tiling        = VK_IMAGE_TILING_LINEAR;
    imgCreateInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Create the image
    VkImage dstImage;
    VK_CHECK_RESULT(vkCreateImage(m_device, &imgCreateInfo, nullptr, &dstImage));
    // Create memory to back up the image
    VkMemoryRequirements memRequirements;
    VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
    VkDeviceMemory       dstImageMemory;
    vkGetImageMemoryRequirements(m_device, dstImage, &memRequirements);
    memAllocInfo.allocationSize = memRequirements.size;
    // Memory must be host visible to copy from
    memAllocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &memAllocInfo, nullptr, &dstImageMemory));
    VK_CHECK_RESULT(vkBindImageMemory(m_device, dstImage, dstImageMemory, 0));

    // Do the actual blit from the offscreen image to our host visible destination image

    nvvk::CommandPool          cmdGen(m_device, m_graphicsQueueIndex);
    std::vector<nvmath::vec4f> img(m_size.width * m_size.height, {0, 0, 0, 0});

    vk::CommandBuffer cmdBuf = cmdGen.createCommandBuffer();

    vks::tools::insertImageMemoryBarrier(
        cmdBuf, m_save.image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});


    // Transition destination image to transfer destination layout
    vks::tools::insertImageMemoryBarrier(
        cmdBuf, dstImage, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // colorAttachment.image is already in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, and does not need to be transitioned

    VkImageCopy imageCopyRegion{};
    imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.srcSubresource.layerCount = 1;
    imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.dstSubresource.layerCount = 1;
    imageCopyRegion.extent.width              = m_size.width;
    imageCopyRegion.extent.height             = m_size.height;
    imageCopyRegion.extent.depth              = 1;

    vkCmdCopyImage(cmdBuf, m_saveImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);

    // Transition destination image to general layout, which is the required layout for mapping the image memory later on
    vks::tools::insertImageMemoryBarrier(
        cmdBuf, dstImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    cmdGen.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();


    // Get layout of the image (including row pitch)
    VkImageSubresource subResource{};
    subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkSubresourceLayout subResourceLayout;

    vkGetImageSubresourceLayout(m_device, dstImage, &subResource, &subResourceLayout);

    // Map image memory so we can start copying from it
    vkMapMemory(m_device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&imagedata);
    imagedata += subResourceLayout.offset;
    stbi_write_hdr("/tmp/testy.hdr", m_size.width, m_size.height, 4, imagedata);

    std::vector<uint8_t> pngdata(m_size.width * m_size.height * 4, 0);
    for(int i = 0; i < m_size.width * m_size.height * 4; i++)
    {
      pngdata[i] = static_cast<uint8_t>(std::min(255.0f, imagedata[i] * 255));
      if(imagedata[i] < 0.0 || isnan(imagedata[i]))
      {
        std::cout << i / 3 << std::endl;
        imagedata[i] = 0;
      }
    }
    stbi_write_png("/tmp/testy.png", m_size.width, m_size.height, 4, pngdata.data(), 0);
    vkUnmapMemory(m_device, dstImageMemory);
    vkDestroyImage(m_device, dstImage, nullptr);
    vkFreeMemory(m_device, dstImageMemory, nullptr);

    return;
  }
}

void HelloVulkan::postFrameWork()
{
  return;
  /*VkCommandPool           commandPool;
  VkCommandPoolCreateInfo cmdPoolInfo = {};
  cmdPoolInfo.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmdPoolInfo.queueFamilyIndex        = 1;
  cmdPoolInfo.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK_RESULT(vkCreateCommandPool(m_device, &cmdPoolInfo, nullptr, &commandPool));

  float* imagedata;
  {
    // Create the linear tiled destination image to copy to and to read the memory from
    VkImageCreateInfo imgCreateInfo(vks::initializers::imageCreateInfo());
    imgCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
    imgCreateInfo.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
    imgCreateInfo.extent.width  = m_size.width;
    imgCreateInfo.extent.height = m_size.height;
    imgCreateInfo.extent.depth  = 1;
    imgCreateInfo.arrayLayers   = 1;
    imgCreateInfo.mipLevels     = 1;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imgCreateInfo.tiling        = VK_IMAGE_TILING_LINEAR;
    imgCreateInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Create the image
    VkImage dstImage;
    VK_CHECK_RESULT(vkCreateImage(m_device, &imgCreateInfo, nullptr, &dstImage));
    // Create memory to back up the image
    VkMemoryRequirements memRequirements;
    VkMemoryAllocateInfo memAllocInfo(vks::initializers::memoryAllocateInfo());
    VkDeviceMemory       dstImageMemory;
    vkGetImageMemoryRequirements(m_device, dstImage, &memRequirements);
    memAllocInfo.allocationSize = memRequirements.size;
    // Memory must be host visible to copy from
    memAllocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &memAllocInfo, nullptr, &dstImageMemory));
    VK_CHECK_RESULT(vkBindImageMemory(m_device, dstImage, dstImageMemory, 0));

    // Do the actual blit from the offscreen image to our host visible destination image
    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
        vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                     1);
    nvvk::CommandPool          cmdGen(m_device, m_graphicsQueueIndex);
    std::vector<nvmath::vec4f> img(m_size.width * m_size.height, {0, 0, 0, 0});

    vk::CommandBuffer cmdBuf = cmdGen.createCommandBuffer();


    // Transition destination image to transfer destination layout
    vks::tools::insertImageMemoryBarrier(
        cmdBuf, dstImage, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    // colorAttachment.image is already in VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, and does not need to be transitioned

    VkImageCopy imageCopyRegion{};
    imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.srcSubresource.layerCount = 1;
    imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.dstSubresource.layerCount = 1;
    imageCopyRegion.extent.width              = m_size.width;
    imageCopyRegion.extent.height             = m_size.height;
    imageCopyRegion.extent.depth              = 1;

    vkCmdCopyImage(cmdBuf, m_saveImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopyRegion);

    // Transition destination image to general layout, which is the required layout for mapping the image memory later on
    vks::tools::insertImageMemoryBarrier(
        cmdBuf, dstImage, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});

    cmdGen.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();


    // Get layout of the image (including row pitch)
    VkImageSubresource subResource{};
    subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkSubresourceLayout subResourceLayout;

    vkGetImageSubresourceLayout(m_device, dstImage, &subResource, &subResourceLayout);

    // Map image memory so we can start copying from it
    vkMapMemory(m_device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&imagedata);
    imagedata += subResourceLayout.offset;
    std::vector<float> max = std::vector<float>(m_size.height, 0);
#pragma omp parallel for
    for(int j = 0; j < m_size.height; j++)
    {

      for(int i = 0; i < m_size.width; i += 4)
      {
        max[j] = std::max(imagedata[j * m_size.width + i] + imagedata[j * m_size.width + i + 1]
                              + imagedata[j * m_size.width + i + 2],
                          max[j]);
      }
    }
    m_rtPushConstants.maxillum = *(std::max_element(std::begin(max), std::end(max)));
    
    return;
  }*/
}

void HelloVulkan::savePhotons()
{

  Photon* photonData;
  {
    VkBufferCreateInfo bfCreateInfo(vks::initializers::bufferCreateInfo());
    bfCreateInfo.sType       = VkStructureType::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bfCreateInfo.size        = sizeof(Photon) * m_size.width * 32;
    bfCreateInfo.usage       = VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bfCreateInfo.sharingMode = VkSharingMode::VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer dstBuffer;
    VK_CHECK_RESULT(vkCreateBuffer(m_device, &bfCreateInfo, nullptr, &dstBuffer));
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, dstBuffer, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    VkDeviceMemory bufferMemory;
    allocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                   | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory));
    VK_CHECK_RESULT(vkBindBufferMemory(m_device, dstBuffer, bufferMemory, 0));


    // Do the actual blit from the offscreen image to our host visible destination image

    nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);


    vk::CommandBuffer cmdBuf = cmdGen.createCommandBuffer();

    VkBufferCopy bufferCopyRegion{};
    bufferCopyRegion.srcOffset = 0;
    bufferCopyRegion.srcOffset = 0;
    bufferCopyRegion.size      = sizeof(Photon) * m_size.width * 32;
    vkCmdCopyBuffer(cmdBuf, m_photonBuffer.buffer, dstBuffer, 1, &bufferCopyRegion);


    cmdGen.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();

    vkMapMemory(m_device, bufferMemory, 0, VK_WHOLE_SIZE, 0, (void**)&photonData);

    for(int i = 0; i< 32*m_size.width; i++){
      if(photonData[i].used > 0){
        m_photons.push_back(photonData[i]);
      }
    }
    std::cout << m_photons.size() << std::endl;



    vkUnmapMemory(m_device, bufferMemory);
    vkDestroyBuffer(m_device, dstBuffer, nullptr);
    vkFreeMemory(m_device, bufferMemory, nullptr);








  }
  {
    VkBufferCreateInfo bfCreateInfo(vks::initializers::bufferCreateInfo());
    bfCreateInfo.sType       = VkStructureType::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bfCreateInfo.size        = sizeof(Photon) * m_size.width * 32;
    bfCreateInfo.usage       = VkBufferUsageFlagBits::VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bfCreateInfo.sharingMode = VkSharingMode::VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer srcBuffer;
    VK_CHECK_RESULT(vkCreateBuffer(m_device, &bfCreateInfo, nullptr, &srcBuffer));
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, srcBuffer, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    VkDeviceMemory bufferMemory;
    allocInfo.memoryTypeIndex = getMemoryTypeIndex(memRequirements.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                   | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory));
    VK_CHECK_RESULT(vkBindBufferMemory(m_device, srcBuffer, bufferMemory, 0));

    void* data;
    vkMapMemory(m_device, bufferMemory, 0, m_size.width * 32 * sizeof(Photon), 0, &data);
    memset(data, 0, sizeof(Photon) * 32 * m_size.width);
    vkUnmapMemory(m_device, bufferMemory);


    // Do the actual blit from the offscreen image to our host visible destination image

    nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);


    vk::CommandBuffer cmdBuf = cmdGen.createCommandBuffer();

    VkBufferCopy bufferCopyRegion{};
    bufferCopyRegion.srcOffset = 0;
    bufferCopyRegion.srcOffset = 0;
    bufferCopyRegion.size      = sizeof(Photon) * m_size.width * 32;
    vkCmdCopyBuffer(cmdBuf, srcBuffer, m_photonBuffer.buffer, 1, &bufferCopyRegion);



    cmdGen.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();


    vkDestroyBuffer(m_device, srcBuffer, nullptr);
    vkFreeMemory(m_device, bufferMemory, nullptr);
  }



}

void HelloVulkan::calculatePhotons()
{

}
