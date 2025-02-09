// Copyright (c) 2020 Michael Koesel and respective contributors
// SPDX-License-Identifier: MIT
// See accompanying LICENSE file for detailed information

#include "dogm/mapping/opengl/framebuffer.h"
#include "dogm/cuda_utils.h"

#include <cstdio>
#include <memory.h>

Framebuffer::Framebuffer(int width, int height)
{
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    CUDA_CALL(cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Framebuffer::~Framebuffer()
{
    glDeleteFramebuffers(1, &framebuffer);
    glDeleteTextures(1, &texture);
}

void Framebuffer::beginCudaAccess(cudaSurfaceObject_t* surfaceObject)
{
    CUDA_CALL(cudaGraphicsMapResources(1, &resource, nullptr));

    cudaArray_t cudaArray;
    CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&cudaArray, resource, 0, 0));

    cudaResourceDesc resourceDesc;
    memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = cudaArray;

    CUDA_CALL(cudaCreateSurfaceObject(surfaceObject, &resourceDesc));
}

void Framebuffer::endCudaAccess(cudaSurfaceObject_t surfaceObject)
{
    CUDA_CALL(cudaGraphicsUnmapResources(1, &resource, nullptr));
    CUDA_CALL(cudaDestroySurfaceObject(surfaceObject));
}

void Framebuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
}

void Framebuffer::unbind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
