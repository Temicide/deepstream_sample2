/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <cmath>

#include "nvdsinfer_custom_impl.h"

#include "utils.h"

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

namespace {

constexpr uint kBoxAttrs = 4;
constexpr int kFallbackNumClasses = 80;
constexpr float kFallbackThreshold = 0.25F;

struct RawYoloShape {
  uint attrs = 0;
  uint boxes = 0;
  bool boxMajor = false;
};

static bool
isVehicleClass(const int classId)
{
  return classId == 1 || classId == 2 || classId == 3 || classId == 5 || classId == 7;
}

static float
thresholdForClass(const int classId, const std::vector<float>& preclusterThreshold)
{
  if (classId >= 0 && static_cast<size_t>(classId) < preclusterThreshold.size()) {
    return preclusterThreshold[classId];
  }
  return kFallbackThreshold;
}

static void
logOutputDimsOnce(const NvDsInferLayerInfo& output, const NvDsInferNetworkInfo& networkInfo)
{
  static bool logged = false;
  if (logged) {
    return;
  }

  std::cerr << "INFO: NvDsInferParseYolo output dims:";
  for (int i = 0; i < output.inferDims.numDims; ++i) {
    std::cerr << " " << output.inferDims.d[i];
  }
  std::cerr << " network=" << networkInfo.width << "x" << networkInfo.height
            << " mode=raw-yolov8 vehicle-classes=1,2,3,5,7" << std::endl;
  logged = true;
}

static bool
resolveRawYoloShape(const NvDsInferLayerInfo& output, RawYoloShape& shape)
{
  const int numDims = output.inferDims.numDims;
  if (numDims < 2) {
    return false;
  }

  int d0 = output.inferDims.d[0];
  int d1 = output.inferDims.d[1];
  int d2 = numDims > 2 ? output.inferDims.d[2] : 0;

  if (numDims >= 3 && d0 == 1) {
    d0 = d1;
    d1 = d2;
  }

  if (d0 <= static_cast<int>(kBoxAttrs) || d1 <= 0) {
    return false;
  }

  if (d0 <= 256 && d1 > d0) {
    shape.attrs = static_cast<uint>(d0);
    shape.boxes = static_cast<uint>(d1);
    shape.boxMajor = false;
    return true;
  }

  if (d1 <= 256 && d0 > d1) {
    shape.attrs = static_cast<uint>(d1);
    shape.boxes = static_cast<uint>(d0);
    shape.boxMajor = true;
    return true;
  }

  return false;
}

static float
rawValueAt(const float* output, const RawYoloShape& shape, const uint box, const uint attr)
{
  if (shape.boxMajor) {
    return output[(box * shape.attrs) + attr];
  }
  return output[(attr * shape.boxes) + box];
}

static NvDsInferParseObjectInfo
convertBBox(const float bx1, const float by1, const float bx2, const float by2, const uint netW, const uint netH)
{
  NvDsInferParseObjectInfo b{};

  float x1 = bx1;
  float y1 = by1;
  float x2 = bx2;
  float y2 = by2;

  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  return b;
}

static void
addBBoxProposal(const float bx1, const float by1, const float bx2, const float by2, const uint netW, const uint netH,
    const int classId, const float confidence, std::vector<NvDsInferParseObjectInfo>& binfo)
{
  NvDsInferParseObjectInfo bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);

  if (bbi.width < 1 || bbi.height < 1) {
    return;
  }

  bbi.detectionConfidence = confidence;
  bbi.classId = classId;
  binfo.push_back(bbi);
}

static std::vector<NvDsInferParseObjectInfo>
decodeRawYoloV8Tensor(const float* output, const RawYoloShape& shape, const uint netW, const uint netH,
    const std::vector<float>& preclusterThreshold)
{
  std::vector<NvDsInferParseObjectInfo> binfo;
  const uint numClasses = shape.attrs > kBoxAttrs ? shape.attrs - kBoxAttrs : kFallbackNumClasses;

  for (uint box = 0; box < shape.boxes; ++box) {
    int bestClass = -1;
    float bestScore = 0.0F;

    for (uint classId = 0; classId < numClasses; ++classId) {
      if (!isVehicleClass(static_cast<int>(classId))) {
        continue;
      }
      const float score = rawValueAt(output, shape, box, kBoxAttrs + classId);
      if (score > bestScore) {
        bestScore = score;
        bestClass = static_cast<int>(classId);
      }
    }

    if (bestClass < 0 || bestScore < thresholdForClass(bestClass, preclusterThreshold)) {
      continue;
    }

    const float cx = rawValueAt(output, shape, box, 0);
    const float cy = rawValueAt(output, shape, box, 1);
    const float w = rawValueAt(output, shape, box, 2);
    const float h = rawValueAt(output, shape, box, 3);

    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) || w <= 0.0F ||
        h <= 0.0F) {
      continue;
    }

    const float bx1 = cx - (w * 0.5F);
    const float by1 = cy - (h * 0.5F);
    const float bx2 = cx + (w * 0.5F);
    const float by2 = cy + (h * 0.5F);

    addBBoxProposal(bx1, by1, bx2, by2, netW, netH, bestClass, bestScore, binfo);
  }

  return binfo;
}

}  // namespace

static bool
NvDsInferParseCustomYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> objects;

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  logOutputDimsOnce(output, networkInfo);

  RawYoloShape shape;
  if (!resolveRawYoloShape(output, shape)) {
    std::cerr << "ERROR: Unsupported YOLO output shape in bbox parsing" << std::endl;
    return false;
  }

  std::vector<NvDsInferParseObjectInfo> outObjs = decodeRawYoloV8Tensor((const float*) (output.buffer), shape,
      networkInfo.width, networkInfo.height, detectionParams.perClassPreclusterThreshold);

  objects.insert(objects.end(), outObjs.begin(), outObjs.end());

  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseYolo(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYolo(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo);
