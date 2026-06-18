import * as tf from '@tensorflow/tfjs'
import { setTensorflow } from '../../tensorflow'
import { getWebGPUComputeContext } from '../../WebGPU/WebGPUDevice'
import { computePackedDistanceTexture } from '../Programs/ComputePackedDistanceTexture'
import { computePackedDistanceTextureWebGPU } from '../Programs/ComputeWebGPUPackedDistanceTexture'
import { WebGPUTensor3D } from './WebGPUTensor3D'
import { map3dInPlaceWebGPU, map3dWebGPU, reduceMinMaxWebGPU, resizeTrilinearWebGPU } from './WebGPUVolumeKernels'

export async function runWebGPUComputeSmokeTest()
{
    await setTensorflow()

    const shape = [2, 2, 2]
    const data = new Float32Array([
        0.00, 0.13,
        0.41, 0.72,
        0.28, 0.95,
        0.36, 1.00,
    ])
    const tensor = tf.tensor3d(data, shape)
    const { device } = await getWebGPUComputeContext()
    const gpuTensor = WebGPUTensor3D.fromTypedArray(device, shape, data, 'float32', 'smoke-volume')
    const gpuTensorInPlace = WebGPUTensor3D.fromTypedArray(device, shape, data, 'float32', 'smoke-volume-in-place')
    const [minValue, maxValue] = await reduceMinMaxWebGPU(gpuTensor)
    const mapped = await map3dWebGPU(gpuTensor, minValue, maxValue)
    await map3dInPlaceWebGPU(gpuTensorInPlace, minValue, maxValue)
    const resized = await resizeTrilinearWebGPU(mapped, [1, 1, 1], false, true)
    const mappedData = await mapped.read()
    const mappedInPlaceData = await gpuTensorInPlace.read()
    const resizedData = await resized.read()

    const webgl = computePackedDistanceTexture(tensor, 'unidirectional', 0.01, 1, '1bit', false)
    const webgpu = await computePackedDistanceTextureWebGPU(tensor, 'unidirectional', 0.01, 1, '1bit', false)

    const mismatches = []

    for (let i = 0; i < webgl.data.length; i += 1)
    {
        if (webgl.data[i] !== webgpu.data[i])
        {
            mismatches.push({ index: i, webgl: webgl.data[i], webgpu: webgpu.data[i] })

            if (mismatches.length >= 16)
            {
                break
            }
        }
    }

    const result =
    {
        passed:
            mismatches.length === 0 &&
            webgl.data.length === webgpu.data.length &&
            minValue === 0 &&
            maxValue === 1 &&
            mappedData[0] === 0 &&
            mappedData[mappedData.length - 1] === 1 &&
            mappedInPlaceData[0] === mappedData[0] &&
            mappedInPlaceData[mappedInPlaceData.length - 1] === mappedData[mappedData.length - 1] &&
            Number.isFinite(resizedData[0]),
        length: webgl.data.length,
        volume:
        {
            minValue,
            maxValue,
            mappedFirst: mappedData[0],
            mappedLast: mappedData[mappedData.length - 1],
            mappedInPlaceFirst: mappedInPlaceData[0],
            mappedInPlaceLast: mappedInPlaceData[mappedInPlaceData.length - 1],
            resizedValue: resizedData[0],
        },
        dimensions:
        {
            webgl: webgl.dimensions.toArray(),
            webgpu: webgpu.dimensions.toArray(),
        },
        mismatches,
    }

    gpuTensor.dispose()
    gpuTensorInPlace.dispose()
    mapped.dispose()
    resized.dispose()
    tensor.dispose()
    webgl.texture.dispose()
    webgpu.texture.dispose()

    return result
}

if (typeof window !== 'undefined')
{
    window.runWebGPUComputeSmokeTest = runWebGPUComputeSmokeTest
}
