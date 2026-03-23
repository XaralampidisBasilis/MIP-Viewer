import * as tf from '@tensorflow/tfjs'

export function setTensorflowFlags()
{
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', true)
    tf.env().set('WEBGL_PACK', true)
    tf.env().set('WEBGL_LAZILY_UNPACK', true)
    tf.env().set('WEBGL_CPU_FORWARD', false)
    tf.env().set('WEBGL_VERSION', 2)
    tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", -1) // -1, 0, 8, 16, 32, 64, 128, 512 * 512 * 512: avoid gpu memory buildup and reuse
    tf.env().set('WEBGL_FLUSH_THRESHOLD', -1) // -1, 1: avoid unnecessary flush overhead
}

export function logTensorflowFlags()
{
    const gl = tf.backend().getGPGPUContext().gl

    console.log('backend', tf.getBackend())
    
    console.log('GL_VERSION', gl.getParameter(gl.VERSION))
    console.log('GLSL_VERSION', gl.getParameter(gl.SHADING_LANGUAGE_VERSION))
    console.log('RENDERER', gl.getParameter(gl.RENDERER))
    console.log('VENDOR', gl.getParameter(gl.VENDOR))
    console.log('MAX_TEXTURE_SIZE', gl.getParameter(gl.MAX_TEXTURE_SIZE))
    console.log('MAX_RENDERBUFFER_SIZE', gl.getParameter(gl.MAX_RENDERBUFFER_SIZE))

    console.log('WEBGL_VERSION', tf.env().getNumber('WEBGL_VERSION'))
    console.log('WEBGL_FORCE_F16_TEXTURES', tf.env().getBool('WEBGL_FORCE_F16_TEXTURES'))
    console.log('WEBGL_PACK', tf.env().getBool('WEBGL_PACK'))
    console.log('WEBGL_LAZILY_UNPACK', tf.env().getBool('WEBGL_LAZILY_UNPACK'))
    console.log('WEBGL_CPU_FORWARD', tf.env().getBool('WEBGL_CPU_FORWARD'))
    console.log('WEBGL_DELETE_TEXTURE_THRESHOLD', tf.env().getNumber('WEBGL_DELETE_TEXTURE_THRESHOLD'))
    console.log('WEBGL_FLUSH_THRESHOLD', tf.env().getNumber('WEBGL_FLUSH_THRESHOLD'))
    console.log('WEBGL_RENDER_FLOAT32_CAPABLE', tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE'))
    console.log('WEBGL_RENDER_FLOAT32_ENABLED', tf.env().getBool('WEBGL_RENDER_FLOAT32_ENABLED'))
}

export async function setTensorflow() 
{
    console.time('setTensorflow')

    tf.enableProdMode()

    setTensorflowFlags()
    // logTensorflowFlags()

    await tf.setBackend('webgl')
    await tf.ready()

    console.timeEnd('setTensorflow')
}