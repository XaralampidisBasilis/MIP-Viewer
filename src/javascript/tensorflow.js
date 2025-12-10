import * as tf from '@tensorflow/tfjs'

export async function setTensorflow() 
{
    console.time('setTensorflow')

    tf.enableProdMode()

    tf.env().set('WEBGL_FORCE_F16_TEXTURES', false)
    tf.env().set('WEBGL_PACK', true)
    tf.env().set('WEBGL_CPU_FORWARD', false)
    tf.env().set('WEBGL_VERSION', 2)
    tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0)
    tf.env().set('WEBGL_FLUSH_THRESHOLD', 1)

    await tf.setBackend('webgl')
    await tf.ready()

    console.timeEnd('setTensorflow')
}