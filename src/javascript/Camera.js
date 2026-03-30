import * as THREE from 'three'
import Experience from './Experience'
import EventEmitter from './Utils/EventEmitter'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'

const FRAME_DIRECTION = new THREE.Vector3(1, 1, 1).normalize()

export default class Camera extends EventEmitter
{
    constructor()
    {
        super()

        this.experience = new Experience()
        this.mouse = this.experience.mouse
        this.sizes = this.experience.sizes
        this.scene = this.experience.scene
        this.canvas = this.experience.canvas

        this.setInstance()
        this.setControls()
    }

    setInstance()
    {
        this.orthographic =
        {
            near: 0.001,
            far: 10,
            frustumHeight: 2,
        }

        this.instance = new THREE.OrthographicCamera(
            -1, 1, 1, -1,
            this.orthographic.near,
            this.orthographic.far
        )

        this.updateFrustumAspect(this.getCSSAspect())
        this.instance.position.set(1, 1, 1)
        this.scene.add(this.instance)
    }

    setControls()
    {
        this.controls = new ArcballControls(this.instance, this.canvas, this.scene)
        this.controls.enableAnimations = true
        this.controls.enableGrid = false
        this.controls.cursorZoom = true
        this.controls.rotateSpeed = 1.0
        this.controls.minZoom = 0.5
        this.controls.maxZoom = 8.0

        this.controls.setGizmosVisible(false)
        this.controls.update()
    }

    frameBounds(center, size)
    {
        const radius = Math.max(size.length() / 2.0, 0.001)

        this.orthographic.frustumHeight = radius * 2.4
        this.instance.near = 0.001
        this.instance.far = radius * 6.0
        this.instance.position.copy(center).addScaledVector(FRAME_DIRECTION, radius * 2.0)

        this.controls.target.copy(center)
        this.resize()
        this.trigger('change')
    }

    getCSSAspect()
    {
        const width = Math.max(this.sizes.width, 1)
        const height = Math.max(this.sizes.height, 1)

        return width / height
    }

    updateFrustumAspect(aspect)
    {
        const safeAspect = Math.max(aspect, 1e-6)
        const halfHeight = this.orthographic.frustumHeight * 0.5
        const halfWidth = halfHeight * safeAspect

        this.instance.left = -halfWidth
        this.instance.right = halfWidth
        this.instance.top = halfHeight
        this.instance.bottom = -halfHeight
    }

    resize()
    {
        this.updateFrustumAspect(this.getCSSAspect())
        this.instance.updateProjectionMatrix()
        this.instance.updateWorldMatrix(true, false)
        this.controls.update()
        this.trigger('change')
    }

    rescale()
    {
        this.controls.update()
        this.instance.updateWorldMatrix(true, false)
        this.trigger('change')
    }

    update()
    {
        this.controls.update()
        this.instance.updateWorldMatrix(true, false)
        this.trigger('change')
    }

    destroy()
    {
        this.scene.remove(this.instance)

        if (this.controls)
        {
            this.controls.dispose()
            this.controls = null
        }

        this.instance = null
        this.experience = null
        this.mouse = null
        this.sizes = null
        this.scene = null
        this.canvas = null

        console.log('Camera destroyed')
    }
}