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
        this.onControlsChange = this.onControlsChange.bind(this)
        
        this.orthographic = {
            near: 0.001,
            far: 10,
            frustumHeight: 2,
        }

        this.setInstance()
        this.setControls()
    }

    setInstance()
    {
        this.instance = new THREE.OrthographicCamera(-1, 1, 1, -1, this.orthographic.near, this.orthographic.far)
        this.updateOrthographicFrustum()
        this.instance.position.set(1, 1, 1)
        this.scene.add(this.instance)
    }

    setControls()
    {
        if (this.controls)
        {
            this.controls.removeEventListener('change', this.onControlsChange)
            this.controls.dispose()
        }

        this.controls = new ArcballControls(this.instance, this.canvas, this.scene)
        this.controls.enableAnimations = true
        this.controls.enableGrid = false
        this.controls.cursorZoom = true
        this.controls.rotateSpeed = 1.0
        this.controls.minZoom = 0.5
        this.controls.maxZoom = 8.0
        
        this.controls.setGizmosVisible(false)
        this.controls.addEventListener('change', this.onControlsChange)
        this.controls.update()
    }

    onControlsChange()
    {
        this.instance.updateWorldMatrix(true, false)
        this.trigger('change')
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

        this.controls.saveState?.()
        this.trigger('change')
    }

    updateOrthographicFrustum()
    {
        const aspect = this.sizes.width / this.sizes.height
        const halfHeight = this.orthographic.frustumHeight * 0.5
        const halfWidth = halfHeight * aspect

        this.instance.left = -halfWidth
        this.instance.right = halfWidth
        this.instance.top = halfHeight
        this.instance.bottom = -halfHeight
    }

    setRaycaster()
    {
        this.raycaster = new THREE.Raycaster()
        this.raycaster.setFromCamera(this.mouse.ndcPosition, this.instance)
    }

    resize()
    {
        this.updateOrthographicFrustum()
        this.instance.updateProjectionMatrix()
        this.controls?.update()
    }

    update()
    {
        if (this.controls) 
            this.controls.update()
        
        if (this.raycaster) 
            this.raycaster.setFromCamera(this.mouse.ndcPosition, this.instance)
    }

    destroy() 
    {
        this.scene.remove(this.instance)

        if (this.orbit) 
        {
            this.orbit.dispose()
            this.orbit = null
        }

        if (this.raycaster)
        {
            this.raycaster = null
        }

        if (this.instance) 
        {
            this.controls?.removeEventListener('change', this.onControlsChange)
            this.controls?.dispose()
            this.instance = null
        }

        this.experience = null
        this.mouse = null
        this.sizes = null
        this.scene = null
        this.canvas = null
        this.controls = null

        console.log('Camera destroyed')
    }
}

