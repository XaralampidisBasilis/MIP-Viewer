import * as THREE from 'three'
import Experience from './Experience'
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js'
// import { TrackballControls } from './Utils/TrackballControls'
// import { FlyControls } from './Utils/FlyControls'
// import { ToggleControls } from './Utils/ToggleControls'

export default class Camera
{
    constructor()
    {
        this.experience = new Experience()
        this.mouse = this.experience.mouse
        this.sizes = this.experience.sizes
        this.scene = this.experience.scene
        this.canvas = this.experience.canvas
        this.time = this.experience.time
        
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
        this.controls?.dispose()

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
        const radius = Math.max(size.length() * 0.5, 1e-3)
        const distance = radius * 2.0
        const direction = new THREE.Vector3(1, 1, 1).normalize()

        this.orthographic.frustumHeight = radius * 2.4
        this.instance.near = 0.001
        this.instance.far = distance + radius * 4.0
        this.instance.position.copy(center).addScaledVector(direction, distance)

        this.controls.target.copy(center)
        this.controls.update()
        this.controls.saveState?.()
        this.resize()
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
        this.controls?.update()
        this.instance.updateProjectionMatrix()
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


