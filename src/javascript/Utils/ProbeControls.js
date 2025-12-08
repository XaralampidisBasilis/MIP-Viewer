import {
	Controls,
	MathUtils,
	MOUSE,
	Quaternion,
	Vector2,
	Vector3
} from 'three';

/**
 * Fires when the camera has been transformed by the controls.
 *
 * @event ProbeControls#change
 * @type {Object}
 */
const _changeEvent = { type: 'change' };

/**
 * Fires when an interaction was initiated.
 *
 * @event ProbeControls#start
 * @type {Object}
 */
const _startEvent = { type: 'start' };

/**
 * Fires when an interaction has finished.
 *
 * @event ProbeControls#end
 * @type {Object}
 */
const _endEvent = { type: 'end' };

const _EPS = 0.000001;
const _STATE = { NONE: - 1, FPS_LOOK: 0, ZOOM: 1, PAN: 2, TOUCH_LOOK: 3, TOUCH_ZOOM_PAN: 4 };

// temporary vectors/quaternions (internals)
const _v2 = new Vector2();
const _mouseChange = new Vector2();
const _pan = new Vector3();
const _tmpV = new Vector3();
const _right = new Vector3();
const _upWorld = new Vector3();
const _fwd = new Vector3();
const _tmpQ = new Quaternion();

/**
 * A hybrid control:
 * - FPS-style keyboard translation and mouse/touch look (yaw/pitch/roll)
 * - Trackball-like pan/zoom behavior
 *
 * No functional changes from the original implementation; only style/formatting and documentation.
 *
 * @augments Controls
 * @three_import // (local/custom) import { ProbeControls } from './ProbeControls.js';
 */
class ProbeControls extends Controls {

	/**
	 * @param {Object3D} object - The object (camera) managed by the controls.
	 * @param {?HTMLElement} domElement - The element used for event listeners.
	 */
	constructor( object, domElement = null ) {

		super( object, domElement );

		// ---------------- Public API ----------------

		/**
		 * Movement (FPS-style translation) base speed.
		 * @type {number}
		 * @default 0.1
		 */
		this.movementSpeed = 0.1;

		/**
		 * Multiplier applied on top of speed (changed via keyboard; SHIFT/SPACE).
		 * @type {number}
		 * @default 1
		 */
		this.speedMultiplier = 1.0;

		/**
		 * Rotation speed for keyboard-based view rotation (yaw/pitch/roll).
		 * @type {number}
		 * @default 0.8
		 */
		this.rollSpeed = 0.8;

		/**
		 * If true, "forward" is applied when not pressing "back".
		 * @type {boolean}
		 * @default false
		 */
		this.autoForward = false;

		/**
		 * Mouse look sensitivity (yaw/pitch) in radians per pixel.
		 * @type {number}
		 * @default 0.002
		 */
		this.lookSpeed = 0.002;

		/**
		 * Whether zooming is disabled or not.
		 * @type {boolean}
		 * @default true
		 */
		this.noZoom = true;

		/**
		 * Whether panning is disabled or not.
		 * @type {boolean}
		 * @default false
		 */
		this.noPan = false;

		/**
		 * The zoom speed (drag or wheel).
		 * @type {number}
		 * @default 1
		 */
		this.zoomSpeed = 1.0;

		/**
		 * The pan speed (screen-space).
		 * @type {number}
		 * @default 0.05
		 */
		this.panSpeed = 0.05;

		/**
		 * Minimum eye distance (perspective camera only).
		 * @type {number}
		 * @default 0
		 */
		this.minDistance = 0;

		/**
		 * Maximum eye distance (perspective camera only).
		 * @type {number}
		 * @default Infinity
		 */
		this.maxDistance = Infinity;

		/**
		 * If true, interactions apply immediately without damping.
		 * @type {boolean}
		 * @default true
		 */
		this.staticMoving = true;

		/**
		 * Damping factor used when staticMoving is false.
		 * @type {number}
		 * @default 0.3
		 */
		this.dynamicDampingFactor = 0.3;

		/**
		 * Flip only the middle-button drag zoom direction.
		 * @type {boolean}
		 * @default true
		 */
		this.invertMiddleDragZoom = true;

		/**
		 * If you want the keyboard controls enabled.
		 * @type {boolean}
		 * @default true
		 */
		this.enableKeyboard = true;

		/**
		 * Mouse bindings (consistent with other controls).
		 * @type {{LEFT:number,MIDDLE:number,RIGHT:number}}
		 * @default { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN }
		 */
		this.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };

		/**
		 * The focus point of the controls.
		 * @type {Vector3}
		 */
		this.target = new Vector3();

		// ---------------- Internals ----------------

		/** @type {Vector3} */
		this._eye = new Vector3();

		/** @type {number} */
		this.state = _STATE.NONE;

		/** @type {Array<PointerEvent>} */
		this._pointers = [];

		/** @type {Record<number,Vector2>} */
		this._pointerPositions = {};

		this._moveState = {
			up: 0, down: 0, left: 0, right: 0, forward: 0, back: 0,
			pitchUp: 0, pitchDown: 0, yawLeft: 0, yawRight: 0, rollLeft: 0, rollRight: 0
		};

		this._fpsLast = new Vector2();

		this._zoomStart = new Vector2();
		this._zoomEnd = new Vector2();
		this._panStart = new Vector2();
		this._panEnd = new Vector2();

		this._touchZoomDistanceStart = 0;
		this._touchZoomDistanceEnd = 0;

		this._lastPosition = new Vector3().copy( this.object.position );
		this._lastQuaternion = new Quaternion().copy( this.object.quaternion );
		this._lastZoom = this.object.zoom || 1;

		this._lastUpdateTime = ( typeof performance !== 'undefined' ? performance.now() : Date.now() );

		/**
		 * Represents the properties of the screen. Automatically set when `handleResize()` is called.
		 * @type {{left:number,top:number,width:number,height:number}}
		 * @readonly
		 */
		this.screen = { left: 0, top: 0, width: 0, height: 0 };

		// --- FIX: lazily realign the very first zoom to the current view direction ---
		// We keep this flag so alignment runs only once and only when needed.
		this._didAutoAlignForZoom = false;

		// ---------------- Event listeners (bound) ----------------

		this._onKeyDown = onKeyDown.bind( this );
		this._onKeyUp = onKeyUp.bind( this );
		this._onWindowBlur = onWindowBlur.bind( this );

		this._onPointerDown = onPointerDown.bind( this );
		this._onPointerMove = onPointerMove.bind( this );
		this._onPointerUp = onPointerUp.bind( this );
		this._onPointerCancel = onPointerCancel.bind( this );
		this._onMouseWheel = onMouseWheel.bind( this );
		this._onContextMenu = onContextMenu.bind( this );

		this._onMouseDown = onMouseDown.bind( this );
		this._onMouseMove = onMouseMove.bind( this );
		this._onMouseUp = onMouseUp.bind( this );

		this._onTouchStart = onTouchStart.bind( this );
		this._onTouchMove = onTouchMove.bind( this );
		this._onTouchEnd = onTouchEnd.bind( this );

		if ( domElement !== null ) {

			this.connect( domElement );
			this.handleResize();

		}

		// Capture initial eye from current transform without moving the object
		this._eye.subVectors( this.object.position, this.target );

		// No initial update() positional rewrite to avoid tiny nudges

	}

	// ---------------- Lifecycle ----------------

	/**
	 * Connect event listeners to a DOM element.
	 * @param {HTMLElement} element
	 */
	connect( element ) {

		super.connect( element );

		if ( this.enableKeyboard ) {

			window.addEventListener( 'keydown', this._onKeyDown );
			window.addEventListener( 'keyup', this._onKeyUp );
			window.addEventListener( 'blur', this._onWindowBlur );

		}

		this.domElement.addEventListener( 'pointerdown', this._onPointerDown );
		this.domElement.addEventListener( 'pointercancel', this._onPointerCancel );
		this.domElement.addEventListener( 'wheel', this._onMouseWheel, { passive: false } );
		this.domElement.addEventListener( 'contextmenu', this._onContextMenu );

		this.domElement.style.touchAction = 'none'; // disable touch scroll

	}

	/**
	 * Disconnect all event listeners.
	 */
	disconnect() {

		window.removeEventListener( 'keydown', this._onKeyDown );
		window.removeEventListener( 'keyup', this._onKeyUp );
		window.removeEventListener( 'blur', this._onWindowBlur );

		this.domElement.removeEventListener( 'pointerdown', this._onPointerDown );
		this.domElement.removeEventListener( 'pointermove', this._onPointerMove );
		this.domElement.removeEventListener( 'pointerup', this._onPointerUp );
		this.domElement.removeEventListener( 'pointercancel', this._onPointerCancel );
		this.domElement.removeEventListener( 'wheel', this._onMouseWheel );
		this.domElement.removeEventListener( 'contextmenu', this._onContextMenu );

		this.domElement.style.touchAction = 'auto'; // re-enable touch scroll

	}

	/**
	 * Dispose of the controls (alias of disconnect).
	 */
	dispose() {

		this.disconnect();

	}

	/**
	 * Must be called if the application window is resized.
	 */
	handleResize() {

		const box = this.domElement.getBoundingClientRect();
		const d = this.domElement.ownerDocument.documentElement;

		this.screen.left = box.left + window.pageXOffset - d.clientLeft;
		this.screen.top = box.top + window.pageYOffset - d.clientTop;
		this.screen.width = box.width;
		this.screen.height = box.height;

	}

	/**
	 * Per-frame update. Applies keyboard-driven translation/rotation and emits change events
	 * when the camera's transform has changed.
	 */
	update() {

		// keep _eye in sync with external camera changes
		_syncEyeFromObject.call( this );

		const now = ( typeof performance !== 'undefined' ? performance.now() : Date.now() );
		let dt = ( now - this._lastUpdateTime ) / 1000;
		if ( dt <= 0 ) dt = 0.016;
		this._lastUpdateTime = now;

		_applyKeyboardTranslation.call( this, dt );
		_applyKeyboardViewRotation.call( this, dt );

		// NOTE: We deliberately DO NOT recompute position = target + eye here,
		// to avoid tiny initial drift. Zoom handler applies it when _eye changes.

		const quatChanged = ( 8 * ( 1 - this._lastQuaternion.dot( this.object.quaternion ) ) > _EPS );
		const posChanged = ( this._lastPosition.distanceToSquared( this.object.position ) > _EPS );

		if ( this.object.isOrthographicCamera ) {

			if ( posChanged || quatChanged || this._lastZoom !== this.object.zoom ) {

				this.dispatchEvent( _changeEvent );
				this._lastPosition.copy( this.object.position );
				this._lastQuaternion.copy( this.object.quaternion );
				this._lastZoom = this.object.zoom;

			}

		} else {

			if ( posChanged || quatChanged ) {

				this.dispatchEvent( _changeEvent );
				this._lastPosition.copy( this.object.position );
				this._lastQuaternion.copy( this.object.quaternion );

			}

		}

	}

	/**
	 * Resets the controls to a neutral origin state.
	 */
	reset() {

		this.state = _STATE.NONE;

		this.target.set( 0, 0, 0 );
		this.object.position.set( 0, 0, 0 );
		this.object.quaternion.identity();
		this.object.up.set( 0, 1, 0 );

		this.object.updateProjectionMatrix?.();

		this._eye.subVectors( this.object.position, this.target );
		this._lastPosition.copy( this.object.position );
		this._lastQuaternion.copy( this.object.quaternion );
		this._lastZoom = this.object.zoom || 1;

		this._didAutoAlignForZoom = false;

		this.dispatchEvent( _changeEvent );

	}

	/**
	 * Map a page position to normalized screen coordinates.
	 * @param {number} pageX
	 * @param {number} pageY
	 * @returns {Vector2}
	 */
	_getMouseOnScreen( pageX, pageY ) {

		// --- FIX: be robust when called before the element is laid out (width/height 0) ---
		if ( this.screen.width === 0 || this.screen.height === 0 ) {

			this.handleResize();

			if ( this.screen.width === 0 || this.screen.height === 0 ) {

				// fallback to client size if layout info is still unavailable
				const w = this.domElement.clientWidth || this.domElement.offsetWidth || 1;
				const h = this.domElement.clientHeight || this.domElement.offsetHeight || 1;

				// Do not change left/top in this fallback; only prevent division by zero
				this.screen.width = w;
				this.screen.height = h;

			}

		}

		_v2.set(
			( pageX - this.screen.left ) / this.screen.width,
			( pageY - this.screen.top ) / this.screen.height
		);

		return _v2;

	}

	// ----- internal pointer helpers (kept as methods for symmetry with TrackballControls) -----

	_addPointer( event ) {

		this._pointers.push( event );

	}

	_removePointer( event ) {

		delete this._pointerPositions[ event.pointerId ];

		for ( let i = 0; i < this._pointers.length; i ++ ) {

			if ( this._pointers[ i ].pointerId == event.pointerId ) {

				this._pointers.splice( i, 1 );
				return;

			}

		}

	}

	_trackPointer( event ) {

		let position = this._pointerPositions[ event.pointerId ];

		if ( position === undefined ) {

			position = new Vector2();
			this._pointerPositions[ event.pointerId ] = position;

		}

		position.set( event.pageX, event.pageY );

	}

	_getSecondPointerPosition( event ) {

		const pointer = ( event.pointerId === this._pointers[ 0 ].pointerId ) ? this._pointers[ 1 ] : this._pointers[ 0 ];
		return this._pointerPositions[ pointer.pointerId ];

	}

}

// ---------------- helpers (module-private) ----------------

/** Keep _eye consistent with external camera changes without moving the camera. */
function _syncEyeFromObject() {

	this._eye.subVectors( this.object.position, this.target );

}

// --- FIX: ensure the first zoom uses the current camera forward vector ---
function _autoAlignEyeWithForwardIfNeeded() {

	if ( this._didAutoAlignForZoom ) return;

	const r = this._eye.length();
	if ( r === 0 ) return;

	// world-space forward (-Z in camera space)
	_fwd.set( 0, 0, - 1 ).applyQuaternion( this.object.quaternion ).normalize();

	// check alignment between eye vector and forward (implementation expects them to be parallel)
	_tmpV.copy( this._eye ).normalize();
	const dot = _fwd.dot( _tmpV );

	if ( dot < 0.999 ) {

		// realign _eye to forward, preserve radius; update target to stay consistent
		this._eye.copy( _fwd ).multiplyScalar( r );
		this.target.copy( this.object.position ).sub( this._eye );

	}

	this._didAutoAlignForZoom = true;

}

// ---------------- Quaternion-based view rotation ----------------

function _applyViewAngles( yaw, pitch, roll ) {

	_upWorld.set( 0, 1, 0 ).applyQuaternion( this.object.quaternion ).normalize();

	if ( yaw ) {

		_tmpQ.setFromAxisAngle( _upWorld, yaw );
		this.object.quaternion.premultiply( _tmpQ );

	}

	_fwd.set( 0, 0, - 1 ).applyQuaternion( this.object.quaternion ).normalize();
	_right.crossVectors( _fwd, _upWorld ).normalize();

	if ( pitch ) {

		_tmpQ.setFromAxisAngle( _right, pitch );
		this.object.quaternion.premultiply( _tmpQ );

	}

	_fwd.set( 0, 0, - 1 ).applyQuaternion( this.object.quaternion ).normalize();

	if ( roll ) {

		_tmpQ.setFromAxisAngle( _fwd, roll );
		this.object.quaternion.premultiply( _tmpQ );

	}

	// keep eye/target consistent in orientation (but DON'T rewrite position)
	const r = this._eye.length() || 1.0;
	this._eye.copy( _fwd ).multiplyScalar( r );
	this.target.copy( this.object.position ).sub( this._eye );

}

// ---------------- Keyboard translation/rotation ----------------

function _applyKeyboardTranslation( dt ) {

	const forward = ( this._moveState.forward || ( this.autoForward && ! this._moveState.back ) ) ? 1 : 0;

	_tmpV.set(
		- this._moveState.left + this._moveState.right,
		- this._moveState.down + this._moveState.up,
		- forward + this._moveState.back
	);

	if ( _tmpV.lengthSq() ) {

		const moveMult = dt * this.movementSpeed * ( this.speedMultiplier || 1 );
		_tmpV.multiplyScalar( moveMult ).applyQuaternion( this.object.quaternion );

		this.object.position.add( _tmpV );
		this.target.add( _tmpV );

		this._eye.subVectors( this.object.position, this.target );

	}

}

function _applyKeyboardViewRotation( dt ) {
    
	const yaw = ( this._moveState.yawLeft - this._moveState.yawRight ) * dt * this.rollSpeed;
	const pitch = ( this._moveState.pitchUp - this._moveState.pitchDown ) * dt * this.rollSpeed;
	const roll = ( this._moveState.rollLeft - this._moveState.rollRight ) * dt * this.rollSpeed;

	if ( yaw || pitch || roll ) {

		_applyViewAngles.call( this, yaw, pitch, roll );

	}

}

// ---------------- Mouse look / Pan / Zoom ----------------

function _rotateCameraFPSByMouse( dx, dy ) {

	const yaw = - dx * this.lookSpeed;
	const pitch = - dy * this.lookSpeed;

	_applyViewAngles.call( this, yaw, pitch, 0 );

	this.dispatchEvent( _changeEvent );

}

function _applyPanEasing() {

	if ( this.staticMoving ) {

		this._panStart.copy( this._panEnd );

	} else {

		this._panStart.add( _mouseChange.subVectors( this._panEnd, this._panStart ).multiplyScalar( this.dynamicDampingFactor ) );

	}

}

function _panCamera() {

	_mouseChange.copy( this._panEnd ).sub( this._panStart );

	if ( _mouseChange.lengthSq() ) {

		const scale = this._eye.length() * this.panSpeed * this.speedMultiplier * ( this.noPan ? 0 : 1 );

		_mouseChange.multiplyScalar( scale );

		_fwd.set( 0, 0, - 1 ).applyQuaternion( this.object.quaternion ).normalize();
		_upWorld.set( 0, 1, 0 ).applyQuaternion( this.object.quaternion ).normalize();
		_right.crossVectors( _fwd, _upWorld ).normalize();

		// Invert X to match TrackballControls
		_pan.copy( _right ).multiplyScalar( - _mouseChange.x );
		_pan.addScaledVector( _upWorld, _mouseChange.y );

		this.object.position.add( _pan );
		this.target.add( _pan );

		_applyPanEasing.call( this );

	}

}

function _clampEyeDistance() {

	if ( ! this.object.isPerspectiveCamera ) return;

	const len = this._eye.length();
	const clamped = MathUtils.clamp( len, this.minDistance, this.maxDistance );

	if ( clamped !== len && len > 0 ) this._eye.multiplyScalar( clamped / len );

}

function _applyEyeToPosition() {

	// Helper: sync world position with eye change
	this.object.position.addVectors( this.target, this._eye );

}

function _zoomCamera( isMiddleDrag ) {

	let factor;

	if ( this.state === _STATE.TOUCH_ZOOM_PAN ) {

		factor = this._touchZoomDistanceStart / this._touchZoomDistanceEnd;
		this._touchZoomDistanceStart = this._touchZoomDistanceEnd;

		if ( this.object.isPerspectiveCamera ) {

			this._eye.multiplyScalar( factor );
			_clampEyeDistance.call( this );
			_applyEyeToPosition.call( this );

		} else if ( this.object.isOrthographicCamera ) {

			this.object.zoom = MathUtils.clamp( this.object.zoom / factor, 0, Infinity );
			if ( this._lastZoom !== this.object.zoom ) this.object.updateProjectionMatrix();

		}

	} else {

		const invert = ( isMiddleDrag && this.invertMiddleDragZoom ) ? - 1 : 1;
		factor = 1.0 + invert * ( this._zoomEnd.y - this._zoomStart.y ) * ( this.zoomSpeed * this.speedMultiplier );

		if ( factor !== 1.0 && factor > 0.0 ) {

			if ( this.object.isPerspectiveCamera ) {

				this._eye.multiplyScalar( factor );
				_clampEyeDistance.call( this );
				_applyEyeToPosition.call( this );

			} else if ( this.object.isOrthographicCamera ) {

				this.object.zoom = MathUtils.clamp( this.object.zoom / factor, 0, Infinity );
				if ( this._lastZoom !== this.object.zoom ) this.object.updateProjectionMatrix();

			}

		}

		if ( this.staticMoving ) {

			this._zoomStart.copy( this._zoomEnd );

		} else {

			this._zoomStart.y += ( this._zoomEnd.y - this._zoomStart.y ) * this.dynamicDampingFactor;

		}

	}

}

// ---------------- Event handlers ----------------

function onPointerDown( event ) {

	if ( this.enabled === false ) return;

	if ( this._pointers.length === 0 ) {

		this.domElement.setPointerCapture( event.pointerId );
		this.domElement.addEventListener( 'pointermove', this._onPointerMove );
		this.domElement.addEventListener( 'pointerup', this._onPointerUp );

	}

	this._addPointer( event );

	if ( event.pointerType === 'touch' ) this._onTouchStart( event );
	else this._onMouseDown( event );

}

function onPointerMove( event ) {

	if ( this.enabled === false ) return;

	if ( event.pointerType === 'touch' ) this._onTouchMove( event );
	else this._onMouseMove( event );

}

function onPointerUp( event ) {

	if ( this.enabled === false ) return;

	if ( event.pointerType === 'touch' ) this._onTouchEnd( event );
	else this._onMouseUp();

	this._removePointer( event );

	if ( this._pointers.length === 0 ) {

		this.domElement.releasePointerCapture( event.pointerId );
		this.domElement.removeEventListener( 'pointermove', this._onPointerMove );
		this.domElement.removeEventListener( 'pointerup', this._onPointerUp );

	}

}

function onPointerCancel( event ) {

	this._removePointer( event );

}

function onMouseDown( event ) {

	// ensure _eye matches camera before we compute zoom/pan scales
	_syncEyeFromObject.call( this );

	let mouseAction;

	switch ( event.button ) {

		case 0: mouseAction = this.mouseButtons.LEFT; break;
		case 1: mouseAction = this.mouseButtons.MIDDLE; break;
		case 2: mouseAction = this.mouseButtons.RIGHT; break;
		default: mouseAction = - 1;

	}

	switch ( mouseAction ) {

		case MOUSE.DOLLY:  this.state = this.noZoom ? _STATE.NONE : _STATE.ZOOM; break;
		case MOUSE.ROTATE: this.state = _STATE.FPS_LOOK; break;
		case MOUSE.PAN:    this.state = this.noPan ? _STATE.NONE : _STATE.PAN; break;
		default:           this.state = _STATE.NONE;

	}

	// --- FIX: when the very first user action is a zoom-drag, align to forward & make sure screen rect is valid
	if ( this.state === _STATE.ZOOM ) {

		_autoAlignEyeWithForwardIfNeeded.call( this );
		if ( this.screen.width === 0 || this.screen.height === 0 ) this.handleResize();

	}

	if ( this.state === _STATE.ZOOM ) {

		this._zoomStart.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );
		this._zoomEnd.copy( this._zoomStart );

	} else if ( this.state === _STATE.PAN ) {

		this._panStart.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );
		this._panEnd.copy( this._panStart );

	} else if ( this.state === _STATE.FPS_LOOK ) {

		this._fpsLast.set( event.pageX, event.pageY );

	}

	this.dispatchEvent( _startEvent );

}

function onMouseMove( event ) {

	if ( this.state === _STATE.ZOOM && ! this.noZoom ) {

		this._zoomEnd.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );
		_zoomCamera.call( this, /*isMiddleDrag*/ true );
		this.dispatchEvent( _changeEvent );

	} else if ( this.state === _STATE.PAN && ! this.noPan ) {

		this._panEnd.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );
		_panCamera.call( this );
		this.dispatchEvent( _changeEvent );

	} else if ( this.state === _STATE.FPS_LOOK ) {

		const dx = event.pageX - this._fpsLast.x;
		const dy = event.pageY - this._fpsLast.y;

		this._fpsLast.set( event.pageX, event.pageY );

		if ( dx || dy ) _rotateCameraFPSByMouse.call( this, dx, dy );

	}

}

function onMouseUp() {

	this.state = _STATE.NONE;
	this.dispatchEvent( _endEvent );

}

function onMouseWheel( event ) {

	if ( this.enabled === false || this.noZoom === true ) return;

	// make sure _eye is fresh so first wheel does not snap
	_syncEyeFromObject.call( this );

	// --- FIX: if the very first interaction is the wheel, align zoom to the current view forward ---
	_autoAlignEyeWithForwardIfNeeded.call( this );

	event.preventDefault();

	let k;
	switch ( event.deltaMode ) {

		case 2: k = 0.025; break; // page
		case 1: k = 0.01; break;  // line
		default: k = 0.00025; break; // pixel

	}

	const factor = 1.0 + ( - event.deltaY * k ) * ( this.zoomSpeed * this.speedMultiplier );

	if ( factor !== 1.0 && factor > 0.0 ) {

		if ( this.object.isPerspectiveCamera ) {

			this._eye.multiplyScalar( factor );
			_clampEyeDistance.call( this );
			_applyEyeToPosition.call( this );

		} else if ( this.object.isOrthographicCamera ) {

			this.object.zoom = MathUtils.clamp( this.object.zoom / factor, 0, Infinity );
			if ( this._lastZoom !== this.object.zoom ) this.object.updateProjectionMatrix();

		}

		this.dispatchEvent( _startEvent );
		this.dispatchEvent( _endEvent );
		this.dispatchEvent( _changeEvent );

	}

}

function onContextMenu( event ) {

	if ( this.enabled === false ) return;
	event.preventDefault();

}

function onTouchStart( event ) {

	// keep _eye fresh at the beginning of a gesture
	_syncEyeFromObject.call( this );

	this._trackPointer( event );

	switch ( this._pointers.length ) {

		case 1:
			this.state = _STATE.TOUCH_LOOK;
			this._fpsLast.set( this._pointers[ 0 ].pageX, this._pointers[ 0 ].pageY );
			break;

		default:
			this.state = _STATE.TOUCH_ZOOM_PAN;

			// --- FIX: first interaction could be a pinch; align zoom direction to camera forward ---
			_autoAlignEyeWithForwardIfNeeded.call( this );

			{
				const dx = this._pointers[ 0 ].pageX - this._pointers[ 1 ].pageX;
				const dy = this._pointers[ 0 ].pageY - this._pointers[ 1 ].pageY;

				this._touchZoomDistanceEnd = this._touchZoomDistanceStart = Math.sqrt( dx * dx + dy * dy );

				const x = ( this._pointers[ 0 ].pageX + this._pointers[ 1 ].pageX ) / 2;
				const y = ( this._pointers[ 0 ].pageY + this._pointers[ 1 ].pageY ) / 2;

				this._panStart.copy( this._getMouseOnScreen( x, y ) );
				this._panEnd.copy( this._panStart );
			}

	}

	this.dispatchEvent( _startEvent );

}

function onTouchMove( event ) {

	this._trackPointer( event );

	switch ( this._pointers.length ) {

		case 1: {

			const dx = event.pageX - this._fpsLast.x;
			const dy = event.pageY - this._fpsLast.y;

			this._fpsLast.set( event.pageX, event.pageY );

			if ( dx || dy ) _rotateCameraFPSByMouse.call( this, dx, dy );
			break;

		}

		default: {

			const pos = this._getSecondPointerPosition( event );

			const dx = event.pageX - pos.x;
			const dy = event.pageY - pos.y;

			this._touchZoomDistanceEnd = Math.sqrt( dx * dx + dy * dy );

			const x = ( event.pageX + pos.x ) / 2;
			const y = ( event.pageY + pos.y ) / 2;

			this._panEnd.copy( this._getMouseOnScreen( x, y ) );

			_zoomCamera.call( this, /*isMiddleDrag*/ false );
			_panCamera.call( this );

		}

	}

}

function onTouchEnd( event ) {

	switch ( this._pointers.length ) {

		case 0:
			this.state = _STATE.NONE;
			break;

		case 1:
			this.state = _STATE.TOUCH_LOOK;
			this._fpsLast.set( event.pageX, event.pageY );
			break;

		case 2:
			this.state = _STATE.TOUCH_ZOOM_PAN;

			for ( let i = 0; i < this._pointers.length; i ++ ) {

				if ( this._pointers[ i ].pointerId !== event.pointerId ) {

					const position = this._pointerPositions[ this._pointers[ i ].pointerId ];
					this._fpsLast.set( position.x, position.y );
					break;

				}

			}
			break;

	}

	this.dispatchEvent( _endEvent );

}

// ---------------- Keyboard handlers ----------------

function onKeyDown( event ) {

	if ( ! this.enabled || ! this.enableKeyboard ) return;

	let used = false;

	switch ( event.code ) {

		case 'ShiftLeft': this.speedMultiplier = 0.2; used = true; break;
		case 'Space':     this.speedMultiplier = 2.0; used = true; break;

		case 'KeyW': this._moveState.forward = 1; used = true; break;
		case 'KeyS': this._moveState.back    = 1; used = true; break;
		case 'KeyA': this._moveState.left    = 1; used = true; break;
		case 'KeyD': this._moveState.right   = 1; used = true; break;
		case 'KeyR': this._moveState.up      = 1; used = true; break;
		case 'KeyF': this._moveState.down    = 1; used = true; break;

		case 'ArrowUp':    this._moveState.pitchUp   = 1; used = true; break;
		case 'ArrowDown':  this._moveState.pitchDown = 1; used = true; break;
		case 'ArrowLeft':  this._moveState.yawLeft   = 1; used = true; break;
		case 'ArrowRight': this._moveState.yawRight  = 1; used = true; break;

		case 'KeyQ': this._moveState.rollLeft  = 1; used = true; break;
		case 'KeyE': this._moveState.rollRight = 1; used = true; break;

	}

	if ( used && event.code.startsWith( 'Arrow' ) ) event.preventDefault();

}

function onKeyUp( event ) {

	if ( ! this.enableKeyboard ) return;

	switch ( event.code ) {

		case 'ShiftLeft': this.speedMultiplier = 1.0; break;
		case 'Space':     this.speedMultiplier = 1.0; break;

		case 'KeyW': this._moveState.forward = 0; break;
		case 'KeyS': this._moveState.back    = 0; break;
		case 'KeyA': this._moveState.left    = 0; break;
		case 'KeyD': this._moveState.right   = 0; break;
		case 'KeyR': this._moveState.up      = 0; break;
		case 'KeyF': this._moveState.down    = 0; break;

		case 'ArrowUp':    this._moveState.pitchUp   = 0; break;
		case 'ArrowDown':  this._moveState.pitchDown = 0; break;
		case 'ArrowLeft':  this._moveState.yawLeft   = 0; break;
		case 'ArrowRight': this._moveState.yawRight  = 0; break;

		case 'KeyQ': this._moveState.rollLeft  = 0; break;
		case 'KeyE': this._moveState.rollRight = 0; break;

	}

}

function onWindowBlur() {

	this._moveState.up = this._moveState.down = this._moveState.left = this._moveState.right = 0;
	this._moveState.forward = this._moveState.back = 0;

	this._moveState.pitchUp = this._moveState.pitchDown = 0;
	this._moveState.yawLeft = this._moveState.yawRight = 0;
	this._moveState.rollLeft = this._moveState.rollRight = 0;

	this.speedMultiplier = 1.0;

}

export { ProbeControls };
