// ToggleControls.js
// Wraps ProbeControls + TrackballControls and switches with Tab without visual jumps.

import {
	Controls,
	Vector3
} from 'three';

import { ProbeControls } from './ProbeControls.js';
import { TrackballControls } from './TrackballControls.js';

/**
 * Fires when the camera has been transformed by the controls.
 *
 * @event ToggleControls#change
 * @type {Object}
 */
const _changeEvent = { type: 'change' };

/**
 * Fires when an interaction was initiated.
 *
 * @event ToggleControls#start
 * @type {Object}
 */
const _startEvent = { type: 'start' };

/**
 * Fires when an interaction has finished.
 *
 * @event ToggleControls#end
 * @type {Object}
 */
const _endEvent = { type: 'end' };

const _EPS = 0.000001;

// temps
const _forward = new Vector3();
const _upVec = new Vector3();
const _p = new Vector3();

/**
 * A wrapper that contains both {@link ProbeControls} and {@link TrackballControls}
 * and lets you toggle between them (default key: Tab) without visual jumps.
 *
 * @augments Controls
 * @three_import import { ToggleControls } from './ToggleControls.js';
 */
class ToggleControls extends Controls {

	/**
	 * Constructs a new controls instance.
	 *
	 * @param {Object3D} object - The object that is managed by the controls.
	 * @param {?HTMLDOMElement} domElement - The HTML element used for event listeners.
	 * @param {Object} [opts]
	 * @param {'probe'|'trackball'} [opts.start='probe'] - Initial mode.
	 * @param {boolean} [opts.interceptTab=true] - Listen for Tab to toggle.
	 * @param {boolean} [opts.onlyWhenPointerOver=false] - Only toggle when the pointer is over the element.
	 */
	constructor( object, domElement = null, opts = {} ) {

		super( object, domElement );

		const { start = 'probe', interceptTab = true, onlyWhenPointerOver = false } = opts;

		// children
		this.probe = new ProbeControls( object, domElement );
		this.trackball = new TrackballControls( object, domElement );

		// public-ish state
		this._useProbe = ( start === 'probe' );
		this._onlyWhenPointerOver = onlyWhenPointerOver;
		this._isPointerOver = false;

		// re-emit events from the active controller
		this._forwardEvent = onForwardEvent.bind( this );

		this.probe.addEventListener( 'start', this._forwardEvent );
		this.probe.addEventListener( 'change', this._forwardEvent );
		this.probe.addEventListener( 'end', this._forwardEvent );

		this.trackball.addEventListener( 'start', this._forwardEvent );
		this.trackball.addEventListener( 'change', this._forwardEvent );
		this.trackball.addEventListener( 'end', this._forwardEvent );

		// keyboard toggle
		this._onKeyDown = onKeyDown.bind( this );

		// hover tracking (optional)
		this._onEnter = onEnter.bind( this );
		this._onLeave = onLeave.bind( this );

		// options
		this._interceptTab = interceptTab;

		// ensure only active one is enabled
		this._applyEnabledMask();

		if ( domElement !== null ) {

			this.connect( domElement );
			this.handleResize();

		}

	}

	/**
	 * Current mode: 'probe' or 'trackball'
	 * @type {string}
	 */
	get mode() {

		return this._useProbe ? 'probe' : 'trackball';

	}

	/**
	 * The currently active controller instance.
	 * @type {ProbeControls|TrackballControls}
	 */
	get activeControls() {

		return this._useProbe ? this.probe : this.trackball;

	}

	connect( element ) {

		super.connect( element );

		// reconnect children
		if ( this.probe.domElement !== element ) {

			this.probe.disconnect?.();
			this.probe.connect?.( element );

		}

		if ( this.trackball.domElement !== element ) {

			this.trackball.disconnect?.();
			this.trackball.connect?.( element );

		}

		// key toggle
		if ( this._interceptTab ) window.addEventListener( 'keydown', this._onKeyDown );

		// hover listeners (for onlyWhenPointerOver)
		element?.addEventListener( 'mouseenter', this._onEnter );
		element?.addEventListener( 'mouseleave', this._onLeave );

		this.handleResize();

	}

	disconnect() {

		window.removeEventListener( 'keydown', this._onKeyDown );

		this.domElement?.removeEventListener( 'mouseenter', this._onEnter );
		this.domElement?.removeEventListener( 'mouseleave', this._onLeave );

		this.probe.disconnect?.();
		this.trackball.disconnect?.();

		super.disconnect?.();

	}

	dispose() {

		this.disconnect();

	}

	/**
	 * Must be called if the application window is resized.
	 */
	handleResize() {

		this.probe.handleResize?.();
		this.trackball.handleResize?.();

	}

	/**
	 * Per-frame update (delegates to active controls).
	 */
	update() {

		this.activeControls.update();

	}

	/**
	 * Toggle between controllers.
	 */
	toggle() {

		this.setActive( this._useProbe ? 'trackball' : 'probe' );

	}

	/**
	 * Programmatically set which controller is active.
	 * @param {'probe'|'trackball'} which
	 */
	setActive( which ) {

		if ( which === this.mode ) return;

		const cam = this.object;
		const from = this.activeControls;
		const to = ( which === 'probe' ) ? this.probe : this.trackball;

		// avoid switching mid-gesture
		if ( from.state !== undefined && from.state !== - 1 ) return;

		// world forward/up from camera orientation
		_forward.set( 0, 0, - 1 ).applyQuaternion( cam.quaternion ).normalize();
		_upVec.set( 0, 1, 0 ).applyQuaternion( cam.quaternion ).normalize();
		_p.copy( cam.position );

		// estimate radius from current controller's target if available
		let r = 1.0;

		if ( from.target ) {

			r = _p.clone().sub( from.target ).length();
			if ( ! isFinite( r ) || r < _EPS ) r = 1.0;

		}

		// reconcile conventions
		if ( to === this.trackball ) {

			// Trackball uses lookAt(target) with target IN FRONT; preserve roll by syncing up vector.
			cam.up.copy( _upVec );
			this.trackball.target.copy( _p ).addScaledVector( _forward, r );

			// neutralize momentum/damping residues
			this.trackball._lastAngle = 0;
			this.trackball._movePrev.copy( this.trackball._moveCurr );
			this.trackball._lastPosition.copy( _p );
			this.trackball._lastZoom = cam.zoom;

		} else {

			// Probe expects _eye aligned with forward and target BEHIND the camera:
			//   _eye = forward * r
			//   target = position - _eye
			const eye = _forward.clone().multiplyScalar( r );
			this.probe._eye.copy( eye );
			this.probe.target.copy( _p ).sub( eye );

			// sync caches so first update() is inert
			this.probe._lastPosition.copy( _p );
			this.probe._lastQuaternion.copy( cam.quaternion );
			this.probe._lastZoom = cam.zoom;

			// skip its one-time auto alignment
			this.probe._didAutoAlignForZoom = true;

			// calm gesture accumulators
			this.probe._zoomStart.copy( this.probe._zoomEnd );
			this.probe._panStart.copy( this.probe._panEnd );

		}

		// flip active controller and propagate enabled mask
		this._useProbe = ( which === 'probe' );
		this._applyEnabledMask();

		// single update to absorb state (no visible motion)
		to.update();

		this.dispatchEvent( _changeEvent );

	}

	// -------- internals --------

	_applyEnabledMask() {

		if ( ! this.probe || ! this.trackball ) return;
		this.probe.enabled = this.enabled && this._useProbe;
		this.trackball.enabled = this.enabled && ! this._useProbe;

	}

}

/* ----------------- free functions (bound in ctor) ----------------- */

function onForwardEvent( event ) {

	// re-emit only from the currently active child
	const isActive =
		( this._useProbe && event.target === this.probe ) ||
		( ! this._useProbe && event.target === this.trackball );

	if ( ! isActive ) return;

	// mirror the event type for external listeners
	if ( event.type === 'start' ) this.dispatchEvent( _startEvent );
	else if ( event.type === 'end' ) this.dispatchEvent( _endEvent );
	else this.dispatchEvent( _changeEvent );

}

function onKeyDown( event ) {

	if ( this.enabled === false ) return;

	if ( event.code !== 'Tab' ) return;

	if ( this._onlyWhenPointerOver && ! this._isPointerOver ) return;

	event.preventDefault();
	this.toggle();

}

function onEnter() {

	this._isPointerOver = true;

}

function onLeave() {

	this._isPointerOver = false;

}

export { ToggleControls };
