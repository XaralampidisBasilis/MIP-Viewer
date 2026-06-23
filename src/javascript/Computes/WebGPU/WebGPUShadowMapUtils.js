export function getOctantSign( directionOctant, axis ) {
	if ( axis === 'x' ) return directionOctant[ 0 ]
	if ( axis === 'y' ) return directionOctant[ 1 ]
	return directionOctant[ 2 ]
}

export function reverseOctant( directionOctant ) {
	return [ ...directionOctant ].map( ( sign ) => sign === '+' ? '-' : '+' ).join( '' )
}

export function axisToShapeIndex( axis ) {
	if ( axis === 'x' ) return 0
	if ( axis === 'y' ) return 1
	return 2
}

export function axisDelta( axis, value ) {
	if ( axis === 'x' ) return [ value, 0, 0 ]
	if ( axis === 'y' ) return [ 0, value, 0 ]
	return [ 0, 0, value ]
}

export function planeSize( shape, axis ) {
	const [ width, height, depth ] = shape
	if ( axis === 'x' ) return [ height, depth ]
	if ( axis === 'y' ) return [ width, depth ]
	return [ width, height ]
}

export function planeCoordsWGSL( axis ) {
	if ( axis === 'x' ) return 'return vec3<i32>(params.slice, i32(gid.x), i32(gid.y));'
	if ( axis === 'y' ) return 'return vec3<i32>(i32(gid.x), params.slice, i32(gid.y));'
	return 'return vec3<i32>(i32(gid.x), i32(gid.y), params.slice);'
}

export function vec3( xyz ) {
	return `vec3<i32>(${xyz[ 0 ]}, ${xyz[ 1 ]}, ${xyz[ 2 ]})`
}

export function signedOffset( x, y, z, dominantAxis, directionOctant ) {
	return applyOctantSigns( canonicalOffset( x, y, z, dominantAxis ), directionOctant, 'signed' )
}

export function unitOffset( x, y, z, dominantAxis, directionOctant ) {
	return applyOctantSigns( canonicalOffset( x, y, z, dominantAxis ), directionOctant, 'unit' )
}

export function sliceOffset( x, y, z, dominantAxis, directionOctant ) {
	const offset = signedOffset( x, y, z, dominantAxis, directionOctant )
	offset[ axisToShapeIndex( dominantAxis ) ] = 0
	return offset
}

function canonicalOffset( x, y, z, dominantAxis ) {
	if ( dominantAxis === 'x' ) return [ z, y, x ]
	if ( dominantAxis === 'y' ) return [ y, z, x ]
	return [ x, y, z ]
}

function applyOctantSigns( xyz, directionOctant, mode ) {
	const offset = [ ...xyz ]

	for ( const axis of [ 'x', 'y', 'z' ] ) {
		const index = axisToShapeIndex( axis )

		if ( getOctantSign( directionOctant, axis ) !== '-' ) continue

		offset[ index ] = mode === 'unit' ? 1 - offset[ index ] : -offset[ index ]
	}

	return offset
}
