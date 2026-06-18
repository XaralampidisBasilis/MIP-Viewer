import { computeBidirectionalShadowMapPathsWebGPU } from './WebGPUShadowMapPaths'

export async function computeBidirectionalShadowMapFacesWebGPU(
	volume,
	dominantAxis,
	directionOctant,
	tolerance,
	blockSize,
	verbose = false,
) {
	// TODO: replace with the face-stencil WGSL port. This preserves the backend
	// option and output ABI while keeping the working conservative path.
	return computeBidirectionalShadowMapPathsWebGPU(
		volume,
		dominantAxis,
		directionOctant,
		tolerance,
		blockSize,
		verbose,
	)
}
