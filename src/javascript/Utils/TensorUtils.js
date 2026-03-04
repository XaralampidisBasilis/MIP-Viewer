import * as tf from '@tensorflow/tfjs'

function makeCartesianAxes3d(shape, useRand = false)
{
    const [Z, Y, X] = shape
    const midZ = Math.floor(Z / 2)
    const midY = Math.floor(Y / 2)
    const midX = Math.floor(X / 2)

    return tf.tidy(() =>
    {
        const z = tf.range(0, Z, 1, 'int32').reshape([Z, 1, 1])
        const y = tf.range(0, Y, 1, 'int32').reshape([1, Y, 1])
        const x = tf.range(0, X, 1, 'int32').reshape([1, 1, X])

        // Axes (3 lines) through the center:
        // X-axis: y==midY and z==midZ (varies along x)
        const xAxis = y.equal(midY).logicalAnd(z.equal(midZ)).toFloat() // [Z,Y,X] via broadcast
        // Y-axis: x==midX and z==midZ (varies along y)
        const yAxis = x.equal(midX).logicalAnd(z.equal(midZ)).toFloat()
        // Z-axis: x==midX and y==midY (varies along z)
        const zAxis = x.equal(midX).logicalAnd(y.equal(midY)).toFloat()

        // Union of the three axes (1 on any axis, 0 elsewhere)
        const mask = tf.maximum(tf.maximum(xAxis, yAxis), zAxis)

        if (!useRand) return mask // ones on axes

        // Efficient: one random per voxel only where mask==1
        const rand = tf.randomUniform([Z, Y, X], 0, 1, 'float32')
        return rand.mul(mask) // random on axes, 0 elsewhere
    })
}

function makeCartesianPlanes3d(shape, useRand = false)
{
    const [Z, Y, X] = shape
    const midZ = Math.floor(Z / 2)
    const midY = Math.floor(Y / 2)
    const midX = Math.floor(X / 2)

    return tf.tidy(() =>
    {
        const z = tf.range(0, Z, 1, 'int32').reshape([Z, 1, 1])
        const y = tf.range(0, Y, 1, 'int32').reshape([1, Y, 1])
        const x = tf.range(0, X, 1, 'int32').reshape([1, 1, X])

        const zPlane = z.equal(midZ).toFloat() // [Z,1,1]
        const yPlane = y.equal(midY).toFloat() // [1,Y,1]
        const xPlane = x.equal(midX).toFloat() // [1,1,X]

        // Binary mask of the 3 orthogonal mid-planes (1 on planes, 0 elsewhere)
        const mask = tf.maximum(tf.maximum(zPlane, yPlane), xPlane) // [Z,Y,X] via broadcast

        if (!useRand) return mask // ones on planes

        const rand = tf.randomUniform([Z, Y, X], 0, 1, 'float32')
        return rand.mul(mask) // random on planes, 0 elsewhere
    })
}

function makeBoundaryPlanes3d(shape, useRand = false)
{
    const [Z, Y, X] = shape;

    return tf.tidy(() =>
    {
        const z = tf.range(0, Z, 1, 'int32').reshape([Z, 1, 1]);
        const y = tf.range(0, Y, 1, 'int32').reshape([1, Y, 1]);
        const x = tf.range(0, X, 1, 'int32').reshape([1, 1, X]);

        // Boundary planes on each axis
        const zBoundary = z.equal(0).logicalOr(z.equal(Z - 1));
        const yBoundary = y.equal(0).logicalOr(y.equal(Y - 1));
        const xBoundary = x.equal(0).logicalOr(x.equal(X - 1));

        // True wherever any boundary plane is true
        const maskBool = zBoundary.logicalOr(yBoundary).logicalOr(xBoundary);

        const mask = maskBool.toFloat(); // [Z,Y,X] via broadcast

        if (!useRand) return mask; // ones on boundary planes, 0 elsewhere

        const rand = tf.randomUniform([Z, Y, X], 0, 1, 'float32');
        return rand.mul(mask); // random on boundary planes, 0 elsewhere
    });
}

function makeCenterPoint3d(shape, useRand = false)
{
    const [Z, Y, X] = shape
    const midZ = Math.floor(Z / 2)
    const midY = Math.floor(Y / 2)
    const midX = Math.floor(X / 2)

    return tf.tidy(() =>
    {
        const z = tf.range(0, Z, 1, 'int32').reshape([Z, 1, 1])
        const y = tf.range(0, Y, 1, 'int32').reshape([1, Y, 1])
        const x = tf.range(0, X, 1, 'int32').reshape([1, 1, X])

        // Center point mask: true only where z==midZ AND y==midY AND x==midX
        const maskBool =
            z.equal(midZ)
             .logicalAnd(y.equal(midY))
             .logicalAnd(x.equal(midX))

        const mask = maskBool.toFloat() // [Z,Y,X] via broadcast

        if (!useRand) return mask // 1 at center voxel, 0 elsewhere

        // If you want a single random scalar at the center voxel:
        const randScalar = tf.randomUniform([1], 0, 1, 'float32')     // shape [1]
        const rand = randScalar.reshape([1, 1, 1]).broadcastTo([Z, Y, X])

        return rand.mul(mask) // random at center voxel, 0 elsewhere
    })
}

export {
    makeCartesianAxes3d,
    makeCartesianPlanes3d,
    makeBoundaryPlanes3d,
    makeCenterPoint3d
}