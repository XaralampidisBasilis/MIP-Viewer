# NIfTI Volume URL Loading Transfer Guide

This note explains the feature that lets this viewer load a NIfTI volume from a URL query parameter and makes it easier to move the same behavior into another similar project, especially an Angular host application.

## What the feature does

The viewer accepts a `volumeUrl` query parameter and uses it as the NIfTI file path to load.

Examples:

```text
/viewer?volumeUrl=nifti/cardiac/ct_train_1002_image.nii.gz
/viewer?volumeUrl=https%3A%2F%2Fmy-api.example.com%2Ffiles%2Fstudy.nii.gz
```

If `volumeUrl` is missing from the viewer URL, the code tries to read it from the parent page URL when the viewer is embedded in an iframe and the parent is same-origin. If neither exists, it falls back to a default local volume.

## Files involved

### `src/javascript/sources.js`

This is the entry point for selecting which volume will load.

Responsibilities:

- Reads `volumeUrl` from `window.location.search`
- If missing, tries `window.parent.location.search`
- Falls back to `defaultVolumePath`
- Returns the resources array with:

```js
{
    name: 'volume',
    type: 'niftiFile',
    path: resolvedVolumePath,
}
```

That `type: 'niftiFile'` is what connects the selected path to the custom NIfTI loader.

### `src/javascript/Experience.js`

This is where the selected source is actually passed into the resource system:

```js
this.resources = new Resources(createSources())
```

Flow:

1. `createSources()` resolves the path for the volume.
2. `Resources` starts loading it.
3. When loading finishes, `this.resources.on('ready', ...)` calls `start()`.
4. `start()` builds the compute/render pipeline and notifies the parent window:

```js
window.parent.postMessage({ type: 'experienceBuildCompleted', url: volumeUrl }, '*')
```

### `src/script.js`

This file handles viewer bootstrapping and reload behavior.

Responsibilities:

- Initializes TensorFlow and WebGL2
- Creates the first `Experience`
- Reads the current `volumeUrl`
- Posts:
  - `experienceBuildStarted`
  - `clearCache`
  - `experienceBuildCompleted` comes from `Experience.js`
- Watches for URL changes and recreates the viewer when `volumeUrl` changes

Important detail:

The viewer does a full `experience.destroy()` and then creates a new `Experience(...)` instead of trying to hot-swap the volume in place. That is a good fit here because the loaded volume affects GPU resources, compute textures, and viewer state.

## The actual loader path

The feature depends on two lower-level files too:

### `src/javascript/Utils/Resources.js`

This maps the resource type to the proper loader:

```js
else if(source.type === 'niftiFile')
{
    this.loaders.niftiLoader.load(source.path, ...)
}
```

### `src/javascript/Utils/NIFTILoader.js`

This is the custom loader that:

- fetches the `.nii` or `.nii.gz` file as an `arraybuffer`
- decompresses it if needed
- parses it with `nifti-reader-js`
- creates a `three/examples/jsm/misc/Volume`
- stores:
  - `data`
  - `dimensions`
  - `spacing`
  - `size`

So the full pipeline is:

```text
volumeUrl query param
-> createSources()
-> Resources
-> NIFTILoader
-> Volume object
-> compute/render pipeline
```

## How URL changes trigger a reload

`src/script.js` watches for URL changes using:

- patched `history.pushState`
- patched `history.replaceState`
- `popstate`
- `hashchange`
- a polling fallback with `setInterval`

When the URL changes, it dispatches a custom `urlchange` event. The handler then:

1. reads the new `volumeUrl`
2. compares it to the previous one
3. destroys the current `Experience`
4. recreates it so the new NIfTI file loads

This means the volume swap is driven by the viewer's own URL, not by internal Angular state by itself.

## Important Angular integration note

There are two separate cases:

### 1. Angular hosts the viewer in an iframe

This is the closest match to the current implementation.

Recommended approach:

- Set the iframe `src` to include `volumeUrl`
- When the selected volume changes, update the iframe URL so the viewer's own query string changes

Example:

```ts
const viewerBaseUrl = '/viewer';
const iframeUrl =
  `${viewerBaseUrl}?volumeUrl=${encodeURIComponent(volumeUrl)}`;
```

If you only change the Angular page URL and do not change the iframe URL, the viewer will not notice future volume changes. The fallback in `sources.js` can read the parent URL only during source resolution, and only when same-origin access is allowed.

### 2. Angular owns the viewer directly without an iframe

Then you can keep the same idea but read `volumeUrl` from the app route and either:

- update the browser query string and let the existing watcher recreate the viewer, or
- skip the watcher and call your own recreate function when the Angular route parameter changes

## Parent window message contract

The viewer sends these messages with `window.parent.postMessage(...)`:

- `experienceBuildStarted`
- `experienceBuildCompleted`
- `clearCache`

Payload shape:

```js
{ type: 'experienceBuildStarted', url: currentVolumeUrl }
{ type: 'experienceBuildCompleted', url: volumeUrl }
{ type: 'clearCache', url: currentVolumeUrl }
```

Angular can listen for them like this:

```ts
window.addEventListener('message', (event) => {
  const data = event.data;

  if (data?.type === 'experienceBuildStarted') {
    // show loading spinner
  }

  if (data?.type === 'experienceBuildCompleted') {
    // hide loading spinner
  }

  if (data?.type === 'clearCache') {
    // clear cached data for this volume if needed
  }
});
```

Note:

In the current `script.js`, `currentVolumeUrl` is updated before `clearCache` is posted, so that event currently carries the new URL, not the old one.

## Minimum pieces to copy into another project

If you want the same behavior elsewhere, these are the important parts:

1. A `createSources()` function that resolves `volumeUrl` and returns a `niftiFile` resource.
2. A resource loader that recognizes `type: 'niftiFile'`.
3. A NIfTI loader equivalent to `NIFTILoader.js`.
4. Viewer bootstrap code that can destroy and recreate the viewer when the selected URL changes.
5. Optional `postMessage` events if Angular or another parent app needs loading state.

## Recommended implementation order in the new project

1. Copy the `NIFTILoader` and the `Resources` mapping for `niftiFile`.
2. Add a `createSources()` function that resolves `volumeUrl`.
3. Make the main viewer constructor use `new Resources(createSources())`.
4. Add a small bootstrap layer that watches the viewer URL and recreates the viewer on changes.
5. If the viewer is embedded in Angular, pass `volumeUrl` through the iframe URL and listen for `postMessage` events.

## Practical caveats

- Remote NIfTI URLs must allow browser access with proper CORS headers.
- The parent URL fallback works only when the parent page is same-origin.
- The extra `createSources()` call in `src/script.js` is only used for logging; the real loading hook is inside `Experience.js`.
- If the new project has a cleaner state-management layer, you can replace the history monkey-patching with router-driven reload logic, as long as the viewer still rebuilds when `volumeUrl` changes.

## Short version

The core idea is simple:

```text
Angular or viewer route sets ?volumeUrl=...
-> viewer resolves that URL in createSources()
-> Resources loads it through NIFTILoader
-> Experience starts once resources are ready
-> if volumeUrl changes later, destroy and recreate the viewer
```
