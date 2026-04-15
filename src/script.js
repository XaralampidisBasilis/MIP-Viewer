import { setTensorflow } from './javascript/tensorflow'
import Experience from './javascript/Experience'
import { getResolvedVolumeUrl } from './javascript/sources'

const URL_CHANGE_EVENT = 'urlchange'
const URL_POLL_INTERVAL_MS = 500

function postViewerMessage(type, url)
{
    window.parent?.postMessage({ type, url }, '*')
}

function dispatchUrlChange()
{
    window.dispatchEvent(new Event(URL_CHANGE_EVENT))
}

function watchUrlChanges()
{
    let lastHref = window.location.href

    const updateHrefAndDispatch = () =>
    {
        lastHref = window.location.href
        dispatchUrlChange()
    }

    const patchHistoryMethod = (methodName) =>
    {
        const originalMethod = window.history[methodName]

        if (typeof originalMethod !== 'function')
        {
            return () => {}
        }

        window.history[methodName] = function(...args)
        {
            const result = originalMethod.apply(this, args)
            updateHrefAndDispatch()
            return result
        }

        return () =>
        {
            window.history[methodName] = originalMethod
        }
    }

    const restorePushState = patchHistoryMethod('pushState')
    const restoreReplaceState = patchHistoryMethod('replaceState')
    const handleNavigationEvent = () => updateHrefAndDispatch()

    window.addEventListener('popstate', handleNavigationEvent)
    window.addEventListener('hashchange', handleNavigationEvent)

    const intervalId = window.setInterval(() =>
    {
        if (window.location.href !== lastHref)
        {
            updateHrefAndDispatch()
        }
    }, URL_POLL_INTERVAL_MS)

    return () =>
    {
        restorePushState()
        restoreReplaceState()
        window.removeEventListener('popstate', handleNavigationEvent)
        window.removeEventListener('hashchange', handleNavigationEvent)
        window.clearInterval(intervalId)
    }
}

(async () =>
{
    await setTensorflow()

    const canvas = document.querySelector('canvas.webgl')

    if (!canvas)
    {
        throw new Error('Viewer canvas ".webgl" was not found.')
    }

    const context = canvas.getContext('webgl2')

    if (!context)
    {
        throw new Error('WebGL2 not supported by your browser or device.')
    }

    let currentVolumeUrl = getResolvedVolumeUrl()

    const createExperience = () =>
    {
        postViewerMessage('experienceBuildStarted', currentVolumeUrl)
        return new Experience(canvas, context)
    }

    let experience = createExperience()

    const handleUrlChange = () =>
    {
        const nextVolumeUrl = getResolvedVolumeUrl()

        if (nextVolumeUrl === currentVolumeUrl)
        {
            return
        }

        currentVolumeUrl = nextVolumeUrl

        postViewerMessage('clearCache', currentVolumeUrl)
        experience?.destroy()
        experience = createExperience()
    }

    const stopWatchingUrl = watchUrlChanges()
    const handleBeforeUnload = () =>
    {
        stopWatchingUrl()
        window.removeEventListener(URL_CHANGE_EVENT, handleUrlChange)
        window.removeEventListener('beforeunload', handleBeforeUnload)
        experience?.destroy()
    }

    window.addEventListener(URL_CHANGE_EVENT, handleUrlChange)
    window.addEventListener('beforeunload', handleBeforeUnload)
})()
