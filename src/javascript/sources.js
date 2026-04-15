export const volumeUrlQueryParam = 'volumeUrl'
export const defaultVolumePath = 'nifti/head/CTA-Head-and-Neck.nii.gz'
// export const defaultVolumePath = 'nifti/cardiac/ct_train_1002_image.nii.gz'

function getVolumeUrlFromSearch(search = '')
{
    const volumeUrl = new URLSearchParams(search).get(volumeUrlQueryParam)

    if (!volumeUrl)
    {
        return null
    }

    const normalizedVolumeUrl = volumeUrl.trim()

    return normalizedVolumeUrl.length > 0 ? normalizedVolumeUrl : null
}

function getParentSearch()
{
    if (typeof window === 'undefined' || window.parent === window)
    {
        return ''
    }

    try
    {
        if (window.parent.location.origin === window.location.origin)
        {
            return window.parent.location.search
        }
    }
    catch (error)
    {
        // Ignore cross-origin parents and fall back to the default volume.
    }

    return ''
}

export function getResolvedVolumeUrl()
{
    if (typeof window === 'undefined')
    {
        return defaultVolumePath
    }

    return (
        getVolumeUrlFromSearch(window.location.search) ??
        getVolumeUrlFromSearch(getParentSearch()) ??
        defaultVolumePath
    )
}

export default function createSources()
{
    return [
        {
            name: 'volume',
            type: 'niftiFile',
            path: getResolvedVolumeUrl(),
        },
    ]
}
